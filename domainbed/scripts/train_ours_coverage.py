# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse, shutil
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import h5py
from neuron_clustering import generate_clustering_method, NeuronClustering, get_norm
from domainbed.cls_config import clustering_param

Ours_ALG = ['CoCo_SelfReg', 'CoCo_CondCAD']


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_layer_feat(step, feat_output, labels, accs, layer_name="avg_pool",
                    dir="./visualizations/layer_state", fmt='E{:04d}_{}_'):
    create_dir(dir)
    file_path = os.path.join(dir, fmt.format(step, layer_name) + "state.h5")
    with h5py.File(file_path, mode='w', libver='latest') as f:
        f.create_dataset('feat', data=feat_output.numpy())
        f.create_dataset('labels', data=labels.numpy())
        f.create_dataset('accs', data=accs.numpy())


def accuracy(network, loader, weights=None, get_feat=False):
    network.eval()
    layer_output, labels, accs = [], [], []
    weights_offset, correct_n, total = 0, 0, 0
    with torch.no_grad():
        torch.cuda.empty_cache()
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            p, feat = network.extract_feat(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)

            if p.size(1) == 1:
                correct = (p.gt(0).eq(y) * batch_weights.view(-1, 1))
            else:
                correct = (p.argmax(1).eq(y) * batch_weights)
            correct_n += correct.sum().item()
            if get_feat:
                accs.append(correct.detach().cpu())
                labels.append(y.detach().cpu())
                layer_output.append(get_norm(feat).detach().cpu())
            total += batch_weights.sum().item()

    network.train()
    if get_feat:
        layer_output = torch.cat(layer_output, dim=0)
        labels = torch.cat(labels, dim=0)
        accs = torch.cat(accs, dim=0)
        return correct_n / total, layer_output, labels, accs
    else:
        return correct_n / total


from collections.abc import Iterable
from torch import cat


def set_freeze_by_layer_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    if hasattr(model, "module"):
        iter_modules = model.module.named_children()
    else:
        iter_modules = model.named_children()
    for name, child in iter_modules:
        if name not in layer_names:
            continue
        # print("set freeze for layer {} as: {}".format(name, freeze))
        for param in child.parameters():
            param.requires_grad = not freeze
            param.grad = None  # in case of numerical accumulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--visualization', "-vs", action='store_true')
    parser.add_argument('--save_for_ft', action='store_true')
    parser.add_argument('--backbone', type=str, default="ERM")
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--multi_level', type=bool, default=True, help="hierarchical clustering")
    parser.add_argument('--clustering', "-c", type=str, default="kmeans", help="topk | kmeans")
    parser.add_argument('--resnet18', type=bool, default=False)
    parser.add_argument('--default_cluster', action='store_true')
    parser.add_argument('--ratio', type=float, default=0, help="only utilizing part of large dataset for neuron clustering")
    args = parser.parse_args()
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    larger_batch = args.dataset in ['OfficeHome', 'DomainNet']

    start_step = 0
    algorithm_dict = None

    json_f = os.path.join(args.output_dir, 'results.jsonl')
    if os.path.exists(json_f):
        print("remove", json_f)
        os.remove(json_f)

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset,
                                                   args=args,
                                                   larger_batch=larger_batch)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed),
                                                  args=args, larger_batch=larger_batch)

    if args.hparams:
        hparams.update(json.loads(args.hparams))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    setup_args = ['resnet18', 'lr', 'ratio']
    for p in setup_args:
        given_value = getattr(args, p)
        if given_value:
            print("update param from args: ", p, given_value)
            hparams[p] = given_value

    ##################### Load Clustering Method ####################
    cluster_steps = clustering_param[args.dataset]["cluster_steps"]
    cluster_config = clustering_param[args.dataset][args.clustering]
    cluster_config['label_ids'] = range(dataset.num_classes)

    if not args.default_cluster:
        cluster_config['max_cluster_size'] = hparams['cluster_size']
        cluster_config['quantile'] = hparams['quantile']
        cluster_config['act_ratio'] = hparams['act_ratio']
        cluster_config['num_concept_clusters'] = hparams['cluster_num']

    hparams['cluster_num'] = dataset.num_classes * cluster_config['num_concept_clusters']
    if args.multi_level:
        hparams['cluster_num'] *= len(dataset)
    NACT = NeuronClustering(**cluster_config)

    # with open(os.path.join(args.output_dir, 'cls_config.json'), 'w') as f:
    #     json.dump(cluster_config, f)

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    print('Cluster_Config:')
    print('Cluster Steps: ', cluster_steps)
    for k, v in sorted(cluster_config.items()):
        print('\t{}: {}'.format(k, v))

    ##################### Process Dataset ####################
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []

    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
                                      int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_) * args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    if larger_batch:
        eval_batch_size = 128
    else:
        eval_batch_size = 64


    #######  Use FastDataloader in large dataset would take too much memory
    if args.dataset in ['DomainNet']:
        loader_clss = FastDataLoader
        works = 2
        tr_eval_loaders = []
    else:
        loader_clss = FastDataLoader
        works = 6
        tr_eval_loaders = [loader_clss(
            dataset=env,
            batch_size=eval_batch_size,
            num_workers=works,)
            for i, (env, _) in enumerate(in_splits)
            if i not in args.test_envs]

    te_eval_loaders = [loader_clss(
        dataset=env,
        batch_size=eval_batch_size,
        num_workers=works,)
        for i, (env, _) in enumerate(in_splits)
        if i in args.test_envs]

    out_eval_loaders = [loader_clss(
        dataset=env,
        batch_size=eval_batch_size,
        num_workers=works,)
        for (env, _) in out_splits + uda_splits]


    if args.dataset in ["DomainNet"]:
        eval_loaders = te_eval_loaders + out_eval_loaders
        eval_loader_names = []
    else:
        eval_loaders = tr_eval_loaders + te_eval_loaders + out_eval_loaders
        eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))
                             if i not in args.test_envs]

    eval_loader_names += ['env{}_in'.format(i) for i in args.test_envs]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i) for i in range(len(uda_splits))]
    eval_weights = [None] * len(eval_loaders)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict['model_dict'], strict=False)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ


    def save_checkpoint(filename="model.pkl", output_dir=args.output_dir):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict(),
            "neuron_clusters": algorithm.c_clusters
        }
        torch.save(save_dict, os.path.join(output_dir, filename))


    def set_clusters(NACT, algorithm, d_layer_output, update=True):
        NACT.compute_neuron_cluster(d_layer_output, padding=True, unact=hparams['include_unact'])
        if update:
            algorithm.c_clusters = {l: torch.stack(v[0], dim=0) for l, v in NACT.layer_act_neuron_dict.items()}
            if hparams['use_c_weights']:
                algorithm.c_weights = {l: torch.stack(v[1], dim=0) for l, v in NACT.layer_act_neuron_dict.items()}
            if hparams['include_unact']:
                algorithm.unact_neurons = {l: v.cuda() for l, v in NACT.layer_unact_neurons.items()}


    def create_cls_splits(in_splits):
        cls_in_split = []
        for env_i, (in_, _) in enumerate(in_splits):
            if env_i not in args.test_envs:
                cls_data, _ = misc.split_dataset(in_,
                                                 int(len(in_) * hparams['ratio']),
                                                 random.randint(0, 50))
                cls_in_split.append(cls_data)
        cls_loaders = [FastDataLoader(
                        dataset=env,
                        batch_size=128,
                        num_workers=2,) for env in cls_in_split]
        return cls_loaders



    def average_by_filtering(result_dict, cond_func):
        values = [v for k, v in result_dict.items() if cond_func(k)]
        return np.mean(values).item()

    def get_clsustering_data(d_layer_output, loader, args):
        # if args.dataset in ['DomainNet']:
        #     loader.dataset.set_clustering_samples_on(ratio=hparams['ratio'], min_num=hparams['min_num'])
        acc, feat, labels, accs = accuracy(algorithm, loader, get_feat=True)
        # print(feat.shape)
        d_layer_output.append({
            "layer4": feat,
            "accs": accs,
            "labels": labels,
        })
        # if args.dataset in ['DomainNet']:
        #     loader.dataset.set_clustering_samples_off()
        return d_layer_output, acc

    vs_dir = "~/dg/visualizations/layer_state/{}/test_env_{}/{}".format(args.dataset, args.test_envs[0], args.algorithm)
    vs_dir = os.path.expanduser(vs_dir)
    create_dir(vs_dir)


    last_results_keys = None

    weights_offset = 0
    coco_start_step = cluster_step = cluster_steps.pop(0)
    coco_start = False # whether you use coco-based loss

    if (args.algorithm in Ours_ALG) and args.dataset in ["DomainNet"]:
        cls_loaders = create_cls_splits(in_splits)
    else:
        cls_loaders = tr_eval_loaders


    for step in range(start_step, n_steps):
        if step > (coco_start_step + 3000):
            # maximum training steps
            break

        if ((step % checkpoint_freq == 0)) or (step == n_steps - 1):
            d_layer_output = []
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
                'loss': 100,  # dummy loss
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val).item()

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                if (loader in tr_eval_loaders) and (args.dataset not in ["DomainNet"]):
                    d_layer_output, acc = get_clsustering_data(d_layer_output, loader, args)
                else:
                    acc = accuracy(algorithm, loader, weights)
                results[name + '_acc'] = acc
                torch.cuda.empty_cache()

            # results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)
            summary_results = {
                    'in_acc_src': average_by_filtering(results,
                                                       lambda k: 'in_acc' in k and k not in [f'env{env_i}_in_acc' for
                                                                                             env_i
                                                                                             in args.test_envs]),
                    'out_acc_src': average_by_filtering(results,
                                                        lambda k: 'out_acc' in k and k not in [f'env{env_i}_out_acc' for
                                                                                               env_i in
                                                                                               args.test_envs]),
                    f'in_acc_tgt{args.test_envs[0]}': average_by_filtering(results,
                                                       lambda k: k in [f'env{env_i}_in_acc' for env_i in
                                                                       args.test_envs]),
                    f'out_acc_tgt{args.test_envs[0]}': average_by_filtering(results,
                                                        lambda k: k in [f'env{env_i}_out_acc' for env_i in
                                                                        args.test_envs]),
                    # 'step_time': results['step_time'],
                    'step': results['step'],
                    'loss': results['bn_loss'] if 'CondCAD' in args.algorithm else results['loss'],
            }
            results_keys = sorted(summary_results.keys())
            misc.print_row(results_keys, colwidth=12)
            misc.print_row([summary_results[key] for key in results_keys],
                           colwidth=12)


            if (args.algorithm in Ours_ALG) and (step == cluster_step):
                print('#' * 10, "Step:{}, Updating neuron group...".format(step), '#' * 10)
                cluster_step = cluster_steps.pop(0)
                set_clusters(NACT, algorithm, d_layer_output, update=True)
                coco_start = True
                print('#' * 10, "Finished! Neuron group updated.".format(step), '#' * 10)
            else:
                set_clusters(NACT, algorithm, d_layer_output, update=False)

            results.update({
                # 'hparams': hparams,
                'args': vars(args),
            })
            with open(json_f, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")


            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

            if args.visualization:
                fmt = 'S{:04d}_{}_state.h5'
                file_path = os.path.join(vs_dir, fmt.format(step, "avg_pool"))
                with h5py.File(file_path, mode='w', libver='latest') as f:
                    evals = zip(eval_loader_names, tr_eval_loaders, eval_weights)
                    for name, loader, weights in evals:
                        grp = f.create_group("train_"+name)
                        _, feat, labels, accs = accuracy(algorithm, loader, get_feat=True)
                        grp.create_dataset('feat', data=feat)
                        grp.create_dataset('labels', data=labels)
                        grp.create_dataset('accs', data=accs)
                        if algorithm.c_clusters is not None:
                            grp.create_dataset('c_clusters', data=algorithm.c_clusters['layer4'].numpy())


        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in next(train_minibatches_iterator)]
        uda_device = None
        if coco_start:
            step_vals = algorithm.update_cl(minibatches_device, uda_device)
        else:
            step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
    # save_checkpoint('model.pkl')



    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
    torch.cuda.empty_cache()
    exit(0)