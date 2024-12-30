# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
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
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import h5py


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_layer_feat(step, feat_output, labels, accs, layer_name="avg_pool",
                    dir="./visualizations/layer_state", fmt='E{:04d}_{}_'):
    create_dir(dir)
    file_path = os.path.join(dir, fmt.format(step, layer_name) + "state.h5")
    with h5py.File(file_path, mode='w', libver='latest') as f:
        f.create_dataset('feat', data=feat_output.numpy())
        f.create_dataset('labels', data=labels.numpy())
        f.create_dataset('accs', data=accs.numpy())


def get_feat(network, loader, device):
    network.eval()
    total_ccp_output, total_layer_output, domain_ids, labels, accs = [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p, feat = network.extract_feat(x)
            if p.size(1) == 1:
                correct = (p.gt(0).eq(y))
            else:
                correct = (p.argmax(1).eq(y))
            accs.append(correct.detach().cpu())
            labels.append(y.detach().cpu())
            total_layer_output.append(feat.detach().cpu())
    total_layer_output = torch.cat(total_layer_output, dim=0)
    labels = torch.cat(labels, dim=0)
    accs = torch.cat(accs, dim=0)
    network.train()
    return total_layer_output.numpy(), labels.numpy(), accs.numpy()


from collections.abc import Iterable
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
    parser.add_argument("--gpu", "-g", type=str, default='0', help="GPU No.")
    parser.add_argument('--visualization', "-vs", action='store_true')
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
    # parser.add_argument('--test_envs', type=str, default="train_output")
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--resnet18', type=bool, default=False)
    parser.add_argument('--no-pretrain', action='store_true')
    parser.add_argument('--freeze', action='store_true')

    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

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
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.resnet18:
        hparams['resnet18'] = True

    if args.no_pretrain:
        hparams['pretrained'] = False



    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

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

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS,)
        for i, (env, _) in enumerate(in_splits)
        if i in args.test_envs]

    eval_loaders += [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS,)
        for (env, _) in out_splits + uda_splits]

    eval_weights = [None] * len(eval_loaders)
    eval_loader_names = ['env{}_in'.format(i) for i in args.test_envs]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i) for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    print("set featurizer freeze state: ", args.freeze)
    set_freeze_by_layer_names(algorithm, "featurizer", freeze=args.freeze)
    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ


    def save_checkpoint(filename, output_dir=args.output_dir,
                        tgt_acc_best=0.0, out_train_acc_dict=None, out_test_acc_dict=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict(),
            "test_acc": tgt_acc_best,
            "out_train_acc_dict": out_train_acc_dict,
            "out_test_acc_dict": out_test_acc_dict,
        }
        torch.save(save_dict, os.path.join(output_dir, filename))


    vs_dir = "~/dg/visualizations/layer_state/{}/test_env_{}/{}".format(args.dataset, args.test_envs[0], args.algorithm)
    vs_dir = os.path.expanduser(vs_dir)
    create_dir(vs_dir)


    def average_by_filtering(result_dict, cond_func):
        values = [v for k, v in result_dict.items() if cond_func(k)]
        return np.mean(values).item()


    def subdict_by_filtering(result_dict, names):
        sub_dict = {k: result_dict[k] for k in result_dict if k in names}
        return sub_dict


    last_results_keys = None
    val_best = 0
    test_best = 0

    out_train_names = ['env{}_out_acc'.format(i) for i in range(len(out_splits)) if i not in args.test_envs]
    out_test_names = ['env{}_out_acc'.format(i) for i in range(len(out_splits)) if i in args.test_envs]

    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                          for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name + '_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            summary_results = {
                'in_acc_src': average_by_filtering(results,
                                                   lambda k: 'in_acc' in k and k not in [f'env{env_i}_in_acc' for
                                                                                         env_i
                                                                                         in args.test_envs]),
                'out_acc_src': average_by_filtering(results,
                                                    lambda k: 'out_acc' in k and k not in [f'env{env_i}_out_acc' for
                                                                                           env_i in
                                                                                           args.test_envs]),
                'in_acc_tgt': average_by_filtering(results,
                                                                       lambda k: k in [f'env{env_i}_in_acc' for env_i in
                                                                                       args.test_envs]),
                'out_acc_tgt': average_by_filtering(results,
                                                                        lambda k: k in [f'env{env_i}_out_acc' for env_i
                                                                                        in
                                                                                        args.test_envs]),
            }

            if summary_results[f'out_acc_tgt'] > test_best:
                test_best = summary_results[f'out_acc_tgt']
                print("save best with test out acc (oracle)", test_best)
                save_checkpoint('model_best_oracle.pkl',
                                tgt_acc_best=summary_results[f'in_acc_tgt'],
                                out_test_acc_dict=subdict_by_filtering(results, out_test_names),
                                out_train_acc_dict=subdict_by_filtering(results, out_train_names),
                                )

            if summary_results['out_acc_src'] > val_best:
                val_best = summary_results['out_acc_src']
                print("save best with val acc", val_best)
                save_checkpoint('model_best_val.pkl',
                                tgt_acc_best=summary_results[f'in_acc_tgt'],
                                out_test_acc_dict=subdict_by_filtering(results, out_test_names),
                                out_train_acc_dict=subdict_by_filtering(results, out_train_names),
                                )

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl',
                                tgt_acc_best=summary_results[f'in_acc_tgt'],
                                out_test_acc_dict=subdict_by_filtering(results, out_test_names),
                                out_train_acc_dict=subdict_by_filtering(results, out_train_names),
                                )

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.visualization:
                fmt = 'S{:04d}_{}_state.h5'
                file_path = os.path.join(vs_dir, fmt.format(step, "avg_pool"))
                with h5py.File(file_path, mode='w', libver='latest') as f:
                    evals = zip(eval_loader_names, eval_loaders, eval_weights)
                    for name, loader, weights in evals:
                        grp = f.create_group(name)
                        feat, labels, accs = get_feat(algorithm, loader, device)
                        grp.create_dataset('feat', data=feat)
                        grp.create_dataset('labels', data=labels)
                        grp.create_dataset('accs', data=accs)

    # save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
