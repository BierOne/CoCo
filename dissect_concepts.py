"""
Create Time: 31/7/2022
Author: BierOne (lyibing112@gmail.com)
"""

import sys, pickle

import numpy as np
import torch, torchvision, argparse, os, shutil, inspect, json, numpy, random
from collections import defaultdict
from easydict import EasyDict as edict

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from neuron_clustering import generate_clustering_method

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["ytick.labelleft"] = False
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['savefig.facecolor'] = 'white'

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False

plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False

def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa('--steps', type=int, default=0,
       help='Number of steps for the checkpoint; if None, then dissect all checkpoints')
    aa('--data_dir', type=str, default="/dg/data/domainbed")
    aa('--layer_name', type=str, default="featurizer.network.layer4")
    aa('--model_dir', type=str, default="results/resnet50/PACS_ERM_0")
    aa('--clustering', "-c", type=str, default="topk", help="topk | kmeans")
    aa("--load_cache", "-l", action='store_true')
    aa('--seg', choices=['net', 'netp', 'netq', 'netpq', 'netpqc', 'netpqxc'],
       default='netpqc')
    aa('--quantile', type=float, default=0.01)
    aa('--miniou', type=float, default=0.04)
    aa('--thumbsize', type=int, default=100)
    aa('--random', action='store_true')
    aa('--original', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    layername = args.layer_name

    ################### --- Load Data --- ###################
    name = "model_step{}.pkl".format(args.steps) if args.steps >=0 else "model.pkl"
    model_path = os.path.join(args.model_dir, name)

    logs = torch.load(model_path)
    hparams = logs['model_hparams']
    m_args = edict(logs['args'])

    print('Loaded Model Args from {}:'.format(model_path))
    for k, v in sorted(vars(m_args).items()):
        print('\t{}: {}'.format(k, v))

    dataset = vars(datasets)[m_args.dataset](m_args.data_dir,
                                             m_args.test_envs, hparams)
    classes = dataset.datasets[0].classes
    ################### --- Neuron Clustering --- ###################
    if args.load_cache and "neuron_clusters" in logs:
        act_neuron_dict = list(logs["neuron_clusters"])
        print(act_neuron_dict[:5])
    else:
        in_splits = []
        out_splits = []
        for env_i, env in enumerate(dataset):
            out, in_ = misc.split_dataset(env,
                                          int(len(env) * m_args.holdout_fraction),
                                          misc.seed_hash(m_args.trial_seed, env_i))
            if env_i in m_args.test_envs:
                _, in_ = misc.split_dataset(in_,
                                            int(len(in_) * m_args.uda_holdout_fraction),
                                            misc.seed_hash(m_args.trial_seed, env_i))
            if hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(in_)
                out_weights = misc.make_weights_for_balanced_classes(out)
            else:
                in_weights, out_weights, uda_weights = None, None, None
            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))

        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=8)
            for i, (env, _) in enumerate(in_splits) if i not in m_args.test_envs]

        algorithm_class = algorithms.get_algorithm_class(m_args.algorithm)
        model = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(m_args.test_envs), hparams)
        model.load_state_dict(logs['model_dict'])
        model = model.cuda().eval()

        if args.clustering == 'kmeans':
            cluster_config, NACT = generate_clustering_method(clustering_method='kmeans',
                                                              multi_level=True,
                                                              class_level=True,
                                                              domain_level=False,
                                                              low_level_merge_method="jaccard",
                                                              high_level_merge_method="jaccard",
                                                              label_ids=range(dataset.num_classes),
                                                              reg_layer_names=[layername],
                                                              num_clusters=5,
                                                              random=args.random,
                                                              quantile=0.03, act_ratio=0.3,
                                                              num_act_neuron=200,
                                                              )
        elif args.clustering == 'topk':
            cluster_config, NACT = generate_clustering_method(clustering_method='topk',
                                                              multi_level=True,
                                                              class_level=True,
                                                              domain_level=False,
                                                              low_level_merge_method="chain",
                                                              high_level_merge_method=None,
                                                              label_ids=range(dataset.num_classes),
                                                              reg_layer_names=[layername],
                                                              num_act_neuron=10,
                                                              random=args.random,
                                                              )
        else:
            raise Exception("unknown type of clustering method:", args.clustering)

        print("clusters are not available, we need run clustering method...")
        d_layer_output = []
        for did, loader in enumerate(eval_loaders):
            feat, labels, accs = get_feat(model, loader)
            d_layer_output.append({
                layername: feat,
                "accs": accs,
                "labels": labels,
            })
        NACT.compute_neuron_cluster(d_layer_output, padding=False)
        logs["nograd_neuron_clusters"] = act_neuron_dict = NACT.layer_act_neuron_dict
        # torch.save(logs, model_path)
        print("############## store neuron clusters to logs #############")
    ################### --- Load Neuron Activation Maps --- ###################
    resdir = '~/dg/visualizations/neuron_state/%s/%s/env%s' % (m_args.dataset, m_args.algorithm, m_args.test_envs[0])
    if args.steps >=0:
        resdir += '-s%d' % (args.steps)
    if args.layer_name is not None:
        resdir += '-' + args.layer_name.split('.')[-1]
    if args.quantile != 0.005:
        resdir += ('-%g' % (args.quantile * 1000))
    if args.thumbsize != 100:
        resdir += ('-t%d' % (args.thumbsize))
    if args.original:
        resdir = os.path.expanduser(os.path.join(resdir, 'org_image'))
    else:
        resdir = os.path.expanduser(os.path.join(resdir, 'image'))

    print("load neuron vis from: ", resdir)

    if args.random:
        outdir = '~/dg/visualizations/concept_state/rand_%s/%s/env%s' % (m_args.algorithm, m_args.dataset, m_args.test_envs[0])
    else:
        outdir = '~/dg/visualizations/concept_state/%s/%s/env%s' % (m_args.algorithm, m_args.dataset, m_args.test_envs[0])
    if args.steps >=0:
        outdir += '-s%d' % (args.steps)
    if args.layer_name is not None:
        outdir += '-' + args.layer_name.split('.')[-1]
    if args.clustering is not None:
        outdir += '-' + args.clustering
    outdir = os.path.expanduser(outdir)
    if os.path.isdir(outdir):
        print("remove the old output dir:", outdir)
        shutil.rmtree(outdir)

    plot_cluster(act_neuron_dict, resdir, outdir)
    Pool.join()

    print('Done!')


sys.path.append(os.path.expanduser("~/dg/dissect"))
from netdissect.workerpool import WorkerBase, WorkerPool


class SaveImageWorker(WorkerBase):
    def work(self, cid, cluster, resdir, data_dir, cols=1):
        nids = [nid for nid in cluster if nid != 2048]
        rows = 60 if len(nids) > 60 else len(nids)
        fig = plt.figure(figsize=(12 * cols, 1.5 * rows))
        i = 0
        for nid in nids:
            act_path = os.path.join(resdir, "unit{}.jpg".format(nid))
            if i == rows or (not os.path.exists(act_path)):
                print("out from: ", act_path)
                break
            i += 1
            img = Image.open(act_path)
            fig.add_subplot(rows, cols, i)

            plt.imshow(img)
            plt.title('neuron id:{}'.format(nid))
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'cluster{}_size{}.png'.format(cid, len(nids))))
        plt.close(fig)


Pool = WorkerPool(worker=SaveImageWorker)


def plot_cluster(clusters, resdir, outdir):
    if isinstance(clusters, dict):
        c_dict = [(k, v[0]) for k, v in clusters.items()]
    elif isinstance(clusters, list):
        if isinstance(clusters[0], list):
            c_dict = [(k, v) for k, v in enumerate(clusters)]
        else:
            c_dict = [(0, v) for v in clusters]
    else:
        raise Exception("unexpected type of cluster:", type(clusters))

    for cid, (key, cluster) in tqdm(enumerate(c_dict), total=len(c_dict)):
        data_dir = os.path.join(outdir, str(key))
        os.makedirs(data_dir, exist_ok=True)
        if not isinstance(cluster, (torch.Tensor, np.ndarray)):
            # in case of multi-level clusters
            plot_cluster(cluster, resdir, data_dir)
        else:
            Pool.add(cid, cluster, resdir, data_dir)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

from torch import nn

def get_feat(network, loader):
    softmax = nn.Softmax(dim=-1)
    network.eval()
    layer_output, labels, accs = [], [], []
    # with torch.no_grad():
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        p, feat = network.extract_feat(x)
        if p.size(1) == 1:
            correct = (p.gt(0).eq(y))
        else:
            correct = (p.argmax(1).eq(y))
        # gt_logits = p.gather(dim=-1, index=y.unsqueeze(1))
        # gradcam = torch.autograd.grad(gt_logits.sum(), feat,
        #                               create_graph=False, retain_graph=False)[0]
        # feat = feat * gradcam
        accs.append(correct.detach().cpu())
        labels.append(y.detach().cpu())
        # labels.append(p.argmax(1).detach().cpu())  # we use model prediction
        layer_output.append(feat.detach().cpu())
    layer_output = torch.cat(layer_output, dim=0)
    labels = torch.cat(labels, dim=0)
    accs = torch.cat(accs, dim=0)
    return layer_output, labels, accs


if __name__ == '__main__':
    main()
