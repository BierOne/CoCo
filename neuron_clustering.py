"""
Create Time: 19/7/2022
Author: BierOne (lyibing112@gmail.com)
"""
import os
import torch
from torch import cat
from torch.nn import functional as F

from collections import OrderedDict
import itertools

import sys
from fast_pytorch_kmeans import KMeans


def generate_clustering_method(label_ids, clustering_method, multi_level, class_level, domain_level, reg_layer_names,
                               low_level_merge_method, high_level_merge_method, num_clusters=5, num_act_neuron=20,
                               random=False, quantile=0.01, act_ratio=0.3, use_umap=False, umap_dim=50
                               ):
    if reg_layer_names is None:
        reg_layer_names = []

    cluster_config = {
        "label_ids": label_ids,
        "reg_layer_list": reg_layer_names,
        "clustering_method": clustering_method,
        "low_level_merge_method": low_level_merge_method,
        "high_level_merge_method": high_level_merge_method,
        "domain_level": domain_level,
        "class_level": class_level,
        "multi_level": multi_level,
        "distance_metric": "euclidean",
        "num_concept_clusters": num_clusters,
        "max_cluster_size": num_act_neuron,
        "random": random,
        "quantile": quantile,
        "act_ratio": act_ratio,
        "use_umap": use_umap,
        "umap_dim": umap_dim
    }
    return cluster_config, NeuronClustering(**cluster_config)


def get_norm(layer_out, p=2):
    if layer_out.dim() > 2:
        layer_out = layer_out.view(*layer_out.shape[:2], -1)
        layer_out = layer_out.mean(dim=-1)  # avg pooling for conv map
    return layer_out / (layer_out.norm(p=p, dim=-1, keepdim=True))


def padding_neuron_cluster(hie_cluster, padding_v=1024, max_cluster_size=50):
    def _padding(cluster, padding_v, max_cluster_size):
        # print(cluster)
        cluster = cluster[:max_cluster_size]
        if len(cluster) < max_cluster_size:
            padding_size = max_cluster_size - len(cluster)
            cluster = F.pad(cluster, (0, padding_size), value=padding_v)
        return cluster

    for i, h_cluster in enumerate(hie_cluster):
        if isinstance(h_cluster, list):
            padding_neuron_cluster(h_cluster, padding_v)
        else:
            hie_cluster[i] = _padding(h_cluster, padding_v, max_cluster_size)




def duplicated_neuron_filtering(act_id_lists):
    def duplicated_neuron_filtering_single(act_id_lists, intersection, mask_list=None):
        for nid in intersection:  # accumulate masks for each overlapped neuron id
            if mask_list:
                mask_list = [mask | act_list.eq(nid) for mask, act_list in zip(mask_list, act_id_lists)]
            else:
                mask_list = [act_list.eq(nid) for act_list in act_id_lists]
        final_list = [act_list[~mask] for mask, act_list in zip(mask_list, act_id_lists)]
        return final_list

    if isinstance(act_id_lists[0], list):
        cls_level_acts = []
        for n_list in act_id_lists:
            if n_list:
                cls_level_acts.append(cat(n_list).unique())
    else:
        cls_level_acts = act_id_lists
    try:
        cat_list, counts = cat(cls_level_acts).unique(return_counts=True)
    except NotImplementedError:
        return act_id_lists
    intersection = cat_list[torch.where(counts.gt(len(cls_level_acts)/2))]
    # intersection = cat_list[torch.where(counts.gt(1))]
    print("filter {} number of overlapped neurons".format(len(intersection)))
    if len(intersection) == 0:
        return act_id_lists
    if isinstance(act_id_lists[0], list):
        final_list = []
        for n_list in act_id_lists:
            final_list.append(duplicated_neuron_filtering_single(n_list, intersection))
    else:
        final_list = duplicated_neuron_filtering_single(act_id_lists, intersection)
    return final_list


def merge_activation_groups(act_id_lists, method='intersection', remain_groups=False):
    def _intersection(act_id_lists):
        cat_list, counts = cat(act_id_lists).unique(return_counts=True)
        final_list = cat_list[torch.where(counts.gt(len(act_id_lists) - 1))]  # at least shown two times
        return final_list

    def _union(act_id_lists):
        final_list = cat(act_id_lists).unique()
        return final_list

    def _xor(act_id_lists):
        cat_list, counts = cat(act_id_lists).unique(return_counts=True)
        final_list = cat_list[torch.where(counts.eq(1))]  # only shown one time
        return final_list

    assert method in ['intersection', 'union', 'xor']
    if remain_groups:
        merged_list = duplicated_neuron_filtering(act_id_lists)
        return merged_list
    if method == 'intersection':
        merged_list = _intersection(act_id_lists)
    elif method == 'union':
        merged_list = _union(act_id_lists)
    elif method == 'xor':
        merged_list = _xor(act_id_lists)
    return merged_list


def merge_hie_activation_groups_by_jaccard(neuron_clusters, threshold=0.8):
    # merge cluster across domains/classes
    chined_clusters = list(itertools.chain(*neuron_clusters))
    final_clusters, skipped_idx = [], []
    for query_idx, query in enumerate(chined_clusters):
        if query_idx in skipped_idx:
            continue
        else:
            skipped_idx.append(query_idx)
        for key_idx, key in enumerate(chined_clusters):
            if key_idx in skipped_idx:
                continue
            cat_list, counts = cat([query, key]).unique(return_counts=True)
            sim = counts.gt(1).sum() / len(cat_list)  # intersection / union
            if sim > threshold:
                query = cat_list
                skipped_idx.append(key_idx)

        final_clusters.append(query)
    return final_clusters


def print_clustering_state(layer, clusters, total_n=0, times=10):
    print("state for layer", layer)
    act_neurons, counts = torch.unique(cat(clusters), return_counts=True)
    print("\tnumber of clusters:{}".format(len(clusters)))
    print("\tnumber of unactivated neurons:{}, total neurons:{}".format(total_n - len(act_neurons),
                                                                        total_n))
    print("\tneurons numbers with activation times > 10:{}".format(len(act_neurons[counts > times])))
    return total_n - len(act_neurons)


def get_thresh(layer_out, quantile):
    # num = round(len(layer_out) * quantile)
    # out_sort = layer_out.ravel().sort(descending=True)[0]
    layer_out = layer_out
    num = round(layer_out.shape[1] * quantile)
    out_sort = layer_out.mean(0).sort(descending=True)[0]
    return (out_sort[num:num + 1] + 1e-6)


class NeuronClustering:
    def __init__(self, label_ids, reg_layer_list, quantile, act_ratio,
                 clustering_method="kmeans",
                 low_level_merge_method="jaccard",
                 high_level_merge_method="jaccard",
                 domain_level=False, class_level=True, random=False,
                 multi_level=False, distance_metric="euclidean",
                 num_concept_clusters=30,
                 max_cluster_size=50, use_umap=True, umap_dim=50
                 ):
        assert clustering_method in ["kmeans", "topk"]
        assert low_level_merge_method in ["jaccard", "xor", "union", "intersection", "chain", None]
        assert high_level_merge_method in ["jaccard", "xor", "chain", "dup_filtering", None]
        # assert domain_level != class_level
        self.domain_level = domain_level
        self.multi_level = multi_level  # clustering at multi-level, e.g., high-level: domain then low-level: class
        self.clustering_method = clustering_method
        self.low_level_merge_method = low_level_merge_method
        self.high_level_merge_method = high_level_merge_method
        self.max_cluster_size = max_cluster_size
        self.num_concept_clusters = num_concept_clusters
        self.kmeans = KMeans(n_clusters=num_concept_clusters, mode=distance_metric, verbose=0)
        self.reg_layer_list = reg_layer_list
        self.layer_act_neuron_dict = OrderedDict({layer: [] for layer in self.reg_layer_list})
        self.layer_padding, self.layer_unact_neurons = None, None
        self.label_ids = label_ids
        self.quantile = quantile
        self.act_ratio = act_ratio
        self.random = random
        self.use_umap = use_umap
        if use_umap:
            import umap
            self.umap = umap.UMAP(random_state=42, n_components=umap_dim)

        # For concept evolution analysis
        self.previous_state = {}
        self.evol_ratio_dict = {}
        self.previous_act_n = None

    def update_target_layer(self, reg_layer_list):
        self.reg_layer_list = reg_layer_list
        self.layer_act_neuron_dict = OrderedDict({layer: [] for layer in reg_layer_list})


    def compute_act_weight(self, clusters, layer_out, norm=True):
        assert isinstance(clusters, list)
        cl_weights = []
        layer_out = layer_out.cuda()
        thresh = get_thresh(layer_out, self.quantile)
        layer_out = layer_out.T

        for cl in clusters:
            stimuli = torch.where(layer_out[cl] > thresh, 1, 0)
            concept_sti = stimuli.sum(dim=0).ge(1).float()  # union all neuron stimuli
            weights = (stimuli.sum(dim=1) / (concept_sti.sum() + 1e-8)).cpu()
            if norm:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            cl_weights.append(weights)
        return cl_weights


    def compute_concept_evo(self, layer_out, clusters=None):
        # For concept evolution analysis
        for quantile in [0.01, 0.1, 0.3, 0.5, 0.8]:
            thresh = get_thresh(layer_out, quantile)
            print("Current Quantile threshold:")
            print(quantile, thresh)
            continue

            new_stimuli = (layer_out > thresh).T
            if quantile not in self.previous_state:
                if clusters is not None:
                    act_neurons = torch.unique(cat(clusters))
                    new_stimuli = new_stimuli[act_neurons]
                    self.previous_act_n = act_neurons
                self.previous_state[quantile] = new_stimuli
                self.evol_ratio_dict[quantile] = 0.0
                continue
            else:
                if self.previous_act_n is not None:
                    new_stimuli = new_stimuli[self.previous_act_n]
                # overlap = (self.previous_state[quantile] & new_stimuli).long().sum(1)
                # union = (self.previous_state[quantile] | new_stimuli).long().sum(1)
                # self.evol_ratio_dict[quantile] = 1 - ((overlap+1e-6)/(union+1e-6)).mean().item()
                num_sample = new_stimuli.shape[1]
                flipping = (self.previous_state[quantile] ^ new_stimuli).long().sum(1)
                self.evol_ratio_dict[quantile] = (flipping/num_sample).mean().item()

                self.previous_state[quantile] = new_stimuli
            print("Current Evo Ratio:", quantile, self.evol_ratio_dict[quantile])


    def compute_act_neurons(self, layer_out, accs=None):
        """
        # get top occured neurons in the given layer
        :param class_layer_out: [b, n] or [b, n, h, w]
        :param accs: accuracy for each entry
        :return: tensor of neuron indexes [n1, n2, n3]
        """
        thresh = get_thresh(layer_out, self.quantile)
        times = round(layer_out.shape[0] * self.act_ratio)
        if self.clustering_method == 'topk':
            act_counts = (layer_out > thresh)
            # we only want the concept based on the correct classification
            if accs is not None:
                act_counts = act_counts * accs.unsqueeze(1)  # [b, n]
            # neuron_clusters = [act_counts.sum(0).topk(k=self.max_cluster_size)[1]]
            neuron_clusters = [torch.where(act_counts.sum(0) > times)[0]]
        elif self.clustering_method == 'kmeans':
            stimuli = torch.where(layer_out > thresh, 1., 0.).T
            if accs is not None:
                stimuli = (stimuli * accs.unsqueeze(0))
            # print(thresh*num_q, thresh, num_q)
            filtered_idx = torch.where(stimuli.sum(1) > times)[0]
            stimuli = stimuli[filtered_idx]

            if len(stimuli) > self.num_concept_clusters:
                c_labels = self.kmeans.fit_predict(stimuli.cuda()).cpu()
                neuron_clusters = [filtered_idx[c_labels == c].cpu()
                                   for c in range(self.num_concept_clusters)]
            else:
                neuron_clusters = list(filtered_idx.split(1))
        else:
            raise Exception("unknown clustering method", self.clustering_method)
        neuron_clusters = [c for c in neuron_clusters if len(c) > 0]
        return neuron_clusters

    def merge(self, clusters, merge_method=None, remain_group=False):
        if merge_method in ["xor", "union", "intersection"]:
            clusters = merge_activation_groups(clusters, merge_method, remain_group)
        elif merge_method == "jaccard":  # kmeans then merge overlapped cluster
            clusters = merge_hie_activation_groups_by_jaccard(clusters)
        elif merge_method == "chain":
            # [[(1,3), (2,3)], [(4,5)]] -> [(1, 3), (2, 3), (4, 5)]
            clusters = list(itertools.chain(*clusters))
        elif merge_method == "dup_filtering":
            clusters = duplicated_neuron_filtering(clusters)
            clusters = list(itertools.chain(*clusters))
            # clusters = merge_hie_activation_groups_by_jaccard(clusters)
        return clusters

    def clutering_then_merge(self, layer_out_list, accs_list, merge_method):
        final_clusters = []
        for l_out, accs in zip(layer_out_list, accs_list):
            # fine-grained concept clusters (topk or kmeans)
            final_clusters.append(self.compute_act_neurons(l_out, accs))
        return self.merge(final_clusters, merge_method)

    def compute_unactivated_neurons(self, padding=True):
        self.layer_unact_neurons = {}
        for layer_name in self.reg_layer_list:
            padding_idx = self.layer_padding[layer_name]
            activated_nid = torch.stack(self.layer_act_neuron_dict[layer_name][0], dim=0).ravel().unique()

            ones = torch.ones(padding_idx + 1)
            ones[activated_nid] = 0
            ind = torch.where(ones == 1)[0]
            if padding:
                padding_size = padding_idx - len(ind)
                ind = F.pad(ind, (0, padding_size), value=padding_idx)
            self.layer_unact_neurons[layer_name] = ind
            # print(layer_name, padding_idx, ind.shape)


    def compute_class_level_neuron_cluster(self, d_layer_out, padding=True, frac=0.2):
        # compute class-level neuron groups
        # d_layer_out = [
        #     {
        #         "layer_name": layer_out,
        #         "accs": d_accs,
        #         "labels": d_labels
        #     },
        # ]
        try:
            accs = [d["accs"] for d in d_layer_out]
        except KeyError:
            accs = [None] * len(d_layer_out)
        labels = [d["labels"] for d in d_layer_out]
        cluster_weights = None
        self.layer_padding = {}
        for layer_name in self.reg_layer_list:
            layer_out = [d[layer_name] for d in d_layer_out]
            padding_idx = layer_out[0].shape[1]
            cluster_list = []
            # print("processing class #: ", self.label_ids)
            for l in self.label_ids:
                class_layer_out, class_accs = [], []
                for d_labels, d_accs, l_out in zip(labels, accs, layer_out):
                    class_ind = (d_labels == l)
                    if class_ind.sum() > 0:
                        class_layer_out.append(l_out[class_ind])
                        if d_accs is not None:
                            class_accs.append(d_accs[class_ind])
                        else:
                            class_accs.append(None)

                if not self.multi_level:
                    # get class-level top activation idx from the overall stat
                    class_layer_out = [cat(class_layer_out)]
                    if d_accs is not None:
                        class_accs = [cat(class_accs)]
                    else:
                        class_accs = [None]

                clusters = self.clutering_then_merge(class_layer_out, class_accs,
                                                     self.low_level_merge_method)
                cluster_list.append(clusters)

            # finally we merge overlapped clusters across classes
            final_clusters = self.merge(cluster_list, self.high_level_merge_method)
            final_clusters = sorted(final_clusters, key=lambda c: c.tolist())

            if self.high_level_merge_method:
                cluster_weights = self.compute_act_weight(final_clusters, cat(layer_out))
                # self.compute_concept_evo(cat(layer_out), final_clusters)

            # Random Clustering
            if self.random:
                print("performing random clustering")
                # print(final_clusters[:1], final_clusters[-1:])
                t = torch.randperm(padding_idx)
                ab_clusters = []
                for c in final_clusters:
                    ab_clusters.append(t[c])
                final_clusters = ab_clusters
                # print(final_clusters[:1], final_clusters[-1:])

            # if self.high_level_merge_method:
                # self.num_unact = print_clustering_state(layer_name, final_clusters, padding_idx)

            if padding:
                padding_neuron_cluster(final_clusters, padding_idx, self.max_cluster_size)
                padding_neuron_cluster(cluster_weights, 0, self.max_cluster_size)
            self.layer_act_neuron_dict[layer_name] = (final_clusters, cluster_weights)
            self.layer_padding[layer_name] = padding_idx


    def compute_unsupervised_neuron_cluster(self, d_layer_out, padding=True, frac=0.2):
        # compute class-level neuron groups
        # d_layer_out = [
        #     {
        #         "layer_name": layer_out,
        #         "accs": d_accs,
        #         "labels": d_labels
        #     },
        # ]
        try:
            accs = [d["accs"] for d in d_layer_out]
        except KeyError:
            accs = [None] * len(d_layer_out)
        labels = [d["labels"] for d in d_layer_out]
        cluster_weights = None
        self.layer_padding = {}
        for layer_name in self.reg_layer_list:
            layer_out = [d[layer_name] for d in d_layer_out]
            padding_idx = layer_out[0].shape[1]
            print("processing class #: ", self.label_ids)
            final_clusters = self.compute_act_neurons(cat(layer_out), cat(accs))
            final_clusters = sorted(final_clusters, key=lambda c: c.tolist())

            if self.high_level_merge_method:
                cluster_weights = self.compute_act_weight(final_clusters, cat(layer_out))

            if self.high_level_merge_method:
                self.num_unact = print_clustering_state(layer_name, final_clusters, padding_idx)

            if padding:
                padding_neuron_cluster(final_clusters, padding_idx, self.max_cluster_size)
                padding_neuron_cluster(cluster_weights, 0, self.max_cluster_size)
            self.layer_act_neuron_dict[layer_name] = (final_clusters, cluster_weights)
            self.layer_padding[layer_name] = padding_idx


    def compute_domain_level_neuron_cluster(self, d_layer_out, padding=True):
        raise NotImplementedError

    def compute_neuron_cluster(self, d_layer_out, padding=True, unact=False, unsupervised=False):
        if unsupervised:
            self.compute_unsupervised_neuron_cluster(d_layer_out, padding)
            if unact:
                self.compute_unactivated_neurons(padding)
            return
        if self.domain_level:
            self.compute_domain_level_neuron_cluster(d_layer_out, padding)
        else:
            self.compute_class_level_neuron_cluster(d_layer_out, padding)

        if unact:
            self.compute_unactivated_neurons(padding)
