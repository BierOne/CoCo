# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Example usage:
python -u -m domainbed.scripts.list_top_hparams \
    --input_dir domainbed/misc/test_sweep_data --algorithm ERM \
    --dataset VLCS --test_env 0
"""

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
from os import path
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings

SELECTION_METHODS = [
    model_selection.IIDAccuracySelectionMethod,
    # model_selection.LeaveOneOutSelectionMethod,
    # model_selection.OracleSelectionMethod,
]

def todo_rename(records, selection_method, latex):

    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm", 
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)




if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--test_env', type=int, default=5)
    args = parser.parse_args()

    records = reporting.load_records(args.input_dir)
    print("Total records:", len(records))

    records = reporting.get_grouped_records(records)

    def get_best_param(env, params, records, args):
        collect_param = ['batch_size', 'class_balanced', 'data_augmentation', 'lr',
                         'nonlinear_classifier', 'resnet_dropout', 'weight_decay']
        if "CAD" in args.algorithm:
            collect_param += ['lmbda', 'temperature', 'is_normalized', 'is_project', 'is_flipped']

        for selection_method in SELECTION_METHODS:
            for group in records:
                trail_seed = group['trial_seed']
                if 'trial_seed_%d' % trail_seed not in params:
                    params['trial_seed_%d' % trail_seed] = {}
                param = {}
                best_hparams = selection_method.hparams_accs(group['records'])[1:2]
                for run_acc, hparam_records in best_hparams:
                    for r in hparam_records:
                        assert (r['hparams'] == hparam_records[0]['hparams'])
                    for k, v in sorted(hparam_records[0]['hparams'].items()):
                        if k in collect_param:
                            param[k] = v
                key = args.dataset + '_env{}'.format(env)
                params['trial_seed_%d' % trail_seed][key] = param
        return params


    records = records.filter(
        lambda r:
            r['dataset'] == args.dataset and
            r['algorithm'] == args.algorithm and
            r['test_env'] == args.test_env
    )

    for selection_method in SELECTION_METHODS:
        print(f'Model selection: {selection_method.name}')
        for group in records:
            print(f"trial_seed: {group['trial_seed']}")
            best_hparams = selection_method.hparams_accs(group['records'])
            for run_acc, hparam_records in best_hparams:
                print(f"\t{run_acc}")
                for r in hparam_records:
                    assert(r['hparams'] == hparam_records[0]['hparams'])
                print("\t\thparams:")
                for k, v in sorted(hparam_records[0]['hparams'].items()):
                    print('\t\t\t\'{}\': {},'.format(k, v))
                print("\t\thyper_seed:")
                hyper_seed = hparam_records.select('args.hparams_seed').unique()
                print('\t\t\t\'{}\': {},'.format("hyper_seed", hyper_seed))
                print("\t\toutput_dirs:")
                output_dirs = hparam_records.select('args.output_dir').unique()
                for output_dir in output_dirs:
                    print(f"\t\t\t{output_dir}")
