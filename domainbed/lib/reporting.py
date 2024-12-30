# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections

import json
import os

import tqdm

from domainbed.lib.query import Q


def recursive_path(dir, valid_paths):
    for name in os.listdir(dir):
        sub_dir = os.path.join(dir, name)
        if os.path.isdir(sub_dir):
            recursive_path(sub_dir, valid_paths)
        else:
            if sub_dir not in valid_paths:
                valid_paths.append(dir)

def load_records(path, recurs=False):
    records = []
    if recurs:
        valid_paths = []
        recursive_path(path, valid_paths)
    else:
        valid_paths = os.listdir(path)

    # print(valid_paths)
    for i, subdir in tqdm.tqdm(list(enumerate(valid_paths)),
                               ncols=80,
                               leave=False):
        results_path = os.path.join(path, subdir, "results.jsonl")

        # checking done is used in ours sweep

        # done = os.path.join(path, subdir, "done")
        # if not os.path.exists(done):
        #     continue
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    if len(records) == 0 and "results.jsonl" in os.listdir(path):
        results_path = os.path.join(path, "results.jsonl")
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)



def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    skipped = 0
    for r in records:
        for test_env in r["args"]["test_envs"]:
            # if r["args"]["trial_seed"] not in range(0, 50):
            #     skipped += 1
            #     continue
            group = (r["args"]["trial_seed"],
                r["args"]["dataset"],
                r["args"]["algorithm"],
                test_env)
            result[group].append(r)
    print("skipped {} records".format(skipped))
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "test_env": e,
        "records": Q(r)} for (t,d,a,e),r in result.items()])
