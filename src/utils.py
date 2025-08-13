import hashlib
from warnings import filterwarnings
import subprocess

from sklearn.model_selection import train_test_split
from typing import List, Union, Dict, Tuple
import numpy
import torch
import json
import os
import networkx as nx
from os.path import exists
from tqdm import tqdm
import pickle
from collections import defaultdict
import random


PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
BOS = "<BOS>"
EOS = "<EOS>"


def getMD5(s):
    '''
    得到字符串s的md5加密后的值

    :param s:
    :return:
    '''
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()



def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="lightning.pytorch.trainer.data_loading",
                   lineno=102)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="lightning.pytorch.utilities.data",
                   lineno=41)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=216)  # save
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=234)  # load
    filterwarnings("ignore",
                   category=DeprecationWarning,
                   module="pytorch_lightning.metrics.__init__",
                   lineno=43)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch._tensor",
                   lineno=575)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="src.models.modules.common_layers",
                   lineno=0)

def write_gpickle(graph, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error writing to gpickle file: {e}")

def read_gpickle(filename):
    try:
        with open(filename, 'rb') as f:
            graph = pickle.load(f)
        return graph
    except Exception as e:
        print(f"Error reading gpickle file: {e}")

def count_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path],
                                    capture_output=True,
                                    encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(
            f"Counting lines in {file_path} failed with error\n{command_result.stderr}"
        )
    return int(command_result.stdout.split()[0])


def create_balanced_dataset(xfg_paths: List[str]) -> List[str]:
    """
    Analyzes XFG duplicates, filters them, and creates a balanced dataset.
    This function incorporates the logic from xfg_duplicate.py.
    """
    md5_to_xfgs = defaultdict(list)
    for xfg_path in tqdm(xfg_paths, desc="Analyzing XFG duplicates"):
        try:
            xfg = read_gpickle(xfg_path)
            if xfg is None:
                continue
            label = xfg.graph["label"]

            edges_md5 = []
            for ln in xfg:
                if "code_sym_token" in xfg.nodes[ln]:
                    ln_md5 = getMD5(str(xfg.nodes[ln]["code_sym_token"]))
                else:
                    ln_md5 = getMD5(str(ln))
                xfg.nodes[ln]["md5"] = ln_md5

            for edge in xfg.edges:
                edge_md5 = xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"]
                edges_md5.append(edge_md5)

            xfg_md5 = getMD5(str(sorted(edges_md5)))

            xfg_info = {'path': xfg_path, 'label': label}
            md5_to_xfgs[xfg_md5].append(xfg_info)
        except Exception as e:
            print(f"Error processing {xfg_path}: {e}")
            continue

    unique_label1_paths = []
    unique_label0_paths = []
    dedup_vul_details = []
    dedup_safe_details = []
    conflicts_count = 0

    for xfg_md5, xfg_list in md5_to_xfgs.items():
        if len(xfg_list) == 1:
            if xfg_list[0]['label'] == 1:
                unique_label1_paths.append(xfg_list[0]['path'])
            else:
                unique_label0_paths.append(xfg_list[0]['path'])
        else:
            labels = set(item['label'] for item in xfg_list)
            if len(labels) > 1:
                conflicts_count += len(xfg_list)
                continue

            label = list(labels)[0]
            # Use the first path as representative for the duplicate group
            path = xfg_list[0]['path']
            count = len(xfg_list)
            if label == 1:
                dedup_vul_details.append({'path': path, 'count': count})
            else:
                dedup_safe_details.append({'path': path, 'count': count})

    print(f"Total conflicting XFGs (discarded): {conflicts_count}")
    
    vul_paths = unique_label1_paths + [item['path'] for item in dedup_vul_details]
    print("vul_paths: " ,len(vul_paths))

    # Số lượng safe_paths cần thiết
    needed_safe_paths = len(vul_paths) * 4

    safe_paths_unique = unique_label0_paths
    len_safe_paths_unique = len(safe_paths_unique)
    if needed_safe_paths < len_safe_paths_unique:
        return vul_paths + safe_paths_unique

    safe_paths_large_dup = [item['path'] for item in dedup_safe_details if item['count'] >= 10]

    temp_count_path = len_safe_paths_unique + len(safe_paths_large_dup)
    if temp_count_path < needed_safe_paths: 
        sample_size = needed_safe_paths - temp_count_path

        safe_paths_small_dup_candidates = [item['path'] for item in dedup_safe_details if item['count'] < 10]
        sample_size = min(sample_size, len(safe_paths_small_dup_candidates))
        safe_paths_small_dup_sampled = random.sample(safe_paths_small_dup_candidates, sample_size)

        return vul_paths + safe_paths_unique + safe_paths_large_dup + safe_paths_small_dup_sampled

    safe_paths_dup_candidates = [item['path'] for item in dedup_safe_details]
    sample_size = needed_safe_paths - len_safe_paths_unique
    sample_size = min(sample_size, len(safe_paths_dup_candidates))
    safe_paths_small_dup_sampled = random.sample(safe_paths_dup_candidates, sample_size)

    safe_paths = safe_paths_unique + safe_paths_large_dup + safe_paths_small_dup_sampled

    return vul_paths + safe_paths


def unique_xfg_raw(xfg_path_list):
    """f
    unique xfg from xfg list
    Args:
        xfg_path_list:

    Returns:
        md5_dict: {xfg md5:{"xfg": xfg_path, "label": 0/1/-1}}, -1 stands for conflict
    """
    md5_dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for xfg_path in xfg_path_list:
        xfg = read_gpickle(xfg_path)
        label = xfg.graph["label"]
        file_path = xfg.graph["file_paths"][0]
        assert exists(file_path), f"{file_path} not exists!"
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_contents = f.readlines()
        for ln in xfg:
            ln_md5 = getMD5(file_contents[ln - 1])
            xfg.nodes[ln]["md5"] = ln_md5
        edges_md5 = list()
        for edge in xfg.edges:
            edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
        xfg_md5 = getMD5(str(sorted(edges_md5)))
        if xfg_md5 not in md5_dict:
            md5_dict[xfg_md5] = dict()
            md5_dict[xfg_md5]["label"] = label
            md5_dict[xfg_md5]["xfg"] = xfg_path
        else:
            md5_label = md5_dict[xfg_md5]["label"]
            if md5_label != -1 and md5_label != label:
                conflict_ct += 1
                md5_dict[xfg_md5]["label"] = -1
            else:
                mul_ct += 1
    print(f"total conflit: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5_dict


def unique_xfg_sym(xfg_path_list):
    """f
    unique xfg from xfg list
    Args:
        xfg_path_list:

    Returns:
        md5_dict: {xfg md5:{"xfg": xfg_path, "label": 0/1/-1}}, -1 stands for conflict
    """
    md5_dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for xfg_path in tqdm(xfg_path_list, total=len(xfg_path_list), desc="xfgs: "):
        xfg = read_gpickle(xfg_path)
        label = xfg.graph["label"]
        file_path = xfg.graph["file_paths"][0]
        assert exists(file_path), f"{file_path} not exists!"
        for ln in xfg:
            ln_md5 = getMD5(str(xfg.nodes[ln]["code_sym_token"]))
            xfg.nodes[ln]["md5"] = ln_md5
        edges_md5 = list()
        for edge in xfg.edges:
            edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
        xfg_md5 = getMD5(str(sorted(edges_md5)))
        if xfg_md5 not in md5_dict:
            md5_dict[xfg_md5] = dict()
            md5_dict[xfg_md5]["label"] = label
            md5_dict[xfg_md5]["xfg"] = xfg_path
        else:
            md5_label = md5_dict[xfg_md5]["label"]
            if md5_label != -1 and md5_label != label:
                conflict_ct += 1
                md5_dict[xfg_md5]["label"] = -1
            else:
                mul_ct += 1
    print(f"total conflit: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5_dict


def split_list(files: List[str], out_root_path: str):
    """

    Args:
        files:
        out_root_path:

    Returns:

    """
    X_train, X_test = train_test_split(files, test_size=0.2)
    X_test, X_val = train_test_split(X_test, test_size=0.5)
    if not exists(f"{out_root_path}"):
        os.makedirs(f"{out_root_path}")
    with open(f"{out_root_path}/train.json", "w") as f:
        json.dump(X_train, f)
    with open(f"{out_root_path}/test.json", "w") as f:
        json.dump(X_test, f)
    with open(f"{out_root_path}/val.json", "w") as f:
        json.dump(X_val, f)
