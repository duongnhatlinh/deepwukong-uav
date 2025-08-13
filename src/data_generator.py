import xml.etree.ElementTree as ET
import networkx as nx
import pickle
from typing import List, Set, Tuple, Dict
from os.path import join, exists
from argparse import ArgumentParser
import os
from tqdm import tqdm
from typing import cast
import dataclasses
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Manager, Pool, Queue
import functools
import tempfile
import pickle
import traceback

USE_CPU = max(1, cpu_count() // 2)  # Giảm số process để tránh overload


def extract_line_number(idx: int, nodes: List) -> int:
    """
    return the line number of node index

    Args:
        idx (int): node index
        nodes (List)
    Returns: line number of node idx
    """
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except Exception as e:
                    print(e)
                    pass
        idx -= 1
    return -1


def read_csv(csv_file_path: str) -> List:
    """
    read csv file
    """
    assert exists(csv_file_path), f"no {csv_file_path}"
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data
    
def write_gpickle(graph, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(graph, f)
    except Exception as e:
        print(f"Error writing to gpickle file: {e}")
    
def extract_nodes_with_location_info(nodes):
    """
    Will return an array identifying the indices of those nodes in nodes array
    another array identifying the node_id of those nodes
    another array indicating the line numbers
    all 3 return arrays should have same length indicating 1-to-1 matching.
    
    """

    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number


def build_PDG(code_path: str, sensi_api_path: str,
              source_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    """
    build program dependence graph from code

    Args:
        code_path (str): source code root path
        sensi_api_path (str): path to sensitive apis
        source_path (str): source file path

    Returns: (PDG, key line map)
    """
    nodes_path = join(code_path, "nodes.csv")
    edges_path = join(code_path, "edges.csv")
    assert exists(sensi_api_path), f"{sensi_api_path} not exists!"
    with open(sensi_api_path, "r", encoding="utf-8") as f:
        sensi_api_set = set([api.strip() for api in f.read().split(",")])
    if not exists(nodes_path) or not exists(edges_path):
        return None, None
    
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    call_lines = set()
    array_lines = set()
    ptr_lines = set()
    arithmatic_lines = set()
    if len(nodes) == 0:
        return None, None
    for node_idx, node in enumerate(nodes):
        ntype = node['type'].strip()
        if ntype == 'CallExpression':
            function_name = nodes[node_idx + 1]['code']
            if function_name is None or function_name.strip() == '':
                continue
            if function_name.strip() in sensi_api_set:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    call_lines.add(line_no)
        # elif ntype
        elif ntype == 'ArrayIndexing':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                array_lines.add(line_no)
        elif ntype == 'PtrMemberAccess':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                ptr_lines.add(line_no)
        elif node['operator'].strip() in ['+', '-', '*', '/', '<=', '>=', '==', '<', '>', '!=']: 
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                arithmatic_lines.add(line_no)

    PDG = nx.DiGraph(file_paths=[source_path])
    control_edges, data_edges = list(), list()
    node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(
        nodes)
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_ln.keys(
            ) or end_node_id not in node_id_to_ln.keys():
                continue
            start_ln = node_id_to_ln[start_node_id]
            end_ln = node_id_to_ln[end_node_id]
            if edge_type == 'CONTROLS':  # Control
                control_edges.append((start_ln, end_ln, {"c/d": "c"}))
            if edge_type == 'REACHES':  # Data
                data_edges.append((start_ln, end_ln, {"c/d": "d"}))
    PDG.add_edges_from(control_edges)
    PDG.add_edges_from(data_edges)
    return PDG, {
        "call": call_lines,
        "array": array_lines,
        "ptr": ptr_lines,
        "arith": arithmatic_lines
    }


def build_XFG(PDG: nx.DiGraph, key_line_map: Dict[str, Set[int]],
              vul_lines: Set[int] = None) -> Dict[str, List[nx.DiGraph]]:
    """
    build XFGs
    Args:
        PDG (nx.DiGraph): program dependence graph
        key_line_map (Dict[str, Set[int]]): key lines
    Returns: XFG map
    """
    if PDG is None or key_line_map is None:
        return None
    # ct0, ct1 = 0, 0
    res = {"call": [], "array": [], "ptr": [], "arith": []}
    for key in ["call", "array", "ptr", "arith"]:
        for ln in key_line_map[key]:
            sliced_lines = set()

            # backward traversal
            bqueue = list()
            visited = set()
            bqueue.append(ln)
            visited.add(ln)
            while bqueue:
                fro = bqueue.pop(0)
                sliced_lines.add(fro)
                if fro in PDG._pred:
                    for pred in PDG._pred[fro]:
                        if pred not in visited:
                            visited.add(pred)
                            bqueue.append(pred)

            # forward traversal
            fqueue = list()
            visited = set()
            fqueue.append(ln)
            visited.add(ln)
            while fqueue:
                fro = fqueue.pop(0)
                sliced_lines.add(fro)
                if fro in PDG._succ:
                    for succ in PDG._succ[fro]:
                        if succ not in visited:
                            visited.add(succ)
                            fqueue.append(succ)
            if len(sliced_lines) != 0:
                XFG = PDG.subgraph(list(sliced_lines)).copy()
                XFG.graph["key_line"] = ln
                if vul_lines is not None:
                    if len(sliced_lines.intersection(vul_lines)) != 0:
                        XFG.graph["label"] = 1
                        # ct1 += 1
                    else:
                        XFG.graph["label"] = 0
                        # ct0 += 1
                else:
                    XFG.graph["label"] = "UNK"
                res[key].append(XFG)
        # print("ct1:", ct1)
        # print("ct0:", ct0)

    return res


def getCodeIDtoPathDict(testcases: List,
                        sourceDir: str) -> Dict[str, Dict[str, Set[int]]]:
    '''build code testcaseid to path map

    use the manifest.xml. build {testcaseid:{filePath:set(vul lines)}}
    filePath use relevant path, e.g., CWE119/cve/source-code/project_commit/...
    :param testcases:
    :return: {testcaseid:{filePath:set(vul lines)}}
    '''
    codeIDtoPath: Dict[str, Dict[str, Set[int]]] = {}
    for testcase in testcases:
        files = testcase.findall("file")
        testcaseid = testcase.attrib["id"]
        codeIDtoPath[testcaseid] = dict()

        for file in files:
            path = file.attrib["path"]
            flaws = file.findall("flaw")
            mixeds = file.findall("mixed")
            fix = file.findall("fix")
            # print(mixeds)
            VulLine = set()
            if (flaws != [] or mixeds != [] or fix != []):
                # targetFilePath = path
                if (flaws != []):
                    for flaw in flaws:
                        VulLine.add(int(flaw.attrib["line"]))
                if (mixeds != []):
                    for mixed in mixeds:
                        VulLine.add(int(mixed.attrib["line"]))

            codeIDtoPath[testcaseid][path] = VulLine

    return codeIDtoPath


def dump_XFG(res: Dict[str, List[nx.DiGraph]], out_root_path: str,
             testcaseid: str):
    """
    dump XFG to file

    Args:
        res: XFGs
        out_root_path: output root path
        testcaseid: testcase id
    Returns:
    """
    if res is None:
        return 0
    testcase_out_root_path = join(out_root_path, testcaseid)
    if not exists(testcase_out_root_path):
        os.makedirs(testcase_out_root_path)
    
    xfg_count = 0
    for k in res:
        k_root_path = join(testcase_out_root_path, k)
        if not exists(k_root_path):
            os.makedirs(k_root_path)
        for XFG in res[k]:
            out_path = join(k_root_path, f"{XFG.graph['key_line']}.xfg.pkl")
            try:
                write_gpickle(XFG, out_path)
                xfg_count += 1
            except Exception as e:
                print(f"Error saving XFG {out_path}: {e}")
    return xfg_count


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    return arg_parser


def process_single_testcase(testcase_id: str, codeIDtoPath: Dict, cwe_root: str,
                           source_root_path: str, out_root_path: str, doneIDs: Set):
    """
    Process a single testcase without multiprocessing queue
    
    Args:
        testcase_id: testcase id
        codeIDtoPath: code id to path mapping
        cwe_root: cwe root path
        source_root_path: source root path  
        out_root_path: output root path
        doneIDs: set of done IDs
    
    Returns:
        tuple: (testcase_id, xfg_count, success)
    """
    if testcase_id in doneIDs:
        return testcase_id, 0, True
    
    try:
        total_xfg_count = 0
        if testcase_id in codeIDtoPath:
            file_map = codeIDtoPath[testcase_id]
            for file_path in file_map:
                vul_lines = file_map[file_path]
                csv_path = join(cwe_root, "csv", os.path.abspath(source_root_path)[1:],
                                file_path)
                source_path = join(source_root_path, file_path)
                try:
                    PDG, key_line_map = build_PDG(csv_path, "data/sensiAPI.txt",
                                                  source_path)
                    res = build_XFG(PDG, key_line_map, vul_lines)
                    xfg_count = dump_XFG(res, out_root_path, testcase_id)
                    total_xfg_count += xfg_count
                except Exception as e:
                    print(f"Error processing file {file_path} in testcase {testcase_id}: {e}")
                    continue
        
        return testcase_id, total_xfg_count, True
    except Exception as e:
        print(f"Error processing testcase {testcase_id}: {e}")
        traceback.print_exc()
        return testcase_id, 0, False


def generate_sequential(config_path: str):
    """
    Sequential version without multiprocessing to avoid EOFError
    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    root = config.data_folder
    cweid = config.dataset.name
    cwe_root = join(root, cweid)
    source_root_path = join(cwe_root, "source-code")
    out_root_path = join(cwe_root, "XFG")
    xml_path = join(source_root_path, "manifest.xml")

    tree = ET.ElementTree(file=xml_path)
    testcases = tree.findall("testcase")
    codeIDtoPath = getCodeIDtoPathDict(testcases, source_root_path)

    if not exists(out_root_path):
        os.makedirs(out_root_path)
    if not exists(join(cwe_root, "doneID.txt")):
        os.system("touch {}".format(join(cwe_root, "doneID.txt")))
    
    with open(join(cwe_root, "doneID.txt"), "r", encoding="utf-8") as f:
        doneIDs = set(f.read().split(","))

    testcase_ids = [tc.attrib["id"] for tc in testcases]
    total_xfg_count = 0
    
    for testcase_id in tqdm(testcase_ids, desc="Processing testcases"):
        testcase_id, xfg_count, success = process_single_testcase(
            testcase_id, codeIDtoPath, cwe_root, source_root_path, 
            out_root_path, doneIDs
        )
        
        if success:
            total_xfg_count += xfg_count
            # Update done IDs
            with open(join(cwe_root, "doneID.txt"), 'a', encoding="utf-8") as f:
                f.write(str(testcase_id))
                f.write(",")
    
    print(f"Total {total_xfg_count} XFGs generated!")


def generate_with_smaller_batches(config_path: str):
    """
    Use smaller batch multiprocessing to reduce memory pressure
    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    root = config.data_folder
    cweid = config.dataset.name
    cwe_root = join(root, cweid)
    source_root_path = join(cwe_root, "source-code")
    out_root_path = join(cwe_root, "XFG")
    xml_path = join(source_root_path, "manifest.xml")

    tree = ET.ElementTree(file=xml_path)
    testcases = tree.findall("testcase")
    codeIDtoPath = getCodeIDtoPathDict(testcases, source_root_path)

    if not exists(out_root_path):
        os.makedirs(out_root_path)
    if not exists(join(cwe_root, "doneID.txt")):
        os.system("touch {}".format(join(cwe_root, "doneID.txt")))
    
    with open(join(cwe_root, "doneID.txt"), "r", encoding="utf-8") as f:
        doneIDs = set(f.read().split(","))

    testcase_ids = [tc.attrib["id"] for tc in testcases if tc.attrib["id"] not in doneIDs]
    
    # Process in smaller batches
    batch_size = 100  # Smaller batch size
    total_xfg_count = 0
    
    for i in range(0, len(testcase_ids), batch_size):
        batch = testcase_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(testcase_ids) + batch_size - 1)//batch_size}")
        
        with Pool(processes=min(4, USE_CPU)) as pool:  # Limit processes
            process_func = functools.partial(
                process_single_testcase,
                codeIDtoPath=codeIDtoPath,
                cwe_root=cwe_root,
                source_root_path=source_root_path,
                out_root_path=out_root_path,
                doneIDs=doneIDs
            )
            
            try:
                results = list(tqdm(
                    pool.imap(process_func, batch),
                    desc=f"Batch progress",
                    total=len(batch)
                ))
                
                # Update done IDs and count
                for testcase_id, xfg_count, success in results:
                    if success:
                        total_xfg_count += xfg_count
                        with open(join(cwe_root, "doneID.txt"), 'a', encoding="utf-8") as f:
                            f.write(str(testcase_id))
                            f.write(",")
                        doneIDs.add(testcase_id)
                
            except Exception as e:
                print(f"Error in batch processing: {e}")
                # Fall back to sequential for this batch
                for testcase_id in batch:
                    testcase_id, xfg_count, success = process_single_testcase(
                        testcase_id, codeIDtoPath, cwe_root, source_root_path,
                        out_root_path, doneIDs
                    )
                    if success:
                        total_xfg_count += xfg_count
                        with open(join(cwe_root, "doneID.txt"), 'a', encoding="utf-8") as f:
                            f.write(str(testcase_id))
                            f.write(",")
                        doneIDs.add(testcase_id)
    
    print(f"Total {total_xfg_count} XFGs generated!")


def generate(config_path: str):
    """
    Main generate function with fallback strategies
    """
    try:
        print("Attempting batch multiprocessing...")
        generate_with_smaller_batches(config_path)
    except Exception as e:
        print(f"Batch multiprocessing failed: {e}")
        print("Falling back to sequential processing...")
        generate_sequential(config_path)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    generate(__args.config)

    # sensi_api_path = "/home/linh/Documents/code/DeepWukong/data/sensiAPI.txt"
    # code_path = "/home/linh/Documents/code/DeepWukong/data_test/csv/home/linh/Documents/code/DeepWukong/data_test/source-code/vulnerable_LinkInterface.h/nodes.csv"
    # source_path = "/home/linh/Documents/code/DeepWukong/data_test/source-code/vulnerable_LinkInterface.h"
    # PDG, key_line_map = build_PDG(
    #     code_path=code_path,
    #     sensi_api_path=sensi_api_path,
    #     source_path=source_path
    # )

    # print("PDG: ", PDG)

    # print("call_lines: ", len(list(key_line_map['call'])),  "call_lines: ", sorted(key_line_map['call']))
    # print("array_lines: ", len(list(key_line_map['array'])),  "array_lines: ", sorted(key_line_map['array']))
    # print("ptr_lines: ", len(list(key_line_map['ptr'])),  "ptr_lines: ", sorted(key_line_map['ptr']))
    # print("arithmatic_lines: ", len(list(key_line_map['arith'])),  "arithmatic_lines: ", sorted(key_line_map['arith']))

    # vul_lines = set([42])

    # res = build_XFG(PDG, key_line_map, vul_lines=vul_lines)
    # for k in res:
    #     print(f"{k} XFGs: {len(res[k])}")
    #     for xfg in res[k]:
    #         print(f"XFG key line: {xfg.graph['key_line']}, label: {xfg.graph['label']}")
            # print("Nodes:", xfg.nodes(data=True))
            # print("Edges:", xfg.edges(data=True))
    