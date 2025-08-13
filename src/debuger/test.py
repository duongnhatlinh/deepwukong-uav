import pickle
import os
from pathlib import Path

def read_gpickle(filename):
    try:
        with open(filename, 'rb') as f:
            graph = pickle.load(f)
        return graph
    except Exception as e:
        print(f"Error reading gpickle file: {e}")

path = "data/UAV/moderate_balanced/XFG/train_vuln_original_00075.xfg.pkl"
path2 = "data/CWE119/XFG/148815/ptr/2148.xfg.pkl"

graph = read_gpickle(path)
print(graph.graph["key_line"])

# for node in graph.nodes(data=True):
#     if 'code_sym_token' in node[1]:
#         tokens = node[1]['code_sym_token']
#         print(tokens, len(tokens))