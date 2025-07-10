#!/usr/bin/env python3
"""
Fix Data Leakage in UAV Dataset
Remove overlapping samples and create proper train/val/test splits
"""

import json
import hashlib
import networkx as nx
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
import pickle


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

def calculate_xfg_signature(xfg_path):
    """Calculate unique signature for XFG to detect real duplicates"""
    try:
        xfg = read_gpickle(xfg_path)
        
        # Create signature from graph structure + content
        nodes_content = []
        for node in sorted(xfg.nodes()):
            node_data = xfg.nodes[node]
            tokens = node_data.get('code_sym_token', [])
            nodes_content.append(f"{node}:{':'.join(tokens[:10])}")  # First 10 tokens
        
        edges_content = []
        for edge in sorted(xfg.edges()):
            edge_data = xfg.edges[edge]
            edge_type = edge_data.get('c/d', 'unknown')
            edges_content.append(f"{edge[0]}-{edge[1]}:{edge_type}")
        
        # Combine structure + label
        signature_data = {
            'nodes': nodes_content,
            'edges': edges_content,
            'label': xfg.graph.get('label', 'unknown'),
            'num_nodes': len(xfg.nodes),
            'num_edges': len(xfg.edges)
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()
        
    except Exception as e:
        print(f"Error processing {xfg_path}: {e}")
        return None

def fix_data_leakage(dataset_path: str):
    """Remove duplicates and create proper splits"""
    dataset_path = Path(dataset_path)
    
    print("🔧 Fixing data leakage in UAV dataset...")
    
    # Load current splits
    splits = {}
    all_paths = []
    
    for split_name in ['train', 'val', 'test']:
        split_file = dataset_path / f"{split_name}.json"
        if split_file.exists():
            with open(split_file, 'r') as f:
                splits[split_name] = json.load(f)
                all_paths.extend(splits[split_name])
    
    print(f"📊 Original dataset: {len(all_paths)} total samples")
    for split_name, paths in splits.items():
        print(f"  {split_name}: {len(paths)} samples")
    
    # Calculate signatures for all XFGs
    print("🔍 Calculating XFG signatures...")
    signature_to_paths = defaultdict(list)
    path_to_label = {}
    
    for i, xfg_path in enumerate(all_paths):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(all_paths)} files...")
        
        signature = calculate_xfg_signature(xfg_path)
        if signature:
            signature_to_paths[signature].append(xfg_path)
            
            # Store label for stratification
            try:
                xfg = read_gpickle(xfg_path)
                path_to_label[xfg_path] = xfg.graph.get('label', 0)
            except:
                path_to_label[xfg_path] = 0
    
    # Remove duplicates - keep only one instance per signature
    print("✂️  Removing duplicates...")
    unique_paths = []
    removed_count = 0
    
    for signature, paths in signature_to_paths.items():
        if len(paths) > 1:
            removed_count += len(paths) - 1
            print(f"  Found {len(paths)} duplicates for signature {signature[:8]}...")
        
        # Keep the first path (arbitrary choice)
        unique_paths.append(paths[0])
    
    print(f"📉 Removed {removed_count} duplicate samples")
    print(f"📊 Unique dataset: {len(unique_paths)} samples")
    
    # Check class distribution
    labels = [path_to_label[path] for path in unique_paths]
    label_0_count = sum(1 for label in labels if label == 0)
    label_1_count = sum(1 for label in labels if label == 1)
    
    print(f"📊 Class distribution:")
    print(f"  Label 0 (clean): {label_0_count} ({label_0_count/len(labels)*100:.1f}%)")
    print(f"  Label 1 (vuln):  {label_1_count} ({label_1_count/len(labels)*100:.1f}%)")
    
    # Create proper stratified splits
    print("🔀 Creating new stratified splits...")
    
    # First split: 80% train, 20% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        unique_paths, labels, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    # Second split: 10% val, 10% test from temp
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )
    
    # Save new splits
    new_splits = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }
    
    for split_name, paths in new_splits.items():
        split_file = dataset_path / f"{split_name}_fixed.json"
        with open(split_file, 'w') as f:
            json.dump(paths, f, indent=2)
        
        # Calculate class distribution for this split
        split_labels = [path_to_label[path] for path in paths]
        split_label_0 = sum(1 for label in split_labels if label == 0)
        split_label_1 = sum(1 for label in split_labels if label == 1)
        
        print(f"✅ {split_name}_fixed.json: {len(paths)} samples")
        print(f"   Label 0: {split_label_0} ({split_label_0/len(split_labels)*100:.1f}%)")
        print(f"   Label 1: {split_label_1} ({split_label_1/len(split_labels)*100:.1f}%)")
    
    # Backup original files
    for split_name in ['train', 'val', 'test']:
        original_file = dataset_path / f"{split_name}.json"
        backup_file = dataset_path / f"{split_name}_original_backup.json"
        if original_file.exists():
            original_file.rename(backup_file)
            print(f"📋 Backed up {split_name}.json to {split_name}_original_backup.json")
    
    # Move fixed files to main files
    for split_name in ['train', 'val', 'test']:
        fixed_file = dataset_path / f"{split_name}_fixed.json"
        main_file = dataset_path / f"{split_name}.json"
        fixed_file.rename(main_file)
    
    print("\n🎉 Data leakage fixed successfully!")
    print("📋 Original files backed up with '_original_backup' suffix")
    print("💾 New clean splits saved as train.json, val.json, test.json")
    
    return {
        'original_size': len(all_paths),
        'unique_size': len(unique_paths),
        'removed_duplicates': removed_count,
        'new_splits': {split: len(paths) for split, paths in new_splits.items()}
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix data leakage in UAV dataset')
    parser.add_argument('dataset_path', help='Path to UAV dataset directory (e.g., data/UAV)')
    
    args = parser.parse_args()
    
    result = fix_data_leakage(args.dataset_path)
    print(f"\n📊 Summary: {result}")

# Usage: python fix_data_leakage.py data/UAV