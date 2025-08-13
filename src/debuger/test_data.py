#!/usr/bin/env python3
"""
UAV Dataset Quality Diagnosis
Analyze why F1 score stays at 0 during training
"""

import json
import pickle
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

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

class UAVDatasetDiagnostics:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
    def load_splits(self):
        """Load train/val/test splits"""
        splits = ['train', 'val', 'test']
        for split in splits:
            split_file = self.data_path / f"{split}.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    data = json.load(f)
                setattr(self, f"{split}_data", data)
                print(f"✅ Loaded {split}: {len(data)} samples")
            else:
                print(f"❌ Missing {split}.json")
    
    def analyze_class_distribution(self):
        """Analyze label distribution across splits"""
        print("\n📊 Class Distribution Analysis")
        print("=" * 50)
        
        for split_name in ['train', 'val', 'test']:
            split_data = getattr(self, f"{split_name}_data")
            labels = []
            
            for xfg_path in split_data:
                try:
                    xfg = read_gpickle(xfg_path)
                    label = xfg.graph.get("label", "unknown")
                    labels.append(label)
                except Exception as e:
                    print(f"⚠️  Error loading {xfg_path}: {e}")
            
            label_counts = Counter(labels)
            total = len(labels)
            
            print(f"\n{split_name.upper()} Split ({total} samples):")
            for label, count in label_counts.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"  Label {label}: {count:4d} ({percentage:5.1f}%)")
            
            # Check for severe imbalance
            if len(label_counts) == 2:
                counts = list(label_counts.values())
                ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
                if ratio > 10:
                    print(f"  ⚠️  SEVERE IMBALANCE: {ratio:.1f}:1 ratio")
    
    def analyze_graph_properties(self):
        """Analyze XFG graph properties"""
        print("\n🔍 XFG Graph Properties Analysis")
        print("=" * 50)
        
        stats = defaultdict(list)
        
        # Sample from train set for analysis
        sample_size = min(100, len(self.train_data))
        sample_paths = self.train_data[:sample_size]
        
        for xfg_path in sample_paths:
            try:
                xfg = read_gpickle(xfg_path)
                
                stats['num_nodes'].append(len(xfg.nodes))
                stats['num_edges'].append(len(xfg.edges))
                stats['density'].append(nx.density(xfg))
                
                # Check for empty or trivial graphs
                if len(xfg.nodes) == 0:
                    stats['empty_graphs'].append(xfg_path)
                elif len(xfg.nodes) == 1:
                    stats['single_node_graphs'].append(xfg_path)
                
                # Check node content
                nodes_with_tokens = 0
                total_tokens = 0
                for node in xfg.nodes(data=True):
                    if 'code_sym_token' in node[1]:
                        tokens = node[1]['code_sym_token']
                        if tokens:
                            nodes_with_tokens += 1
                            total_tokens += len(tokens)
                
                stats['nodes_with_tokens_ratio'].append(
                    nodes_with_tokens / len(xfg.nodes) if len(xfg.nodes) > 0 else 0
                )
                stats['avg_tokens_per_node'].append(
                    total_tokens / nodes_with_tokens if nodes_with_tokens > 0 else 0
                )
                
            except Exception as e:
                stats['corrupted_files'].append(str(xfg_path))
                print(f"⚠️  Error analyzing {xfg_path}: {e}")
        
        # Print statistics
        for metric, values in stats.items():
            if metric in ['empty_graphs', 'single_node_graphs', 'corrupted_files']:
                if values:
                    print(f"\n❌ {metric.replace('_', ' ').title()}: {len(values)}")
                    for item in values[:5]:  # Show first 5
                        print(f"    {item}")
                    if len(values) > 5:
                        print(f"    ... and {len(values) - 5} more")
            elif values:
                print(f"\n{metric.replace('_', ' ').title()}:")
                print(f"  Mean: {sum(values)/len(values):.2f}")
                print(f"  Min:  {min(values):.2f}")
                print(f"  Max:  {max(values):.2f}")
    
    def analyze_token_vocabulary(self):
        """Analyze token vocabulary coverage"""
        print("\n📝 Token Vocabulary Analysis")
        print("=" * 50)
        
        all_tokens = set()
        token_freq = Counter()
        
        sample_size = min(50, len(self.train_data))
        sample_paths = self.train_data[:sample_size]
        
        for xfg_path in sample_paths:
            try:
                xfg = read_gpickle(xfg_path)
                for node in xfg.nodes(data=True):
                    if 'code_sym_token' in node[1]:
                        tokens = node[1]['code_sym_token']
                        for token in tokens:
                            all_tokens.add(token)
                            token_freq[token] += 1
            except Exception as e:
                print(f"⚠️  Error processing tokens in {xfg_path}: {e}")
        
        print(f"Total unique tokens: {len(all_tokens)}")
        print(f"Top 20 most frequent tokens:")
        for token, freq in token_freq.most_common(20):
            print(f"  {token:20s}: {freq:4d}")
        
        # Check for suspicious patterns
        empty_token_count = token_freq.get('', 0)
        if empty_token_count > 0:
            print(f"\n⚠️  Found {empty_token_count} empty tokens")
        
        # Check UAV-specific tokens
        uav_tokens = [token for token in all_tokens 
                     if any(keyword in token.lower() 
                           for keyword in ['uav', 'drone', 'flight', 'autopilot', 'mavlink'])]
        if uav_tokens:
            print(f"\n✅ UAV-specific tokens found: {len(uav_tokens)}")
            for token in uav_tokens[:10]:
                print(f"  {token}")
        else:
            print(f"\n❌ No UAV-specific tokens found!")
    
    def check_data_leakage(self):
        """Check for potential data leakage between splits"""
        print("\n🔍 Data Leakage Detection")
        print("=" * 50)
        
        # Simple check: look for identical file paths across splits
        train_files = set(Path(p).name for p in self.train_data)
        val_files = set(Path(p).name for p in self.val_data)
        test_files = set(Path(p).name for p in self.test_data)
        
        train_val_overlap = train_files & val_files
        train_test_overlap = train_files & test_files
        val_test_overlap = val_files & test_files
        
        if train_val_overlap:
            print(f"⚠️  Train-Val overlap: {len(train_val_overlap)} files")
        if train_test_overlap:
            print(f"⚠️  Train-Test overlap: {len(train_test_overlap)} files")
        if val_test_overlap:
            print(f"⚠️  Val-Test overlap: {len(val_test_overlap)} files")
        
        if not any([train_val_overlap, train_test_overlap, val_test_overlap]):
            print("✅ No obvious data leakage detected")
    
    def suggest_fixes(self):
        """Suggest potential fixes based on analysis"""
        print("\n💡 Suggested Fixes")
        print("=" * 50)
        
        print("1. **Check Class Balance:**")
        print("   - If severe imbalance, use weighted loss or SMOTE")
        print("   - Consider focal loss for hard examples")
        
        print("\n2. **Validate Graph Quality:**")
        print("   - Remove empty or single-node graphs")
        print("   - Ensure all nodes have meaningful tokens")
        
        print("\n3. **Check Learning Rate:**")
        print("   - Current: 0.002 might be too high for UAV domain")
        print("   - Try: 0.0005 or 0.001")
        
        print("\n4. **Increase Model Capacity:**")
        print("   - Current: 256 hidden size, 3 layers")
        print("   - Try: 512 hidden size, 4-5 layers")
        
        print("\n5. **Domain Adaptation:**")
        print("   - Retrain Word2Vec on UAV-specific corpus")
        print("   - Add UAV-specific sensitive APIs")
        
        print("\n6. **Debug Training Process:**")
        print("   - Add per-class metrics logging")
        print("   - Visualize embeddings with t-SNE")
        print("   - Check gradient flow")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to UAV dataset (e.g., data/UAV)')
    args = parser.parse_args()
    
    print("🔍 UAV Dataset Quality Diagnostics")
    print("=" * 50)
    
    diagnostics = UAVDatasetDiagnostics(args.data_path)
    
    # Run all analyses
    diagnostics.load_splits()
    diagnostics.analyze_class_distribution()
    diagnostics.analyze_graph_properties()
    diagnostics.analyze_token_vocabulary()
    diagnostics.check_data_leakage()
    # diagnostics.suggest_fixes()

if __name__ == "__main__":
    main()

# Usage: python uav_diagnostics.py data/UAV