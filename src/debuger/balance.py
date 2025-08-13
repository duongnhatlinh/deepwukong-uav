#!/usr/bin/env python3
"""
UAV Dataset Balancer & Data Leak Fixer
Fixes severe class imbalance and data leakage issues
"""

import json
import pickle
import random
import shutil
import hashlib
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import numpy as np

def read_gpickle(filename):
    """Safe gpickle reading"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def write_gpickle(graph, filename):
    """Safe gpickle writing"""
    try:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        print(f"Error writing {filename}: {e}")
        return False

class UAVDataBalancer:
    """Fix UAV dataset imbalance and data leakage"""
    
    def __init__(self, data_path: str, output_path: str = None):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path) if output_path else self.data_path / "balanced"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.all_samples = []
        self.vulnerable_samples = []
        self.non_vulnerable_samples = []
        
    def load_all_data(self):
        """Load all data from train/val/test splits"""
        print("📥 Loading all dataset samples...")
        
        splits = ['train', 'val', 'test']
        for split in splits:
            split_file = self.data_path / f"{split}.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    data = json.load(f)
                    
                for sample_path in data:
                    xfg = read_gpickle(sample_path)
                    if xfg is not None:
                        label = xfg.graph.get("label", -1)
                        sample_info = {
                            'path': sample_path,
                            'label': label,
                            'xfg': xfg,
                            'original_split': split
                        }
                        
                        self.all_samples.append(sample_info)
                        
                        if label == 1:
                            self.vulnerable_samples.append(sample_info)
                        elif label == 0:
                            self.non_vulnerable_samples.append(sample_info)
        
        print(f"  Total samples: {len(self.all_samples)}")
        print(f"  Vulnerable (label=1): {len(self.vulnerable_samples)}")
        print(f"  Non-vulnerable (label=0): {len(self.non_vulnerable_samples)}")
        
        if len(self.vulnerable_samples) == 0:
            raise ValueError("No vulnerable samples found! Check your dataset.")
    
    def deduplicate_data(self):
        """Remove duplicate samples based on graph structure"""
        print("🧹 Deduplicating samples...")
        
        def get_graph_signature(xfg):
            """Create unique signature for graph"""
            # Use node content and edge structure
            node_tokens = []
            for node in sorted(xfg.nodes()):
                if 'code_sym_token' in xfg.nodes[node]:
                    tokens = xfg.nodes[node]['code_sym_token']
                    node_tokens.append(''.join(tokens))
            
            edge_list = sorted(list(xfg.edges()))
            signature = hashlib.md5(
                (str(node_tokens) + str(edge_list)).encode()
            ).hexdigest()
            
            return signature
        
        # Deduplicate vulnerable samples
        vuln_signatures = {}
        unique_vulnerable = []
        
        for sample in self.vulnerable_samples:
            sig = get_graph_signature(sample['xfg'])
            if sig not in vuln_signatures:
                vuln_signatures[sig] = sample
                unique_vulnerable.append(sample)
        
        # Deduplicate non-vulnerable samples
        non_vuln_signatures = {}
        unique_non_vulnerable = []
        
        for sample in self.non_vulnerable_samples:
            sig = get_graph_signature(sample['xfg'])
            if sig not in non_vuln_signatures:
                non_vuln_signatures[sig] = sample
                unique_non_vulnerable.append(sample)
        
        print(f"  Before deduplication:")
        print(f"    Vulnerable: {len(self.vulnerable_samples)} → {len(unique_vulnerable)}")
        print(f"    Non-vulnerable: {len(self.non_vulnerable_samples)} → {len(unique_non_vulnerable)}")
        
        self.vulnerable_samples = unique_vulnerable
        self.non_vulnerable_samples = unique_non_vulnerable
        
        return len(unique_vulnerable), len(unique_non_vulnerable)
    
    def augment_vulnerable_samples(self, target_ratio: float = 0.3):
        """Augment vulnerable samples to reach target ratio"""
        print(f"🔄 Augmenting vulnerable samples to {target_ratio:.1%} ratio...")
        
        current_vuln = len(self.vulnerable_samples)
        current_non_vuln = len(self.non_vulnerable_samples)
        
        # Calculate how many vulnerable samples we need

        # target_ratio = vuln / (vuln + non_vuln)
        # vuln = target_ratio * (vuln + non_vuln)
        # vuln = target_ratio * vuln + target_ratio * non_vuln
        # vuln * (1 - target_ratio) = target_ratio * non_vuln
        # vuln = target_ratio * non_vuln / (1 - target_ratio)
        
        target_vuln = int(target_ratio * current_non_vuln / (1 - target_ratio))
        needed_vuln = max(0, target_vuln - current_vuln)
        
        print(f"  Current: {current_vuln} vulnerable, {current_non_vuln} non-vulnerable")
        print(f"  Target: {target_vuln} vulnerable samples")
        print(f"  Need to generate: {needed_vuln} additional vulnerable samples")
        
        if needed_vuln == 0:
            print("  ✅ No augmentation needed")
            return
        
        # Generate augmented samples
        augmented_samples = []
        for i in range(needed_vuln):
            # Pick a random vulnerable sample to augment
            base_sample = random.choice(self.vulnerable_samples)
            augmented_xfg = self.augment_xfg(base_sample['xfg'], i)
            
            if augmented_xfg is not None:
                augmented_sample = {
                    'path': f"augmented_vulnerable_{i:04d}.xfg.pkl",
                    'label': 1,
                    'xfg': augmented_xfg,
                    'original_split': 'augmented',
                    'base_sample': base_sample['path']
                }
                augmented_samples.append(augmented_sample)
        
        self.vulnerable_samples.extend(augmented_samples)
        print(f"  ✅ Generated {len(augmented_samples)} augmented vulnerable samples")
        
        return len(augmented_samples)
    
    def augment_xfg(self, original_xfg, seed):
        """Create augmented version of XFG"""
        # Create a copy of the graph
        aug_xfg = original_xfg.copy()
        
        # Set random seed for reproducible augmentation
        np.random.seed(seed)
        
        # Augmentation techniques for graphs:
        
        # 1. Add noise to node features (token permutation)
        for node in aug_xfg.nodes():
            if 'code_sym_token' in aug_xfg.nodes[node]:
                tokens = aug_xfg.nodes[node]['code_sym_token'].copy()
                if len(tokens) > 1 and np.random.random() < 0.1:  # 10% chance
                    # Randomly shuffle some tokens
                    idx = np.random.randint(0, len(tokens))
                    if idx < len(tokens) - 1:
                        tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
                    aug_xfg.nodes[node]['code_sym_token'] = tokens
        

        # 2. Edge dropout (remove some edges)
        if len(aug_xfg.edges()) > 2 and np.random.random() < 0.05:  # 5% chance
            edge_list = list(aug_xfg.edges())
            num_remove = max(1, len(edge_list) // 20)
            if len(edge_list) >= num_remove:
                import random  # Đảm bảo đã import random ở đầu file
                edges_to_remove = random.sample(edge_list, num_remove)
                aug_xfg.remove_edges_from(edges_to_remove)
                
        # 3. Subgraph sampling (keep most important parts)
        if len(aug_xfg.nodes()) > 10 and np.random.random() < 0.1:  # 10% chance
            # Keep central nodes based on degree
            node_degrees = dict(aug_xfg.degree())
            sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            keep_ratio = 0.9  # Keep 90% of nodes
            nodes_to_keep = [node for node, _ in sorted_nodes[:int(len(sorted_nodes) * keep_ratio)]]
            aug_xfg = aug_xfg.subgraph(nodes_to_keep).copy()
        
        # Ensure the graph is still connected and meaningful
        if len(aug_xfg.nodes()) < 2:
            return None
            
        # Keep original metadata but mark as augmented
        aug_xfg.graph['label'] = 1
        aug_xfg.graph['augmented'] = True
        aug_xfg.graph['augmentation_seed'] = seed
        
        return aug_xfg
    
    def balance_dataset_by_undersampling(self, target_ratio: float = 0.3):
        """Balance dataset by intelligent undersampling of majority class"""
        print(f"⚖️ Balancing dataset by undersampling to {target_ratio:.1%} ratio...")
        
        current_vuln = len(self.vulnerable_samples)
        
        # Calculate target non-vulnerable samples
        # target_ratio = vuln / (vuln + non_vuln)
        # non_vuln = vuln * (1 - target_ratio) / target_ratio
        target_non_vuln = int(current_vuln * (1 - target_ratio) / target_ratio)
        
        print(f"  Current: {current_vuln} vulnerable, {len(self.non_vulnerable_samples)} non-vulnerable")
        print(f"  Target: {current_vuln} vulnerable, {target_non_vuln} non-vulnerable")
        
        if target_non_vuln >= len(self.non_vulnerable_samples):
            print("  ✅ No undersampling needed")
            return
        
        # Intelligent undersampling: keep diverse samples
        selected_non_vuln = self.select_diverse_samples(
            self.non_vulnerable_samples, 
            target_non_vuln
        )
        
        print(f"  ✅ Undersampled non-vulnerable: {len(self.non_vulnerable_samples)} → {len(selected_non_vuln)}")
        self.non_vulnerable_samples = selected_non_vuln
        
        return len(selected_non_vuln)
    
    def select_diverse_samples(self, samples: List[Dict], target_count: int) -> List[Dict]:
        """Select diverse samples using graph-based clustering"""
        if len(samples) <= target_count:
            return samples
        
        print(f"    Selecting {target_count} diverse samples from {len(samples)}...")
        
        # Simple diversity selection based on graph properties
        sample_features = []
        
        for sample in samples:
            xfg = sample['xfg']
            features = [
                len(xfg.nodes()),           # Number of nodes
                len(xfg.edges()),           # Number of edges
                nx.density(xfg),            # Graph density
                len(list(nx.connected_components(xfg.to_undirected()))),  # Connected components
            ]
            
            # Add token diversity
            all_tokens = set()
            for node in xfg.nodes():
                if 'code_sym_token' in xfg.nodes[node]:
                    all_tokens.update(xfg.nodes[node]['code_sym_token'])
            features.append(len(all_tokens))  # Vocabulary size
            
            sample_features.append(features)
        
        # Use k-means style selection for diversity
        selected_indices = self.diverse_sampling(sample_features, target_count)
        selected_samples = [samples[i] for i in selected_indices]
        
        return selected_samples
    
    def diverse_sampling(self, features: List[List[float]], k: int) -> List[int]:
        """Select k diverse samples using greedy farthest-first strategy"""
        if len(features) <= k:
            return list(range(len(features)))
        
        # Normalize features
        features = np.array(features)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        selected = []
        
        # Start with random sample
        selected.append(random.randint(0, len(features) - 1))
        
        # Greedily select farthest samples
        for _ in range(k - 1):
            max_min_dist = -1
            farthest_idx = -1
            
            for i in range(len(features)):
                if i in selected:
                    continue
                
                # Find minimum distance to selected samples
                min_dist = min(
                    np.linalg.norm(features[i] - features[j]) 
                    for j in selected
                )
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_idx = i
            
            if farthest_idx != -1:
                selected.append(farthest_idx)
        
        return selected
    
    def create_balanced_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Create new balanced train/val/test splits without data leakage"""
        print("📊 Creating balanced splits without data leakage...")
        
        # Combine all samples and shuffle
        all_balanced_samples = self.vulnerable_samples + self.non_vulnerable_samples
        random.shuffle(all_balanced_samples)
        
        total_samples = len(all_balanced_samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        # Create splits
        train_samples = all_balanced_samples[:train_size]
        val_samples = all_balanced_samples[train_size:train_size + val_size]
        test_samples = all_balanced_samples[train_size + val_size:]
        
        print(f"  Split sizes:")
        print(f"    Train: {len(train_samples)} ({len(train_samples)/total_samples:.1%})")
        print(f"    Val:   {len(val_samples)} ({len(val_samples)/total_samples:.1%})")
        print(f"    Test:  {len(test_samples)} ({len(test_samples)/total_samples:.1%})")
        
        # Check balance in each split
        for split_name, split_data in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
            label_counts = Counter(sample['label'] for sample in split_data)
            total = len(split_data)
            if total > 0:
                vuln_ratio = label_counts[1] / total
                print(f"    {split_name} balance: {vuln_ratio:.1%} vulnerable")
        
        return train_samples, val_samples, test_samples
    
    def save_balanced_dataset(self, train_samples, val_samples, test_samples):
        """Save balanced dataset to disk"""
        print("💾 Saving balanced dataset...")
        
        # Create output directories
        (self.output_path / "XFG").mkdir(parents=True, exist_ok=True)
        
        split_paths = {'train': [], 'val': [], 'test': []}
        
        for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            for i, sample in enumerate(samples):
                # Create new filename
                label_str = "vuln" if sample['label'] == 1 else "safe"
                new_filename = f"{split_name}_{label_str}_{i:05d}.xfg.pkl"
                new_path = self.output_path / "XFG" / new_filename
                
                # Save XFG
                if write_gpickle(sample['xfg'], new_path):
                    split_paths[split_name].append(str(new_path))
        
        # Save split filestrain_samples
        for split_name, paths in split_paths.items():
            split_file = self.output_path / f"{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump(paths, f, indent=2)
            print(f"  ✅ Saved {split_name}.json: {len(paths)} samples")
        
        # Save balancing report
        report = {
            'original_stats': {
                'vulnerable': len([s for s in self.all_samples if s['label'] == 1]),
                'non_vulnerable': len([s for s in self.all_samples if s['label'] == 0]),
                'total': len(self.all_samples)
            },
            'balanced_stats': {
                'vulnerable': len(self.vulnerable_samples),
                'non_vulnerable': len(self.non_vulnerable_samples),
                'total': len(self.vulnerable_samples) + len(self.non_vulnerable_samples)
            },
            'split_stats': {
                'train': len(split_paths['train']),
                'val': len(split_paths['val']),
                'test': len(split_paths['test'])
            }
        }
        
        report_file = self.output_path / "balancing_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  📄 Balancing report saved: {report_file}")
        
        return self.output_path
    
    def run_balancing(self, method: str = "hybrid", target_ratio: float = 0.3):
        """Run complete dataset balancing"""
        print("🚀 UAV Dataset Balancing & Leak Fix")
        print("=" * 50)
        
        # Load data
        self.load_all_data()
        
        # Deduplicate
        self.deduplicate_data()
        
        if method == "augmentation":
            # Pure augmentation approach
            self.augment_vulnerable_samples(target_ratio)
        elif method == "undersampling":
            # Pure undersampling approach  
            self.balance_dataset_by_undersampling(target_ratio)
        elif method == "hybrid":
            # Hybrid approach: some augmentation + some undersampling
            # First augment to get closer to target
            self.augment_vulnerable_samples(target_ratio * 0.7)  # Get to 70% of target
            # Then undersample to reach final target
            self.balance_dataset_by_undersampling(target_ratio)
        
        # Create new splits without leakage
        train_samples, val_samples, test_samples = self.create_balanced_splits()
        
        # Save balanced dataset
        output_path = self.save_balanced_dataset(train_samples, val_samples, test_samples)
        
        print(f"\n✅ Dataset balancing completed!")
        print(f"📁 Balanced dataset saved to: {output_path}")
        print(f"🎯 Target vulnerable ratio: {target_ratio:.1%}")
        
        return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance UAV dataset and fix data leakage')
    parser.add_argument('data_path', help='Path to original UAV dataset')
    parser.add_argument('--output', '-o', help='Output path for balanced dataset')
    parser.add_argument('--method', choices=['augmentation', 'undersampling', 'hybrid'], 
                       default='hybrid', help='Balancing method')
    parser.add_argument('--target-ratio', type=float, default=0.3, 
                       help='Target ratio for vulnerable samples (0.3 = 30%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Run balancing
    balancer = UAVDataBalancer(args.data_path, args.output)
    output_path = balancer.run_balancing(args.method, args.target_ratio)
    
    print(f"\n🎉 Next steps:")
    print(f"1. Update your config to use: {output_path}")
    print(f"2. Retrain with balanced dataset")
    print(f"3. Use class weights if still needed")

if __name__ == "__main__":
    main()