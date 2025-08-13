#!/usr/bin/env python3
"""
Graph-SMOTE: SMOTE adapted for graph-structured data
"""
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import random
import json


class AdvancedGraphAugmentation:
    """Advanced graph augmentation preserving vulnerability patterns"""
    
    def __init__(self, target_ratio=0.3):
        self.target_ratio = target_ratio
        
    def create_variants(self, minority_samples, majority_count):
        """Create multiple variants of each vulnerable sample"""
        
        target_minority = int(majority_count * self.target_ratio / (1 - self.target_ratio))
        needed = max(0, target_minority - len(minority_samples))
        
        print(f"🔄 Creating {needed} graph variants from {len(minority_samples)} base samples")
        
        variants = []
        base_samples_cycle = minority_samples * (needed // len(minority_samples) + 1)
        
        for i in range(needed):
            base_sample = base_samples_cycle[i]
            variant = self.create_semantic_variant(base_sample['xfg'], i)
            
            if variant is not None:
                variant_sample = {
                    'path': f"variant_{i:05d}.xfg.pkl",
                    'xfg': variant,
                    'label': 1,
                    'synthetic': True,
                    'method': 'semantic_variant'
                }
                variants.append(variant_sample)
        
        return variants
    
    def create_semantic_variant(self, original_xfg, seed):
        """Create semantically meaningful variant"""
        np.random.seed(seed)
        variant = original_xfg.copy()
        
        # 1. Sensitive API substitution
        sensi_api_groups = {
            'memory': ['malloc', 'calloc', 'realloc', 'free'],
            'string': ['strcpy', 'strncpy', 'strcat', 'strncat', 'sprintf', 'snprintf'],
            'input': ['gets', 'fgets', 'scanf', 'sscanf'],
            'mavlink': ['mavlink_parse', 'mavlink_send', 'mavlink_receive']
        }
        
        for node in variant.nodes():
            if 'code_sym_token' in variant.nodes[node]:
                tokens = variant.nodes[node]['code_sym_token']
                new_tokens = []
                
                for token in tokens:
                    replaced = False
                    for group_name, apis in sensi_api_groups.items():
                        if token in apis and np.random.random() < 0.1:
                            # Replace with another API from same group
                            new_token = random.choice([api for api in apis if api != token])
                            new_tokens.append(new_token)
                            replaced = True
                            break
                    
                    if not replaced:
                        new_tokens.append(token)
                
                variant.nodes[node]['code_sym_token'] = new_tokens
        
        # 2. Graph structure variants
        self.add_vulnerability_patterns(variant)
        
        variant.graph['label'] = 1
        variant.graph['synthetic'] = True
        variant.graph['method'] = 'semantic_variant'
        
        return variant
    
    def add_vulnerability_patterns(self, graph):
        """Add common vulnerability patterns"""
        nodes = list(graph.nodes())
        
        if len(nodes) < 3:
            return
        
        # Pattern 1: Add buffer overflow pattern
        if np.random.random() < 0.2:
            array_nodes = []
            for node in nodes:
                if 'code_sym_token' in graph.nodes[node]:
                    tokens = graph.nodes[node]['code_sym_token']
                    if any('[' in str(token) and ']' in str(token) for token in tokens):
                        array_nodes.append(node)
            
            if array_nodes:
                array_node = random.choice(array_nodes)
                new_tokens = graph.nodes[array_node]['code_sym_token'].copy()
                if 'bounds_check' not in new_tokens:
                    new_tokens.append('unchecked_access')
                graph.nodes[array_node]['code_sym_token'] = new_tokens
        
        # Pattern 2: Add use-after-free pattern
        if np.random.random() < 0.15:
            for node in nodes:
                if 'code_sym_token' in graph.nodes[node]:
                    tokens = graph.nodes[node]['code_sym_token']
                    if 'free' in tokens or 'delete' in tokens:
                        tokens.append('ptr_use_after_free')
                        graph.nodes[node]['code_sym_token'] = tokens
                        break
        
        # ADDED: Ensure all edges have 'c/d' attribute
        for u, v, data in graph.edges(data=True):
            if 'c/d' not in data:
                data['c/d'] = 'd' if np.random.random() < 0.6 else 'c'

class GraphSMOTE:
    """SMOTE for graph data with topology preservation"""
    
    def __init__(self, k_neighbors=5, sampling_strategy=0.3):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy  # Target ratio for minority class

    def validate_and_fix_edges(self, graph):
        """Validate and fix missing 'c/d' attributes in edges"""
        fixed_count = 0
        
        for u, v, data in graph.edges(data=True):
            if 'c/d' not in data:
                # Assign realistic c/d based on heuristics
                # Data flow is more common (~60%) than control flow (~40%)
                data['c/d'] = 'd' if np.random.random() < 0.6 else 'c'
                fixed_count += 1
        
        if fixed_count > 0:
            print(f"    ✅ Fixed {fixed_count} missing 'c/d' attributes")
        
        return graph
        
    def extract_graph_features(self, xfg):
        """Extract numerical features from graph for SMOTE"""
        features = []
        
        # 1. Graph-level features
        features.extend([
            len(xfg.nodes()),                           # Node count
            len(xfg.edges()),                           # Edge count
            nx.density(xfg),                            # Density
            len(list(nx.connected_components(xfg.to_undirected()))),  # Components
            nx.average_clustering(xfg.to_undirected()) if len(xfg.nodes()) > 0 else 0,
        ])
        
        # 2. Node degree statistics
        degrees = [xfg.degree(n) for n in xfg.nodes()]
        if degrees:
            features.extend([
                np.mean(degrees),                       # Average degree
                np.std(degrees) if len(degrees) > 1 else 0,  # Degree std
                max(degrees),                           # Max degree
                min(degrees),                           # Min degree
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 3. Token-level features
        all_tokens = []
        token_lengths = []
        
        for node in xfg.nodes(data=True):
            if 'code_sym_token' in node[1]:
                tokens = node[1]['code_sym_token']
                all_tokens.extend(tokens)
                token_lengths.append(len(tokens))
        
        features.extend([
            len(set(all_tokens)),                      # Unique tokens
            np.mean(token_lengths) if token_lengths else 0,  # Avg tokens per node
            len(all_tokens),                           # Total tokens
        ])
        
        # 4. Control/Data flow features
        control_edges = sum(1 for u, v, d in xfg.edges(data=True) 
                           if d.get('c/d') == 'c')
        data_edges = sum(1 for u, v, d in xfg.edges(data=True) 
                        if d.get('c/d') == 'd')
        
        features.extend([
            control_edges,
            data_edges,
            control_edges / len(xfg.edges()) if len(xfg.edges()) > 0 else 0,
        ])
        
        return np.array(features)
    
    def fit_resample(self, minority_samples):
        """Generate synthetic minority samples using Graph-SMOTE"""
        print(f"🔄 Applying Graph-SMOTE to {len(minority_samples)} vulnerable samples...")
        
        # Extract features for all minority samples
        features = []
        for sample in minority_samples:
            feat = self.extract_graph_features(sample['xfg'])
            features.append(feat)
        
        features = np.array(features)
        
        # Fit k-NN on feature space
        if len(features) <= self.k_neighbors:
            k = len(features) - 1
        else:
            k = self.k_neighbors
            
        if k <= 0:
            print("⚠️  Not enough samples for Graph-SMOTE")
            return []
        
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(features)  # +1 because it includes self
        
        # Calculate number of synthetic samples needed
        current_minority = len(minority_samples)
        # For 30% target ratio: minority / (minority + majority) = 0.3
        # So majority should be: minority / 0.3 - minority = minority * (1/0.3 - 1) = minority * 2.33
        # We need: minority_new = majority * 0.3 / 0.7 = majority * 0.43
        
        # Estimate majority samples (this should be provided, but we estimate)
        estimated_majority = current_minority * 4  # Based on 10:1 ratio
        target_minority = int(estimated_majority * self.sampling_strategy / (1 - self.sampling_strategy))
        n_synthetic = max(0, target_minority - current_minority)
        
        print(f"  Current minority: {current_minority}")
        print(f"  Target minority: {target_minority}")
        print(f"  Generating: {n_synthetic} synthetic samples")
        
        synthetic_samples = []
        
        for i in range(n_synthetic):
            # Random sample from minority class
            sample_idx = random.randint(0, len(minority_samples) - 1)
            base_sample = minority_samples[sample_idx]
            
            # Find its k-NN
            distances, indices = nbrs.kneighbors([features[sample_idx]])
            neighbor_indices = indices[0][1:]  # Exclude self
            
            if len(neighbor_indices) == 0:
                # Fallback: use the base sample
                synthetic_xfg = self.simple_augment_graph(base_sample['xfg'], i)
            else:
                # Choose random neighbor
                neighbor_idx = random.choice(neighbor_indices)
                neighbor_sample = minority_samples[neighbor_idx]
                
                # Create synthetic sample between base and neighbor
                synthetic_xfg = self.interpolate_graphs(
                    base_sample['xfg'], 
                    neighbor_sample['xfg'], 
                    alpha=random.random(),
                    seed=i
                )
            
            if synthetic_xfg is not None:
                synthetic_xfg = self.validate_and_fix_edges(synthetic_xfg)

                synthetic_sample = {
                    'path': f"smote_synthetic_{i:05d}.xfg.pkl",
                    'xfg': synthetic_xfg,
                    'label': 1,
                    'synthetic': True,
                    'method': 'graph_smote'
                }
                synthetic_samples.append(synthetic_sample)
        
        print(f"  ✅ Generated {len(synthetic_samples)} synthetic vulnerable samples")
        return synthetic_samples
    
    def interpolate_graphs(self, graph1, graph2, alpha=0.5, seed=42):
        """Create synthetic graph by interpolating between two graphs"""
        np.random.seed(seed)
        
        # Choose the base graph (larger one for stability)
        if len(graph1.nodes()) >= len(graph2.nodes()):
            base_graph = graph1.copy()
            other_graph = graph2
        else:
            base_graph = graph2.copy()
            other_graph = graph1
        
        # 1. Node interpolation: blend token features
        base_nodes = list(base_graph.nodes())
        other_nodes = list(other_graph.nodes())
        
        for i, node in enumerate(base_nodes):
            if 'code_sym_token' in base_graph.nodes[node]:
                base_tokens = base_graph.nodes[node]['code_sym_token']
                
                # Find corresponding node in other graph
                if i < len(other_nodes) and 'code_sym_token' in other_graph.nodes[other_nodes[i]]:
                    other_tokens = other_graph.nodes[other_nodes[i]]['code_sym_token']
                    
                    # Token blending
                    if np.random.random() < alpha:
                        # Use tokens from other graph
                        if len(other_tokens) > 0:
                            blend_ratio = np.random.uniform(0.3, 0.7)
                            n_from_other = int(len(base_tokens) * blend_ratio)
                            n_from_other = min(n_from_other, len(other_tokens))
                            
                            if n_from_other > 0:
                                selected_other = random.sample(other_tokens, n_from_other)
                                # Replace some base tokens
                                n_replace = min(n_from_other, len(base_tokens))
                                for j in range(n_replace):
                                    if j < len(selected_other):
                                        base_tokens[j] = selected_other[j]
                
                base_graph.nodes[node]['code_sym_token'] = base_tokens
        
        # 2. Edge interpolation: selectively add/remove edges
        other_edges = list(other_graph.edges(data=True))
        
        # Add some edges from other graph WITH proper 'c/d' attribute
        for edge in other_edges:
            u, v, data = edge
            if u < len(base_nodes) and v < len(base_nodes):
                base_u = base_nodes[u] if u < len(base_nodes) else base_nodes[0]
                base_v = base_nodes[v] if v < len(base_nodes) else base_nodes[-1]
                
                if not base_graph.has_edge(base_u, base_v) and np.random.random() < 0.1:
                    # FIXED: Ensure 'c/d' attribute exists
                    edge_data = data.copy()
                    if 'c/d' not in edge_data:
                        edge_data['c/d'] = 'd' if np.random.random() < 0.6 else 'c'
                    base_graph.add_edge(base_u, base_v, **edge_data)
        
        # Remove some edges randomly
        current_edges = list(base_graph.edges())
        n_remove = int(len(current_edges) * 0.05)  # Remove 5%
        if n_remove > 0 and len(current_edges) > n_remove:
            edges_to_remove = random.sample(current_edges, n_remove)
            base_graph.remove_edges_from(edges_to_remove)
        
        # Ensure graph is still connected
        if len(base_graph.nodes()) > 1 and not nx.is_weakly_connected(base_graph):
            # Add edges to make it connected WITH proper 'c/d' attribute
            components = list(nx.weakly_connected_components(base_graph))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                # FIXED: Add proper 'c/d' attribute
                base_graph.add_edge(node1, node2, synthetic=True, **{'c/d': 'd'})
        
        # Update metadata
        base_graph.graph['label'] = 1
        base_graph.graph['synthetic'] = True
        base_graph.graph['method'] = 'graph_smote'
        base_graph.graph['alpha'] = alpha
        
        return base_graph
    
    def simple_augment_graph(self, graph, seed):
        """Fallback: simple graph augmentation"""
        np.random.seed(seed)
        aug_graph = graph.copy()
        
        # Token shuffling
        for node in aug_graph.nodes():
            if 'code_sym_token' in aug_graph.nodes[node]:
                tokens = aug_graph.nodes[node]['code_sym_token'].copy()
                if len(tokens) > 1 and np.random.random() < 0.2:
                    random.shuffle(tokens)
                    aug_graph.nodes[node]['code_sym_token'] = tokens
        
        aug_graph.graph['label'] = 1
        aug_graph.graph['synthetic'] = True
        aug_graph.graph['method'] = 'simple_augment'
        
        return aug_graph
    



#!/usr/bin/env python3
"""
Complete Graph-Aware Dataset Balancer
"""

class GraphDatasetBalancer:
    def __init__(self, original_path, output_path, method='graph_smote', target_ratio=0.25):
        self.original_path = Path(original_path)
        self.output_path = Path(output_path)
        self.method = method
        self.target_ratio = target_ratio
        
    def load_original_samples(self):
        """Load original unbalanced data"""
        all_samples = []
        
        for split in ['train', 'val', 'test']:
            split_file = self.original_path / f"{split}.json"
            with open(split_file, 'r') as f:
                paths = json.load(f)
            
            for path in paths:
                try:
                    with open(path, 'rb') as f:
                        xfg = pickle.load(f)
                    
                    all_samples.append({
                        'path': path,
                        'xfg': xfg,
                        'label': xfg.graph.get('label', -1)
                    })
                except:
                    continue
        
        return all_samples
    
    def balance_dataset(self):
        """Main balancing function"""
        print("🔄 Starting graph-aware dataset balancing...")
        
        # Load original data
        all_samples = self.load_original_samples()
        
        # Separate by class
        minority_samples = [s for s in all_samples if s['label'] == 1]
        majority_samples = [s for s in all_samples if s['label'] == 0]
        
        print(f"  Original: {len(minority_samples)} vulnerable, {len(majority_samples)} safe")
        
        # Generate synthetic samples
        if self.method == 'graph_smote':
            smote = GraphSMOTE(sampling_strategy=self.target_ratio)
            synthetic_samples = smote.fit_resample(minority_samples)
        elif self.method == 'advanced_aug':
            augmenter = AdvancedGraphAugmentation(target_ratio=self.target_ratio)
            synthetic_samples = augmenter.create_variants(minority_samples, len(majority_samples))
        else:
            synthetic_samples = []
        
        # Combine all samples
        all_balanced = minority_samples + synthetic_samples + majority_samples
        random.shuffle(all_balanced)
        
        print(f"  Balanced: {len(minority_samples + synthetic_samples)} vulnerable, {len(majority_samples)} safe")
        
        # Create new splits
        return self.create_splits(all_balanced)
    
    def create_splits(self, balanced_samples):
        """Create train/val/test splits"""
        total = len(balanced_samples)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        
        train = balanced_samples[:train_size]
        val = balanced_samples[train_size:train_size + val_size]
        test = balanced_samples[train_size + val_size:]
        
        return self.save_splits(train, val, test)
    
    def save_splits(self, train, val, test):
        """Save balanced splits with edge validation"""
        self.output_path.mkdir(parents=True, exist_ok=True)
        xfg_dir = self.output_path / "XFG"
        xfg_dir.mkdir(exist_ok=True)
        
        split_paths = {}
        
        for split_name, samples in [('train', train), ('val', val), ('test', test)]:
            paths = []
            fixed_count = 0
            
            for i, sample in enumerate(samples):
                # ADDED: Validate each graph before saving
                graph = sample['xfg']
                
                # Fix missing 'c/d' attributes
                for u, v, data in graph.edges(data=True):
                    if 'c/d' not in data:
                        data['c/d'] = 'd' if np.random.random() < 0.6 else 'c'
                        fixed_count += 1
                
                label_str = "vuln" if sample['label'] == 1 else "safe"
                method_str = sample.get('method', 'original')
                filename = f"{split_name}_{label_str}_{method_str}_{i:05d}.xfg.pkl"
                path = xfg_dir / filename
                
                with open(path, 'wb') as f:
                    pickle.dump(sample['xfg'], f, pickle.HIGHEST_PROTOCOL)
                
                paths.append(str(path))
            
            if fixed_count > 0:
                print(f"    ✅ Fixed {fixed_count} edge attributes in {split_name}")
            
            split_paths[split_name] = paths
            
            # Save split file
            with open(self.output_path / f"{split_name}.json", 'w') as f:
                json.dump(paths, f, indent=2)
        
        print(f"✅ Balanced dataset saved to {self.output_path}")
        return self.output_path

#!/usr/bin/env python3
"""
Fixed Graph Balancer - Run this script
"""
import sys
sys.path.append('.')

# Copy the fixed classes above into this file or import them

def main():
    print("🔧 Running Fixed Graph-SMOTE Balancing...")
    
    balancer = GraphDatasetBalancer(
        original_path="data/UAV",
        output_path="data/UAV/graph_balanced",
        method='graph_smote',
        target_ratio=0.35
    )
    
    balanced_path = balancer.balance_dataset()
    
    print(f"\n✅ Fixed balanced dataset created at: {balanced_path}")
    print("🎯 All graphs now have proper 'c/d' edge attributes")
    
    # Update config to use new dataset
    print(f"\n📝 Update your config:")
    print(f"dataset.name: UAV/graph_balanced")

if __name__ == "__main__":
    main()