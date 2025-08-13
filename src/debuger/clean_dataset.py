#!/usr/bin/env python3
"""
Clean UAV Dataset Creator - No Artificial Augmentation
"""
import json
import pickle
import random
import networkx as nx
from pathlib import Path
from collections import Counter
import hashlib

class CleanDatasetCreator:
    def __init__(self, original_path, output_path):
        self.original_path = Path(original_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def load_original_data(self):
        """Load from original unbalanced data"""
        print("📥 Loading original high-quality data...")
        
        # Go back to original data before augmentation
        original_splits = ['train', 'val', 'test'] 
        all_samples = []
        
        for split in original_splits:
            split_file = self.original_path / f"{split}.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    paths = json.load(f)
                
                for path in paths:
                    try:
                        with open(path, 'rb') as f:
                            xfg = pickle.load(f)
                        
                        # ONLY keep original samples (no augmented)
                        if not xfg.graph.get('augmented', False):
                            all_samples.append({
                                'path': path,
                                'xfg': xfg,
                                'label': xfg.graph.get('label', -1)
                            })
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
        
        print(f"  Loaded {len(all_samples)} original samples")
        return all_samples
    
    def quality_filter(self, samples):
        """Apply quality filters"""
        print("🔍 Applying quality filters...")
        
        filtered = []
        stats = {
            'too_small': 0,
            'too_large': 0, 
            'no_tokens': 0,
            'duplicates': 0,
            'passed': 0
        }
        
        seen_signatures = set()
        
        for sample in samples:
            xfg = sample['xfg']
            
            # Filter 1: Size constraints
            if len(xfg.nodes()) < 2:
                stats['too_small'] += 1
                continue
            if len(xfg.nodes()) > 200:
                stats['too_large'] += 1
                continue
            
            # Filter 2: Token validation
            has_meaningful_tokens = False
            for node in xfg.nodes(data=True):
                if 'code_sym_token' in node[1]:
                    tokens = node[1]['code_sym_token']
                    if len(tokens) > 0 and any(len(t) > 1 for t in tokens):
                        has_meaningful_tokens = True
                        break
            
            if not has_meaningful_tokens:
                stats['no_tokens'] += 1
                continue
            
            # Filter 3: Deduplication
            signature = self.get_graph_signature(xfg)
            if signature in seen_signatures:
                stats['duplicates'] += 1
                continue
            
            seen_signatures.add(signature)
            filtered.append(sample)
            stats['passed'] += 1
        
        print(f"  Quality filter results:")
        for key, count in stats.items():
            print(f"    {key}: {count}")
        
        return filtered
    
    def get_graph_signature(self, xfg):
        """Create unique signature for graph"""
        # Use structure + content
        nodes_content = []
        for node in sorted(xfg.nodes()):
            if 'code_sym_token' in xfg.nodes[node]:
                tokens = xfg.nodes[node]['code_sym_token']
                nodes_content.append('|'.join(tokens))
        
        edges = sorted(list(xfg.edges()))
        signature_str = f"nodes:{':'.join(nodes_content)}|edges:{str(edges)}"
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def smart_balanced_split(self, samples, target_ratio=0.25):
        """Create balanced splits with smart sampling"""
        print(f"⚖️ Creating balanced splits (target: {target_ratio:.1%} vulnerable)...")
        
        # Separate by label
        vulnerable = [s for s in samples if s['label'] == 1]
        safe = [s for s in samples if s['label'] == 0]
        
        print(f"  Available: {len(vulnerable)} vulnerable, {len(safe)} safe")
        
        # Calculate balanced sizes
        if len(vulnerable) * 3 < len(safe):  # If we have 3x more safe samples
            # Use all vulnerable + sample safe
            max_safe = int(len(vulnerable) / target_ratio * (1 - target_ratio))
            selected_safe = random.sample(safe, min(max_safe, len(safe)))
        else:
            # Use all safe + sample vulnerable (rare case)
            max_vulnerable = int(len(safe) * target_ratio / (1 - target_ratio))
            vulnerable = random.sample(vulnerable, min(max_vulnerable, len(vulnerable)))
            selected_safe = safe
        
        # Combine and shuffle
        balanced_samples = vulnerable + selected_safe
        random.shuffle(balanced_samples)
        
        print(f"  Balanced dataset: {len(vulnerable)} vulnerable + {len(selected_safe)} safe = {len(balanced_samples)}")
        
        # Split into train/val/test
        total = len(balanced_samples)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        
        train_samples = balanced_samples[:train_size]
        val_samples = balanced_samples[train_size:train_size + val_size]
        test_samples = balanced_samples[train_size + val_size:]
        
        return train_samples, val_samples, test_samples
    
    def save_clean_dataset(self, train_samples, val_samples, test_samples):
        """Save the clean dataset"""
        print("💾 Saving clean dataset...")
        
        xfg_dir = self.output_path / "XFG"
        xfg_dir.mkdir(exist_ok=True)
        
        split_paths = {'train': [], 'val': [], 'test': []}
        
        for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            for i, sample in enumerate(samples):
                # New clean filename
                label_str = "vuln" if sample['label'] == 1 else "safe"
                filename = f"clean_{split_name}_{label_str}_{i:05d}.xfg.pkl"
                new_path = xfg_dir / filename
                
                # Save XFG
                with open(new_path, 'wb') as f:
                    pickle.dump(sample['xfg'], f, pickle.HIGHEST_PROTOCOL)
                
                split_paths[split_name].append(str(new_path))
        
        # Save split files
        for split_name, paths in split_paths.items():
            split_file = self.output_path / f"{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump(paths, f, indent=2)
            print(f"  ✅ {split_name}: {len(paths)} samples")
        
        return self.output_path

def create_clean_dataset():
    # Use ORIGINAL data (before balancing)
    creator = CleanDatasetCreator("data/UAV", "data/UAV/clean")
    
    # Load original high-quality samples
    samples = creator.load_original_data()
    
    # Apply quality filters
    filtered_samples = creator.quality_filter(samples)
    
    # Create balanced splits
    train, val, test = creator.smart_balanced_split(filtered_samples)
    
    # Save clean dataset
    output_path = creator.save_clean_dataset(train, val, test)
    
    print(f"\n✅ Clean dataset created: {output_path}")
    return output_path

# Create clean dataset
clean_path = create_clean_dataset()