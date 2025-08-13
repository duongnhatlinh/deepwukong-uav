#!/usr/bin/env python3
"""
Advanced UAV Dataset Validator
"""
import pickle
import networkx as nx
from pathlib import Path
import json
import hashlib
from collections import Counter, defaultdict

class AdvancedDataValidator:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
    
    def check_augmentation_artifacts(self):
        """Check if augmented samples are too easy to distinguish"""
        print("🔍 Checking augmentation artifacts...")
        
        with open(self.data_path / "train.json") as f:
            train_paths = json.load(f)
        
        augmented_count = 0
        real_count = 0
        
        for path in train_paths[:1000]:  # Sample check
            try:
                with open(path, 'rb') as f:
                    xfg = pickle.load(f)
                
                if xfg.graph.get('augmented', False):
                    augmented_count += 1
                else:
                    real_count += 1
                    
            except Exception as e:
                print(f"Error reading {path}: {e}")
        
        print(f"Real samples: {real_count}")
        print(f"Augmented samples: {augmented_count}")
        
        if augmented_count > real_count * 2:
            print("⚠️  TOO MANY AUGMENTED SAMPLES - Model may learn artifacts!")
            
    def check_token_leakage(self):
        """Check for token-level data leakage"""
        print("🔍 Checking token-level patterns...")
        
        train_tokens = set()
        val_tokens = set()
        
        # Sample from train/val
        for split in ['train', 'val']:
            with open(self.data_path / f"{split}.json") as f:
                paths = json.load(f)
            
            for path in paths[:100]:
                try:
                    with open(path, 'rb') as f:
                        xfg = pickle.load(f)
                    
                    for node in xfg.nodes(data=True):
                        if 'code_sym_token' in node[1]:
                            tokens = tuple(node[1]['code_sym_token'])
                            if split == 'train':
                                train_tokens.add(tokens)
                            else:
                                val_tokens.add(tokens)
                except:
                    continue
        
        overlap = train_tokens & val_tokens
        print(f"Token sequence overlap: {len(overlap)}/{len(val_tokens)} = {len(overlap)/len(val_tokens)*100:.1f}%")
        
        if len(overlap) / len(val_tokens) > 0.8:
            print("⚠️  HIGH TOKEN OVERLAP - Data leakage detected!")

# Run validator
validator = AdvancedDataValidator("data/UAV")
validator.check_augmentation_artifacts()
validator.check_token_leakage()