#!/usr/bin/env python3
"""
Quick debug script to check dataset balance and fix issues
"""
import json
import pickle
from pathlib import Path

def check_dataset_balance(data_path):
    """Check if dataset is properly balanced"""
    data_path = Path(data_path)
    
    print(f"🔍 Checking dataset: {data_path}")
    
    # Check if balanced dataset exists
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        return False
    
    # Check splits
    for split in ['train', 'val', 'test']:
        split_file = data_path / f"{split}.json"
        if not split_file.exists():
            print(f"❌ Missing {split}.json")
            return False
        
        with open(split_file, 'r') as f:
            paths = json.load(f)
        
        # Sample a few files to check labels
        vuln_count = 0
        safe_count = 0
        sample_size = min(100, len(paths))
        
        for i, path in enumerate(paths[:sample_size]):
            try:
                with open(path, 'rb') as f:
                    xfg = pickle.load(f)
                label = xfg.graph.get('label', -1)
                if label == 1:
                    vuln_count += 1
                elif label == 0:
                    safe_count += 1
            except:
                continue
            
            if i >= sample_size - 1:
                break
        
        total_sampled = vuln_count + safe_count
        vuln_ratio = vuln_count / total_sampled if total_sampled > 0 else 0
        
        print(f"📊 {split}: {len(paths):,} total, sample shows {vuln_ratio:.1%} vulnerable")
        
        if vuln_ratio < 0.25:
            print(f"⚠️  {split} seems imbalanced (only {vuln_ratio:.1%} vulnerable)")
            return False
    
    print(f"✅ Dataset appears balanced!")
    return True

def create_balanced_dataset_if_needed():
    """Create balanced dataset if it doesn't exist"""
    
    # Check if moderate_balanced exists
    moderate_path = Path("data/UAV/moderate_balanced")
    
    if check_dataset_balance(moderate_path):
        print("✅ Balanced dataset already exists and looks good!")
        return moderate_path
    
    print("🔄 Creating balanced dataset...")
    
    # Import and run the balancer
    import sys
    sys.path.append('.')
    
    try:
        # Try to import the moderate balancer
        from src.debuger.moderate_hybrid_balancer import ModerateHybridBalancer
        
        balancer = ModerateHybridBalancer(
            original_path="data/UAV",
            output_path="data/UAV/moderate_balanced",
            target_ratio=0.35,
            max_vuln_multiplier=2.5,
            undersample_strategy='cluster_centroids'
        )
        
        balanced_path = balancer.balance_dataset()
        print(f"✅ Created balanced dataset at: {balanced_path}")
        return balanced_path
        
    except Exception as e:
        print(f"❌ Failed to create balanced dataset: {e}")
        print("💡 Please run manually:")
        print("python src/debuger/moderate_hybrid_balancer.py")
        return None

def check_word2vec():
    """Check if word2vec file exists for balanced dataset"""
    w2v_path = Path("data/UAV/moderate_balanced/w2v.wv")
    
    if w2v_path.exists():
        print(f"✅ Word2Vec found: {w2v_path}")
        return True
    else:
        print(f"❌ Word2Vec missing: {w2v_path}")
        print("💡 Please train word2vec:")
        print("python src/preprocess/word_embedding.py -c configs/dwk_fixed.yaml")
        return False

def main():
    print("🔧 Quick Debug: Dataset Balance Check")
    print("="*50)
    
    # Step 1: Check original dataset
    print("\n1️⃣ Checking original dataset...")
    original_balanced = check_dataset_balance("data/UAV")
    
    # Step 2: Check/create balanced dataset
    print("\n2️⃣ Checking balanced dataset...")
    balanced_path = create_balanced_dataset_if_needed()
    
    # Step 3: Check word2vec
    print("\n3️⃣ Checking word2vec...")
    w2v_exists = check_word2vec()
    
    # Summary and recommendations
    print("\n" + "="*50)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*50)
    
    if balanced_path and w2v_exists:
        print("✅ Everything looks good! You can train with:")
        print("PYTHONPATH='.' python src/run.py -c configs/dwk_fixed.yaml")
    else:
        print("❌ Issues found. Please fix:")
        if not balanced_path:
            print("  • Create balanced dataset: python src/debuger/moderate_hybrid_balancer.py")
        if not w2v_exists:
            print("  • Train word2vec: python src/preprocess/word_embedding.py -c configs/dwk_fixed.yaml")
    
    print(f"\n🎯 Expected balanced dataset stats:")
    print(f"  • ~25,955 vulnerable (35%)")
    print(f"  • ~48,490 safe (65%)")
    print(f"  • Total: ~74,445 samples")
    print(f"  • Much better for training!")

if __name__ == "__main__":
    main()