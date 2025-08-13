#!/usr/bin/env python3
"""
XFG Duplicate Analysis Tool for DeepWukong
Phân tích số lượng XFG trùng lặp, xung đột và thống kê chi tiết
"""

import os
import hashlib
import networkx as nx
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from os.path import join, exists
import pickle
import random
import json


def read_gpickle(filename):
    try:
        with open(filename, 'rb') as f:
            graph = pickle.load(f)
        return graph
    except Exception as e:
        print(f"Error reading gpickle file: {e}")

def getMD5(s: str) -> str:
    """Tạo MD5 hash cho string"""
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()


def analyze_xfg_duplicates(xfg_paths: List[str], verbose: bool = True) -> Dict:
    """
    Phân tích chi tiết các XFG trùng lặp và xung đột
    
    Args:
        xfg_paths: List đường dẫn đến các file XFG
        verbose: In chi tiết quá trình xử lý
        
    Returns:
        Dictionary chứa thống kê chi tiết
    """
    
    # Dictionary để lưu thông tin XFG theo MD5
    md5_to_xfgs = defaultdict(list)
    
    # Counters
    total_xfgs = len(xfg_paths)
    processed = 0
    
    # Thêm counters cho label=1 và label=0 ban đầu
    original_label1 = 0
    original_label0 = 0

    print(f"🔍 Analyzing {total_xfgs} XFGs...")
    
    # Process each XFG
    for xfg_path in tqdm(xfg_paths, desc="Processing XFGs"):
        try:
            xfg = read_gpickle(xfg_path)
            label = xfg.graph["label"]
            
            # Đếm số lượng label=1 và label=0 ban đầu
            if label == 1:
                original_label1 += 1
            elif label == 0:
                original_label0 += 1
            
            # Tạo MD5 cho mỗi node dựa trên symbolic token
            for ln in xfg:
                if "code_sym_token" in xfg.nodes[ln]:
                    ln_md5 = getMD5(str(xfg.nodes[ln]["code_sym_token"]))
                    xfg.nodes[ln]["md5"] = ln_md5
                else:
                    # Fallback nếu chưa có symbolic token
                    ln_md5 = getMD5(str(ln))
                    xfg.nodes[ln]["md5"] = ln_md5
            
            # Tạo MD5 cho toàn bộ XFG dựa trên cấu trúc edges
            edges_md5 = []
            for edge in xfg.edges:
                edge_md5 = xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"]
                edges_md5.append(edge_md5)
            
            # Tạo unique ID cho XFG
            xfg_md5 = getMD5(str(sorted(edges_md5)))
            
            # Lưu thông tin XFG
            xfg_info = {
                'path': xfg_path,
                'label': label,
                'num_nodes': len(xfg.nodes),
                'num_edges': len(xfg.edges),
                'file_path': xfg.graph.get("file_paths", ["unknown"])[0],
                'key_line': xfg.graph.get("key_line", -1)
            }
            
            md5_to_xfgs[xfg_md5].append(xfg_info)
            processed += 1
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Error processing {xfg_path}: {e}")
            continue
    
    print(f"✅ Successfully processed {processed}/{total_xfgs} XFGs")
    
    # Analyze duplicates and conflicts
    stats = analyze_duplicate_statistics(md5_to_xfgs, verbose)
    
    # Thêm số lượng label=1 và label=0 ban đầu vào stats
    stats['original_label1'] = original_label1
    stats['original_label0'] = original_label0
    
    return stats


def analyze_duplicate_statistics(md5_to_xfgs: Dict, verbose: bool = True) -> Dict:
    """
    Phân tích thống kê chi tiết về duplicates và conflicts
    """
    
    # Counters
    unique_xfgs = 0
    duplicate_groups = 0
    conflict_groups = 0
    
    vul_duplicates = 0  # XFG vulnerable trùng lặp
    safe_duplicates = 0  # XFG safe trùng lặp
    conflicts = 0  # XFG xung đột
    
    # Chi tiết về duplicates
    vul_duplicate_details = []
    safe_duplicate_details = []
    conflict_details = []
    
    # Thêm counters cho label=1 và label=0 sau khi deduplicate
    dedup_label1 = 0
    dedup_label0 = 0
    
    # Thêm counters cho unique label=1 và label=0
    unique_label1 = 0
    unique_label0 = 0
    
    # Thêm list để lưu trữ paths cho việc tạo dataset cân bằng
    unique_label1_paths = []
    unique_label0_paths = []
    dedup_vul_details = [] # list of {'path': path, 'count': count}
    dedup_safe_details = [] # list of {'path': path, 'count': count}
    
    print("\n📊 ANALYZING DUPLICATE STATISTICS...")
    print("=" * 60)
    
    for xfg_md5, xfg_list in md5_to_xfgs.items():
        if len(xfg_list) == 1:
            # Unique XFG
            unique_xfgs += 1
            # Đếm label cho unique XFG
            label = xfg_list[0]['label']
            path = xfg_list[0]['path']
            if label == 1:
                dedup_label1 += 1
                unique_label1 += 1
                unique_label1_paths.append(path)
            elif label == 0:
                dedup_label0 += 1
                unique_label0 += 1
                unique_label0_paths.append(path)
        else:
            # Multiple XFGs with same structure
            labels = set([xfg['label'] for xfg in xfg_list])
            
            if len(labels) == 1:
                # Duplicates (same structure, same label)
                duplicate_groups += 1
                label = list(labels)[0]
                # Chỉ giữ lại 1 bản đại diện cho dedup
                path = xfg_list[0]['path']
                count = len(xfg_list)
                if label == 1:
                    dedup_label1 += 1
                    dedup_vul_details.append({'path': path, 'count': count})
                elif label == 0:
                    dedup_label0 += 1
                    dedup_safe_details.append({'path': path, 'count': count})
                
                if label == 1:  # Vulnerable
                    vul_duplicates += len(xfg_list) - 1  # Trừ 1 vì chỉ đếm bản sao
                    vul_duplicate_details.append({
                        'md5': xfg_md5,
                        'count': len(xfg_list),
                        'xfgs': xfg_list
                    })
                elif label == 0:  # Safe
                    safe_duplicates += len(xfg_list) - 1
                    safe_duplicate_details.append({
                        'md5': xfg_md5,
                        'count': len(xfg_list),
                        'xfgs': xfg_list
                    })
                
                if verbose:
                    label_str = "VUL" if label == 1 else "SAFE"
                    print(f"🔄 DUPLICATE [{label_str}]: {len(xfg_list)} XFGs with same structure")
                    for i, xfg in enumerate(xfg_list[:3]):  # Show first 3
                        print(f"   {i+1}. {xfg['path']} (nodes: {xfg['num_nodes']}, edges: {xfg['num_edges']})")
                    if len(xfg_list) > 3:
                        print(f"   ... and {len(xfg_list) - 3} more")
                    print()
            else:
                # Conflicts (same structure, different labels)
                conflict_groups += 1
                conflicts += len(xfg_list)
                conflict_details.append({
                    'md5': xfg_md5,
                    'count': len(xfg_list),
                    'labels': list(labels),
                    'xfgs': xfg_list
                })
                # Không cộng vào dedup vì xung đột không giữ lại
                if verbose:
                    print(f"⚠️  CONFLICT: {len(xfg_list)} XFGs with same structure but different labels {list(labels)}")
                    for i, xfg in enumerate(xfg_list):
                        label_str = "VUL" if xfg['label'] == 1 else "SAFE"
                        print(f"   {i+1}. [{label_str}] {xfg['path']} (nodes: {xfg['num_nodes']}, edges: {xfg['num_edges']})")
                    print()
    # Tạo summary statistics
    total_original_xfgs = sum(len(xfg_list) for xfg_list in md5_to_xfgs.values())
    
    # Thống kê số lượng nhóm duplicate theo số lượng file duplicate
    from collections import Counter
    # Tách thống kê cho label=1 và label=0
    vul_duplicate_group_size_stats = Counter()
    for group in vul_duplicate_details:
        sz = group['count']
        vul_duplicate_group_size_stats[sz] += 1
    vul_duplicate_group_size_stats = dict(sorted(vul_duplicate_group_size_stats.items()))
    
    safe_duplicate_group_size_stats = Counter()
    for group in safe_duplicate_details:
        sz = group['count']
        safe_duplicate_group_size_stats[sz] += 1
    safe_duplicate_group_size_stats = dict(sorted(safe_duplicate_group_size_stats.items()))

    stats = {
        'total_original_xfgs': total_original_xfgs,
        'unique_structures': len(md5_to_xfgs),
        'unique_xfgs': unique_xfgs,
        'unique_label1': unique_label1,
        'unique_label0': unique_label0,
        'duplicate_groups': duplicate_groups,
        'conflict_groups': conflict_groups,
        
        'vul_duplicates': vul_duplicates,
        'safe_duplicates': safe_duplicates,
        'total_duplicates': vul_duplicates + safe_duplicates,
        'conflicts': conflicts,
        
        'vul_duplicate_details': vul_duplicate_details,
        'safe_duplicate_details': safe_duplicate_details,
        'conflict_details': conflict_details,
        
        'final_dataset_size': dedup_label1 + dedup_label0,  # Số lượng XFG sau khi loại duplicates và conflicts
        'dedup_label1': dedup_label1,
        'dedup_label0': dedup_label0,
        'vul_duplicate_group_size_stats': vul_duplicate_group_size_stats,
        'safe_duplicate_group_size_stats': safe_duplicate_group_size_stats,
        
        # Paths for balancing
        'unique_label1_paths': unique_label1_paths,
        'unique_label0_paths': unique_label0_paths,
        'dedup_vul_details': dedup_vul_details,
        'dedup_safe_details': dedup_safe_details,
    }
    
    return stats


def print_summary_report(stats: Dict):
    """In báo cáo tổng kết"""
    
    print("\n" + "="*80)
    print("📋 SUMMARY REPORT")
    print("="*80)
    
    print(f"📦 Total Original XFGs: {stats['total_original_xfgs']}")
    print(f"   • Original label=1: {stats.get('original_label1', '?')}")
    print(f"   • Original label=0: {stats.get('original_label0', '?')}")
    print(f"🔧 Unique Structures: {stats['unique_structures']}")
    print(f"✨ Unique XFGs: {stats['unique_xfgs']}")
    print(f"   • Unique label=1: {stats.get('unique_label1', '?')}")
    print(f"   • Unique label=0: {stats.get('unique_label0', '?')}")
    print()
    
    print("🔄 DUPLICATES:")
    print(f"   • Vulnerable (label=1) duplicates: {stats['vul_duplicates']} XFGs")
    print(f"   • Safe (label=0) duplicates: {stats['safe_duplicates']} XFGs") 
    print(f"   • Total duplicates: {stats['total_duplicates']} XFGs")
    print(f"   • Duplicate groups: {stats['duplicate_groups']} groups")
    print(f"   • VUL duplicate group size stats (group_size: num_groups): {stats.get('vul_duplicate_group_size_stats', {})}")
    print(f"   • SAFE duplicate group size stats (group_size: num_groups): {stats.get('safe_duplicate_group_size_stats', {})}")
    print()
    
    print("⚠️  CONFLICTS:")
    print(f"   • XFGs with conflicts: {stats['conflicts']} XFGs")
    print(f"   • Conflict groups: {stats['conflict_groups']} groups")
    print()
    
    print("📊 FINAL DATASET:")
    print(f"   • After deduplication: {stats['final_dataset_size']} XFGs")
    print(f"   • Deduplicated label=1: {stats.get('dedup_label1', '?')}")
    print(f"   • Deduplicated label=0: {stats.get('dedup_label0', '?')}")
    print(f"   • Reduction rate: {((stats['total_original_xfgs'] - stats['final_dataset_size']) / stats['total_original_xfgs'] * 100):.1f}%")
    
    print("\n" + "="*80)


def create_balanced_dataset(stats: Dict, output_file: str):
    """
    Tạo bộ dữ liệu cân bằng theo tỉ lệ 3:7 (vul:safe) và lưu danh sách file
    """
    print("⚖️  CREATING BALANCED DATASET...")
    
    # 1. Lấy toàn bộ XFG label=1 sau deduplicate
    vul_paths = stats['unique_label1_paths'] + [item['path'] for item in stats['dedup_vul_details']]
    num_vul = len(vul_paths)
    
    # 2. Tính toán số lượng XFG label=0 cần thiết
    target_num_safe = int(num_vul / 3 * 7)
    
    print(f"   • Target Vul (label=1): {num_vul}")
    print(f"   • Target Safe (label=0): {target_num_safe}")
    
    # 3. Lấy XFG label=0 từ các nguồn ưu tiên
    selected_safe_paths = []
    
    # Ưu tiên 1: Lấy từ các nhóm duplicate lớn (size >= 10)
    safe_from_large_groups = [
        item['path'] for item in stats['dedup_safe_details'] 
        if item['count'] >= 10
    ]
    selected_safe_paths.extend(safe_from_large_groups)
    
    print(f"   • Selected {len(selected_safe_paths)} samples from large duplicate groups (size >= 10)")
    
    # Ưu tiên 2: Lấy ngẫu nhiên từ các XFG unique label=0
    needed = target_num_safe - len(selected_safe_paths)
    if needed > 0:
        unique_safe_paths = stats['unique_label0_paths']
        if needed >= len(unique_safe_paths):
            # Nếu cần nhiều hơn số unique hiện có, lấy hết
            selected_safe_paths.extend(unique_safe_paths)
            print(f"   • Selected all {len(unique_safe_paths)} unique samples")
        else:
            # Nếu không, lấy ngẫu nhiên
            random_unique_paths = random.sample(unique_safe_paths, needed)
            selected_safe_paths.extend(random_unique_paths)
            print(f"   • Randomly selected {len(random_unique_paths)} unique samples")
    
    final_num_safe = len(selected_safe_paths)

    # 4. Kết hợp và lưu file
    final_paths = vul_paths + selected_safe_paths
    random.shuffle(final_paths) # Xáo trộn bộ dữ liệu cuối cùng
    
    report = {
        'description': 'Balanced dataset with Vul:Safe ratio ~3:7',
        'num_vulnerable': num_vul,
        'num_safe': final_num_safe,
        'total': len(final_paths),
        'paths': final_paths
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "-"*60)
    print(f"✅ Balanced dataset created successfully!")
    print(f"   • Final Vul (label=1): {num_vul}")
    print(f"   • Final Safe (label=0): {final_num_safe}")
    print(f"   • Total XFGs: {len(final_paths)}")
    print(f"   • Saved to: {output_file}")
    print("-" * 60)


def get_all_xfg_paths(data_folder: str, dataset_name: str) -> List[str]:
    """
    Lấy tất cả đường dẫn XFG từ thư mục dữ liệu
    """
    XFG_root_path = join(data_folder, dataset_name, "XFG")
    
    if not exists(XFG_root_path):
        raise FileNotFoundError(f"XFG directory not found: {XFG_root_path}")
    
    xfg_paths = []
    testcaseids = os.listdir(XFG_root_path)
    
    for testcase in testcaseids:
        testcase_root_path = join(XFG_root_path, testcase)
        if not os.path.isdir(testcase_root_path):
            continue
            
        for k in ["arith", "array", "call", "ptr"]:
            k_root_path = join(testcase_root_path, k)
            if not exists(k_root_path):
                continue
                
            xfg_files = os.listdir(k_root_path)
            for xfg_file in xfg_files:
                if xfg_file.endswith('.xfg.pkl'):
                    xfg_path = join(k_root_path, xfg_file)
                    xfg_paths.append(xfg_path)
    
    return xfg_paths


def save_detailed_report(stats: Dict, output_file: str):
    """Lưu báo cáo chi tiết ra file"""
    
    import json
    
    # Convert để có thể serialize JSON
    report = {
        'summary': {
            'total_original_xfgs': stats['total_original_xfgs'],
            'unique_structures': stats['unique_structures'],
            'unique_xfgs': stats['unique_xfgs'],
            'vul_duplicates': stats['vul_duplicates'],
            'safe_duplicates': stats['safe_duplicates'],
            'conflicts': stats['conflicts'],
            'final_dataset_size': stats['final_dataset_size']
        },
        'vul_duplicate_groups': len(stats['vul_duplicate_details']),
        'safe_duplicate_groups': len(stats['safe_duplicate_details']),
        'conflict_groups': len(stats['conflict_details'])
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"💾 Detailed report saved to: {output_file}")


def main():
    parser = ArgumentParser(description="Analyze XFG duplicates and conflicts")
    parser.add_argument("-c", "--config", 
                       help="Path to YAML configuration file",
                       default="configs/dwk.yaml", type=str)
    parser.add_argument("-o", "--output",
                       help="Output file for detailed report",
                       default="xfg_analysis_report.json", type=str)
    parser.add_argument("-v", "--verbose",
                       help="Verbose output", action="store_true")
    parser.add_argument("--balance-output",
                        help="Output file for balanced dataset paths (JSON format)",
                        type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = OmegaConf.load(args.config)
        data_folder = config.data_folder
        dataset_name = config.dataset.name
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    try:
        # Get all XFG paths
        print(f"🔍 Scanning XFGs in {data_folder}/{dataset_name}/XFG/...")
        xfg_paths = get_all_xfg_paths(data_folder, dataset_name)
        print(f"📁 Found {len(xfg_paths)} XFG files")
        
        if len(xfg_paths) == 0:
            print("❌ No XFG files found!")
            return
        
        # Analyze duplicates
        stats = analyze_xfg_duplicates(xfg_paths, verbose=args.verbose)
        
        # Print summary
        print_summary_report(stats)
        
        # Save detailed report
        # save_detailed_report(stats, args.output)
        
        # Create balanced dataset if requested
        # if args.balance_output:
        #     create_balanced_dataset(stats, args.balance_output)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


if __name__ == "__main__":
    main()