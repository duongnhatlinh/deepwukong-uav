#!/usr/bin/env python3
"""
Enhanced Vulnerability Detection Pipeline
Automatically processes source code files through the complete pipeline:
Source Code → Joern → PDG → XFG → Vulnerability Detection
"""

import os
import sys
import tempfile
import shutil
import subprocess
import json
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Dict, Tuple, Optional
import torch
import networkx as nx
from torch_geometric.data import Batch
from os import system


# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import build_PDG, build_XFG
from src.models.vd import DeepWuKong
from src.datas.graphs import XFG
from src.preprocess.symbolizer import clean_gadget, tokenize_code_line
from src.utils import filter_warnings, write_gpickle, read_gpickle
from omegaconf import OmegaConf


class VulnerabilityDetector:
    """Enhanced vulnerability detector with automated pipeline"""
    
    def __init__(self, checkpoint_path: str):
        """
        Initialize the detector
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Optional config path (will use model's config if None)
        """
        filter_warnings()
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = DeepWuKong.load_from_checkpoint(checkpoint_path).to(self.device)
        self.model.eval()
        
        # Load config and vocab from model
        self.config = self.model.hparams["config"]
        self.vocab = self.model.hparams["vocab"]
        
        
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {self.vocab.get_vocab_size()}")

    def prepare_input_files(self, source_path: str, temp_dir: str) -> Tuple[bool, str]:
        extensions = {'.c', '.cpp', '.h'}

        # Create temp directories
        temp_code_dir = os.path.join(temp_dir, "data_test", "code")
        os.makedirs(temp_code_dir, exist_ok=True)
        try:
            if os.path.isfile(source_path):
                file_ext = os.path.splitext(source_path)[1].lower()
                if file_ext not in extensions:
                    print(f"Error: File {source_path} has unsupported extension {file_ext}")
                    return False, ""
                
                # Copy single file
                file_name = os.path.basename(source_path)
                dest_path = os.path.join(temp_code_dir, file_name)
                shutil.copy2(source_path, dest_path)
                print(f"Copied file: {source_path} → {dest_path}")
                
            elif os.path.isdir(source_path):
                valid_files = []
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in extensions:
                            valid_files.append(os.path.join(root, file))

                if not valid_files:
                    print(f"Error: No files with supported extensions found in {source_path}")
                    return False, ""
                
                # Copy valid files maintaining directory structure
                for file_path in valid_files:
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.join(temp_code_dir, file_name)
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied file: {file_path} → {dest_path}")
                

            else:
                print(f"Error: {source_path} is neither a file nor directory")
                return False, ""
            
            return True, temp_code_dir

        except Exception as e:
            print(f"Error preparing input files: {e}")
            return False, ""
        
    def run_joern_parse(self, source_dir: str, temp_dir: str) -> Tuple[bool, str]:
        try:
            #source_dir =  /tmp/deepwukong_x9murjjp/data_test/code

            # Create CSV output directory
            csv_output_dir = os.path.join(temp_dir, "data_test", "csv")
            os.makedirs(csv_output_dir, exist_ok=True)

            joern_cmd = f"{self.config.joern_path} {csv_output_dir} {source_dir}"
            print(f"Running Joern: {joern_cmd}")

            system(joern_cmd)

            csv_path = csv_output_dir + source_dir

            return True, csv_path

        except Exception as e:
            print(f"Error running Joern: {e}")
            return False, ""
        
            
    
    def add_syms(self, xfg: nx.DiGraph) -> nx.DiGraph:
        """
        Add symbolic tokens to XFG nodes
        
        Args:
            xfg: The XFG graph
            
        Returns:
            XFG with symbolic tokens added
        """
        file_path = xfg.graph["file_paths"][0]
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents = f.readlines()
        except Exception as e:
            print(f"Error reading source file {file_path}: {e}")
            return xfg
            
        code_lines = []
        for n in xfg:
            if n <= len(file_contents):
                code_lines.append(file_contents[n - 1])
            else:
                code_lines.append("")  # Handle case where line number exceeds file length
                
        sym_code_lines = clean_gadget(code_lines)
        for idx, n in enumerate(xfg):
            xfg.nodes[n]["code_sym_token"] = tokenize_code_line(
                sym_code_lines[idx], 
                self.config.get('split_token', False)
            )
            
        return xfg
    
    def detect_files(self, source_path: str) -> List[Dict]:
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
    
        # Create base temporary directory
        temp_dir = tempfile.mkdtemp(prefix="deepwukong_")

        try:
            print(f"Processing file: {source_path}")

            # Step 1: Prepare input files
            success, prepared_source_dir = self.prepare_input_files(source_path, temp_dir)
            if not success:
                return []
                        
            # Step 2: Run Joern to generate CSV
            success, csv_path = self.run_joern_parse(prepared_source_dir, temp_dir)
            if not success:
                print("Failed to parse source code with Joern")
                return []
            
            # Step 3: Check for sensitive API file
            sensi_api_path = os.path.join(self.config.data_folder, "sensiAPI.txt")
            if not os.path.exists(sensi_api_path):
                raise FileNotFoundError(f"SensiAPI file not found: {sensi_api_path}")
            
            # Step 4: Process each source file for PDG building
            all_results = []

            # Find all source files in prepared directory
            source_files = []
            for root, dirs, files in os.walk(prepared_source_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in ['.c', '.cpp', '.h']):
                        file_path = os.path.join(root, file)

                        rel_path = os.path.relpath(file_path, prepared_source_dir)
                        if os.path.isfile(source_path):
                            original_path = source_path
                        else:
                            original_path = os.path.join(source_path, rel_path)
                        source_files.append((file_path, original_path))

            print(f"Found {len(source_files)} source files to analyze")

            for prepared_file, original_file in source_files:
                try:
                    print(f"Analyzing: {original_file}")

                    file_csv_path = os.path.join(csv_path, os.path.basename(prepared_file))
                    # Build PDG and extract XFGs
                    PDG, key_line_map = build_PDG(file_csv_path, sensi_api_path, prepared_file)
                    
                    if PDG is None or key_line_map is None:
                        print(f"  → Failed to build PDG for {original_file}")
                        continue

                    xfg_dict = build_XFG(PDG, key_line_map)

                    if xfg_dict is None:
                        print(f"  → Failed to build XFGs for {original_file}")
                        continue

                    # Prepare data for model
                    graph_data = []
                    meta_data = []

                    for api_type in xfg_dict:
                        for xfg in xfg_dict[api_type]:
                            try:
                                # Update file path to original
                                xfg.graph["file_paths"] = [prepared_file]
                                # Add symbolic tokens
                                xfg_sym = self.add_syms(xfg)

                                # Convert to torch format
                                torch_data = XFG(xfg=xfg_sym).to_torch(
                                    self.vocab, 
                                    self.config.dataset.token.max_parts
                                )

                                graph_data.append(torch_data)
                                meta_data.append({
                                    'file_path': original_file,  # Use original path for reporting
                                    'key_line': xfg_sym.graph["key_line"],
                                    'api_type': api_type
                                })
                            
                            except Exception as e:
                                print(f"  → Error processing XFG for line {xfg.graph.get('key_line', 'unknown')}: {e}")
                                continue
                    
                    if not graph_data:
                        print(f"  → No valid XFGs found for {original_file}")
                        continue

                    # Run inference
                    print(f"  → Running inference on {len(graph_data)} XFGs...")
                    batch = Batch.from_data_list(graph_data).to(self.device)

                    with torch.no_grad():
                        logits = self.model(batch)
                        probabilities = torch.softmax(logits, dim=1)
                        _, predictions = logits.max(dim=1)
                        confidence_scores = probabilities.max(dim=1)[0]

                    # Compile results
                    file_results = []
                    for i, meta in enumerate(meta_data):
                        is_vulnerable = predictions[i].item() == 1
                        confidence = confidence_scores[i].item()
                        vuln_prob = probabilities[i][1].item()
                        
                        result = {
                            'file_path': meta['file_path'],
                            'line_number': meta['key_line'],
                            'api_type': meta['api_type'],
                            'is_vulnerable': is_vulnerable,
                            'confidence': confidence,
                            'vulnerability_probability': vuln_prob,
                            'prediction_class': predictions[i].item()
                        }
                        file_results.append(result)
                    
                    all_results.extend(file_results)

                    # Print file summary
                    vulnerable_count = sum(1 for r in file_results if r['is_vulnerable'])
                    print(f"  → Found {vulnerable_count} potential vulnerabilities")


                except Exception as e:
                    print(f"  → Error processing {original_file}: {e}")
                    continue

            return all_results              
            

        
            
        except Exception as e:
            print(f"Error processing file {source_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not cleanup temp directory {temp_dir}: {e}")
            
            
    
    
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save detection results to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of detection results"""
        if not results:
            print("No results to display")
            return
        
        vulnerable_results = [r for r in results if r['is_vulnerable']]
        
        print(f"\n{'='*60}")
        print("VULNERABILITY DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total XFGs analyzed: {len(results)}")
        print(f"Potential vulnerabilities found: {len(vulnerable_results)}")
        
        if vulnerable_results:
            print(f"\nVULNERABLE LOCATIONS:")
            print(f"{'File':<40} {'Line':<8} {'Type':<10} {'Confidence':<12}")
            print("-" * 70)
            
            # Sort by confidence (highest first)
            vulnerable_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            for result in vulnerable_results:
                file_name = os.path.basename(result['file_path'])
                line_num = result['line_number']
                api_type = result['api_type']
                confidence = result['confidence']
                
                print(f"{file_name:<40} {line_num:<8} {api_type:<10} {confidence:<12.3f}")
        
        print(f"{'='*60}")


def configure_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Enhanced DeepWuKong Vulnerability Detector")
    
    parser.add_argument("-c", "--checkpoint", 
                       required=True,
                       help="Path to trained model checkpoint")
    
    parser.add_argument("-s", "--source", 
                       required=True,
                       help="Path to source code file or directory")
    
    parser.add_argument("-o", "--output",
                       help="Output file to save results (JSON format)")
    
   
    return parser



    
def main():
    parser = configure_arg_parser()
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = VulnerabilityDetector(args.checkpoint)

        results = detector.detect_files(args.source)        
        
        # # Save results if requested
        # if args.output and os.path.isfile(args.source):
        #     detector.save_results(results, args.output)
        
        # # Print summary
        # if not args.quiet:
        #     detector.print_summary(results)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


# python src/enhanced_detect.py \
#     -c /home/linh/Documents/code/DeepWukong/ts_logger/DeepWuKong/graph_balanced/version_0/checkpoints/epoch=79-step=90960-val_loss=0.0216.ckpt \
#     -s /home/linh/Documents/code/DeepWukong/data_test/vulnerable_AP_BattMonitor_SMBus_Solo.cpp



'''
# Các bước thực hiện:
- Nhận đầu vào là một file, hoặc thư mục chứa nhiều file của người dùng (kiểm tra tránh lỗi, chỉ nhận file extension: '.c', '.cpp', '.h')
- Tạo một thư mục temp "data_test/code" để copy toàn bộ file từ người dùng 
- Tạo một thư mục temp "data_test/csv" để lưu đầu ra file csv nodes/edges khi dùng joern/joern-parse
- Thực hiện các bước tiếp theo:
    + Build PDG and extract XFGs
    + Prepare data for model
    + Run inference
    + Compile results

'''