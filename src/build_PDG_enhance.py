"""
Enhanced PDG Builder for Comprehensive CWE Detection
Supports detection of: CWE-119, CWE-120, CWE-125, CWE-190, CWE-362, CWE-399, CWE-415, CWE-416, CWE-476, CWE-787
"""

import networkx as nx
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import re
import os
from os.path import join, exists

def read_csv(csv_file_path: str) -> List:
    """
    read csv file
    """
    assert exists(csv_file_path), f"no {csv_file_path}"
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data

# Enhanced Sensitive API Categories for different CWE types
SENSITIVE_APIS = {
    # CWE-119, CWE-120: Buffer Operations
    'buffer_unsafe': [
        'strcpy', 'strcat', 'sprintf', 'vsprintf', 'gets', 'scanf', 'sscanf',
        'strncpy', 'strncat', 'snprintf', 'vsnprintf'  # These can be safer but still risky
    ],
    
    # CWE-399, CWE-415, CWE-416: Memory Management
    'memory_mgmt': [
        'malloc', 'calloc', 'realloc', 'free', 'new', 'delete', 'delete[]',
        'GlobalAlloc', 'HeapAlloc', 'VirtualAlloc', 'LocalAlloc'
    ],
    
    # CWE-399: Resource Management
    'resource_mgmt': [
        'fopen', 'fclose', 'open', 'close', 'socket', 'closesocket',
        'CreateFile', 'CloseHandle', 'CreateThread', 'CreateProcess'
    ],
    
    # CWE-362: Thread/Synchronization
    'thread_sync': [
        'pthread_create', 'pthread_join', 'pthread_mutex_lock', 'pthread_mutex_unlock',
        'pthread_rwlock_rdlock', 'pthread_rwlock_wrlock', 'pthread_rwlock_unlock',
        'CreateThread', 'WaitForSingleObject', 'EnterCriticalSection', 'LeaveCriticalSection'
    ],
    
    # String operations that can cause buffer issues
    'string_ops': [
        'strlen', 'strcmp', 'strncmp', 'strchr', 'strstr', 'strtok',
        'memcpy', 'memset', 'memmove', 'memcmp'
    ]
}

def extract_line_number(idx: int, nodes: List) -> int:
    """Extract line number from node index"""
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except Exception as e:
                    print(e)
                    pass
        idx -= 1
    return -1

def get_flat_sensitive_apis(sensi_api_path: str) -> Set[str]:
    """Get all sensitive APIs from file and built-in categories"""
    try:
        with open(sensi_api_path, "r", encoding="utf-8") as f:
            file_apis = set([api.strip() for api in f.read().split(",")])
    except:
        file_apis = set()
    
    # Combine with built-in categories
    all_apis = file_apis.copy()
    for category_apis in SENSITIVE_APIS.values():
        all_apis.update(category_apis)
    
    return all_apis

def find_function_calls(nodes: List) -> Dict[str, List[Tuple[int, str, int]]]:
    """
    Find all function calls and categorize them
    Returns: {api_category: [(node_idx, function_name, line_number)]}
    """
    function_calls = defaultdict(list)
    
    for node_idx, node in enumerate(nodes):
        ntype = node.get('type', '').strip()
        
        if ntype == 'CallExpression':
            # Try to get function name from next node or current node
            function_name = None
            
            # Method 1: Check next node
            if node_idx + 1 < len(nodes):
                function_name = nodes[node_idx + 1].get('code', '').strip()
            
            # Method 2: Check current node's code
            if not function_name:
                function_name = node.get('code', '').strip()
            
            # Method 3: Look for Identifier nodes nearby
            if not function_name:
                for offset in range(1, min(5, len(nodes) - node_idx)):
                    nearby_node = nodes[node_idx + offset]
                    if nearby_node.get('type') == 'Identifier':
                        function_name = nearby_node.get('code', '').strip()
                        break
            
            if function_name:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    # Categorize the function call
                    for category, apis in SENSITIVE_APIS.items():
                        if function_name in apis:
                            function_calls[category].append((node_idx, function_name, line_no))
                    
                    # Also add to general calls
                    function_calls['all_calls'].append((node_idx, function_name, line_no))
    
    return function_calls

def find_variable_operations(nodes: List) -> Dict[str, List[Tuple[int, str, int]]]:
    """
    Find variable declarations, assignments, and uses
    Important for tracking pointer lifecycle and data flow
    """
    var_operations = defaultdict(list)
    
    for node_idx, node in enumerate(nodes):
        ntype = node.get('type', '').strip()
        code = node.get('code', '').strip()
        line_no = extract_line_number(node_idx, nodes)
        
        if line_no <= 0:
            continue
            
        # CWE-476, CWE-416: Variable declarations (especially pointers)
        if ntype in ['VariableDeclarator', 'Parameter', 'FieldDeclaration']:
            var_operations['declarations'].append((node_idx, code, line_no))
            
            # Check if it's a pointer declaration
            if '*' in code or 'ptr' in code.lower():
                var_operations['pointer_declarations'].append((node_idx, code, line_no))
        
        # CWE-416, CWE-415: Assignment operations
        elif ntype == 'AssignmentExpression' or '=' in code:
            var_operations['assignments'].append((node_idx, code, line_no))
            
            # Check for NULL assignments (important for tracking pointer state)
            if 'NULL' in code or 'null' in code or '= 0' in code:
                var_operations['null_assignments'].append((node_idx, code, line_no))
        
        # CWE-476: Conditional expressions (null checks)
        elif ntype in ['ConditionalExpression', 'IfStatement', 'Condition']:
            if any(op in code for op in ['==', '!=', '!']) and ('NULL' in code or 'null' in code):
                var_operations['null_checks'].append((node_idx, code, line_no))
    
    return var_operations

def find_memory_operations(nodes: List) -> Dict[str, List[Tuple[int, str, int]]]:
    """
    Track memory allocation, deallocation, and access patterns
    Critical for CWE-415, CWE-416, CWE-399
    """
    memory_ops = defaultdict(list)
    
    allocation_funcs = ['malloc', 'calloc', 'realloc', 'new']
    deallocation_funcs = ['free', 'delete']
    
    for node_idx, node in enumerate(nodes):
        ntype = node.get('type', '').strip()
        code = node.get('code', '').strip()
        line_no = extract_line_number(node_idx, nodes)
        
        if line_no <= 0:
            continue
        
        # CWE-399: Memory allocations
        if ntype == 'CallExpression':
            # Check next node for function name
            func_name = ''
            if node_idx + 1 < len(nodes):
                func_name = nodes[node_idx + 1].get('code', '').strip()
            
            if func_name in allocation_funcs:
                memory_ops['allocations'].append((node_idx, func_name, line_no))
            elif func_name in deallocation_funcs:
                memory_ops['deallocations'].append((node_idx, func_name, line_no))
        
        # CWE-416: Pointer dereferences
        elif ntype == 'PointerExpression' or ('*' in code and ntype not in ['Comment', 'Literal']):
            memory_ops['pointer_dereferences'].append((node_idx, code, line_no))
        
        # CWE-119, CWE-125, CWE-787: Array accesses  
        elif ntype == 'ArrayIndexing':
            memory_ops['array_accesses'].append((node_idx, code, line_no))
        
        # CWE-476: Pointer member access
        elif ntype == 'PtrMemberAccess':
            memory_ops['pointer_member_access'].append((node_idx, code, line_no))
    
    return memory_ops

def find_arithmetic_operations(nodes: List) -> Dict[str, List[Tuple[int, str, int]]]:
    """
    Find arithmetic operations that could lead to integer overflow
    Important for CWE-190
    """
    arith_ops = defaultdict(list)
    
    # Operators that can cause overflow
    overflow_ops = ['+', '-', '*', '/', '<<', '>>', '++', '--']
    comparison_ops = ['<=', '>=', '==', '<', '>', '!=']
    
    for node_idx, node in enumerate(nodes):
        operator = node.get('operator', '').strip()
        ntype = node.get('type', '').strip()
        code = node.get('code', '').strip()
        line_no = extract_line_number(node_idx, nodes)
        
        if line_no <= 0:
            continue
        
        # CWE-190: Arithmetic operations that can overflow
        if operator in overflow_ops:
            arith_ops['overflow_prone'].append((node_idx, operator, line_no))
            
            # Special attention to multiplication and left shift
            if operator in ['*', '<<']:
                arith_ops['high_risk_overflow'].append((node_idx, operator, line_no))
        
        # CWE-125, CWE-787: Comparisons (bounds checking)
        elif operator in comparison_ops:
            arith_ops['comparisons'].append((node_idx, operator, line_no))
        
        # CWE-190: Increment/decrement operations
        elif ntype in ['PostIncrement', 'PreIncrement', 'PostDecrement', 'PreDecrement']:
            arith_ops['increment_ops'].append((node_idx, ntype, line_no))
    
    return arith_ops

def find_thread_operations(nodes: List) -> Dict[str, List[Tuple[int, str, int]]]:
    """
    Find thread-related operations for race condition detection
    Important for CWE-362
    """
    thread_ops = defaultdict(list)
    
    lock_funcs = ['pthread_mutex_lock', 'EnterCriticalSection', 'lock']
    unlock_funcs = ['pthread_mutex_unlock', 'LeaveCriticalSection', 'unlock']
    thread_funcs = ['pthread_create', 'CreateThread']
    
    for node_idx, node in enumerate(nodes):
        ntype = node.get('type', '').strip()
        line_no = extract_line_number(node_idx, nodes)
        
        if line_no <= 0:
            continue
        
        if ntype == 'CallExpression':
            # Get function name
            func_name = ''
            if node_idx + 1 < len(nodes):
                func_name = nodes[node_idx + 1].get('code', '').strip()
            
            # CWE-362: Lock operations
            if any(lock in func_name for lock in lock_funcs):
                thread_ops['lock_operations'].append((node_idx, func_name, line_no))
            elif any(unlock in func_name for unlock in unlock_funcs):
                thread_ops['unlock_operations'].append((node_idx, func_name, line_no))
            elif any(thread in func_name for thread in thread_funcs):
                thread_ops['thread_creation'].append((node_idx, func_name, line_no))
    
    return thread_ops

def analyze_control_flow_patterns(nodes: List, edges: List) -> Dict[str, List[Tuple[int, str, int]]]:
    """
    Analyze control flow patterns that might indicate vulnerabilities
    """
    patterns = defaultdict(list)
    
    for node_idx, node in enumerate(nodes):
        ntype = node.get('type', '').strip()
        code = node.get('code', '').strip()
        line_no = extract_line_number(node_idx, nodes)
        
        if line_no <= 0:
            continue
        
        # CWE-399: Exception handling (or lack thereof)
        if ntype in ['TryStatement', 'CatchBlock', 'ThrowStatement']:
            patterns['exception_handling'].append((node_idx, ntype, line_no))
        
        # CWE-362: Critical sections
        if ntype in ['Block', 'CompoundStatement']:
            patterns['code_blocks'].append((node_idx, ntype, line_no))
        
        # Error handling patterns
        if 'error' in code.lower() or 'fail' in code.lower():
            patterns['error_handling'].append((node_idx, code, line_no))
    
    return patterns

def build_enhanced_PDG(code_path: str, sensi_api_path: str, source_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    """
    Enhanced PDG builder with comprehensive CWE detection capabilities
    
    Detects patterns for:
    - CWE-119: Buffer Overflow
    - CWE-120: Buffer Copy without Checking Size  
    - CWE-125: Out-of-bounds Read
    - CWE-190: Integer Overflow
    - CWE-362: Race Condition
    - CWE-399: Resource Management Errors
    - CWE-415: Double Free
    - CWE-416: Use After Free
    - CWE-476: NULL Pointer Dereference  
    - CWE-787: Out-of-bounds Write
    """
    
    nodes_path = os.path.join(code_path, "nodes.csv")
    edges_path = os.path.join(code_path, "edges.csv")
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        return None, None
    
    # Read CSV files
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    
    if len(nodes) == 0:
        return None, None
    
    # Get all sensitive APIs
    # sensi_api_set = get_flat_sensitive_apis(sensi_api_path)
    
    # Initialize line sets for different vulnerability types
    vulnerability_lines = {
        # CWE-119, CWE-120: Buffer operations
        'buffer_unsafe': set(),
        'buffer_safe': set(),
        
        # CWE-125, CWE-787: Array operations  
        'array_access': set(),
        'array_bounds_check': set(),
        
        # CWE-476: Pointer operations
        'pointer_deref': set(),
        'pointer_null_check': set(),
        'pointer_member_access': set(),
        
        # CWE-190: Arithmetic operations
        'arithmetic_overflow': set(),
        'arithmetic_safe': set(),
        
        # CWE-399: Resource management
        'resource_alloc': set(),
        'resource_dealloc': set(),
        'resource_use': set(),
        
        # CWE-415, CWE-416: Memory management
        'memory_alloc': set(),
        'memory_free': set(),
        'memory_use_after_free': set(),
        'memory_double_free': set(),
        
        # CWE-362: Thread operations
        'thread_lock': set(),
        'thread_unlock': set(),
        'thread_shared_access': set(),
        
        # General patterns
        'sensitive_calls': set(),
        'error_handling': set(),
        'null_assignments': set(),
        'variable_declarations': set()
    }
    
    # === FUNCTION CALL ANALYSIS ===
    function_calls = find_function_calls(nodes)
    
    for category, calls in function_calls.items():
        for node_idx, func_name, line_no in calls:
            if category == 'buffer_unsafe':
                # CWE-119, CWE-120: Unsafe buffer operations
                vulnerability_lines['buffer_unsafe'].add(line_no)
                # vulnerability_lines['sensitive_calls'].add(line_no)
            elif category == 'memory_mgmt':
                # CWE-399, CWE-415, CWE-416: Memory management
                if func_name in ['malloc', 'calloc', 'realloc', 'new']:
                    vulnerability_lines['memory_alloc'].add(line_no)
                    vulnerability_lines['resource_alloc'].add(line_no)
                elif func_name in ['free', 'delete']:
                    vulnerability_lines['memory_free'].add(line_no)
                    vulnerability_lines['resource_dealloc'].add(line_no)
            elif category == 'resource_mgmt':
                # CWE-399: Resource management
                if 'open' in func_name or 'Create' in func_name:
                    vulnerability_lines['resource_alloc'].add(line_no)
                elif 'close' in func_name or 'Close' in func_name:
                    vulnerability_lines['resource_dealloc'].add(line_no)
                vulnerability_lines['resource_use'].add(line_no)
            elif category == 'thread_sync':
                # CWE-362: Thread synchronization
                if 'lock' in func_name or 'Enter' in func_name:
                    vulnerability_lines['thread_lock'].add(line_no)
                elif 'unlock' in func_name or 'Leave' in func_name:
                    vulnerability_lines['thread_unlock'].add(line_no)
            
            # Mark all sensitive API calls
            vulnerability_lines['sensitive_calls'].add(line_no)
    
    # === VARIABLE OPERATIONS ANALYSIS ===
    var_operations = find_variable_operations(nodes)
    
    for op_type, operations in var_operations.items():
        for node_idx, code, line_no in operations:
            if op_type == 'pointer_declarations':
                # CWE-476: Pointer declarations
                vulnerability_lines['variable_declarations'].add(line_no)
            elif op_type == 'null_assignments':
                # CWE-476, CWE-416: NULL assignments
                vulnerability_lines['null_assignments'].add(line_no)
            elif op_type == 'null_checks':
                # CWE-476: NULL pointer checks (good practice)
                vulnerability_lines['pointer_null_check'].add(line_no)
    
    # === MEMORY OPERATIONS ANALYSIS ===
    memory_operations = find_memory_operations(nodes)
    
    for op_type, operations in memory_operations.items():
        for node_idx, code, line_no in operations:
            if op_type == 'array_accesses':
                # CWE-119, CWE-125, CWE-787: Array access
                vulnerability_lines['array_access'].add(line_no)
            elif op_type == 'pointer_dereferences':
                # CWE-476: Pointer dereference
                vulnerability_lines['pointer_deref'].add(line_no)
            elif op_type == 'pointer_member_access':
                # CWE-476: Pointer member access
                vulnerability_lines['pointer_member_access'].add(line_no)
    
    # === ARITHMETIC OPERATIONS ANALYSIS ===
    arith_operations = find_arithmetic_operations(nodes)
    
    for op_type, operations in arith_operations.items():
        for node_idx, operator, line_no in operations:
            if op_type in ['overflow_prone', 'high_risk_overflow']:
                # CWE-190: Integer overflow
                vulnerability_lines['arithmetic_overflow'].add(line_no)
            elif op_type == 'comparisons':
                # CWE-125, CWE-787: Bounds checking (good practice)
                vulnerability_lines['array_bounds_check'].add(line_no)
    
    # === THREAD OPERATIONS ANALYSIS ===
    thread_operations = find_thread_operations(nodes)
    
    for op_type, operations in thread_operations.items():
        for node_idx, func_name, line_no in operations:
            if op_type == 'lock_operations':
                vulnerability_lines['thread_lock'].add(line_no)
            elif op_type == 'unlock_operations':
                vulnerability_lines['thread_unlock'].add(line_no)
    
    # === LEGACY SUPPORT: Original categories ===
    # Keep original categories for backward compatibility
    call_lines = vulnerability_lines['sensitive_calls'].copy()
    array_lines = vulnerability_lines['array_access'].copy()
    ptr_lines = vulnerability_lines['pointer_deref'] | vulnerability_lines['pointer_member_access']
    arithmatic_lines = vulnerability_lines['arithmetic_overflow'].copy()
    
    # === BUILD PDG ===
    PDG = nx.DiGraph(file_paths=[source_path])
    
    # Extract node information with location
    node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
    
    # Build edges
    control_edges, data_edges = [], []
    
    for edge in edges:
        edge_type = edge['type'].strip()
        start_node_id = edge['start'].strip()
        end_node_id = edge['end'].strip()
        
        if start_node_id not in node_id_to_ln or end_node_id not in node_id_to_ln:
            continue
        
        start_ln = node_id_to_ln[start_node_id]
        end_ln = node_id_to_ln[end_node_id]
        
        if edge_type == 'CONTROLS':  # Control dependency
            control_edges.append((start_ln, end_ln, {"c/d": "c"}))
        elif edge_type == 'REACHES':  # Data dependency
            data_edges.append((start_ln, end_ln, {"c/d": "d"}))
        # Add more edge types for enhanced analysis
        elif edge_type in ['FLOWS_TO', 'IS_AST_PARENT']:
            data_edges.append((start_ln, end_ln, {"c/d": "d"}))
    
    PDG.add_edges_from(control_edges)
    PDG.add_edges_from(data_edges)
    
    # Return comprehensive vulnerability line mapping
    return PDG, {
        # Original categories (backward compatibility)
        "call": call_lines,
        "array": array_lines,
        "ptr": ptr_lines,
        "arith": arithmatic_lines,
        
        # Enhanced categories for specific CWE detection
        "buffer_unsafe": vulnerability_lines['buffer_unsafe'],           # CWE-119, CWE-120
        "array_access": vulnerability_lines['array_access'],             # CWE-125, CWE-787
        "pointer_deref": vulnerability_lines['pointer_deref'],           # CWE-476
        "pointer_member": vulnerability_lines['pointer_member_access'],  # CWE-476
        "arithmetic_overflow": vulnerability_lines['arithmetic_overflow'], # CWE-190
        "memory_alloc": vulnerability_lines['memory_alloc'],             # CWE-399, CWE-415, CWE-416
        "memory_free": vulnerability_lines['memory_free'],               # CWE-415, CWE-416
        "resource_alloc": vulnerability_lines['resource_alloc'],         # CWE-399
        "resource_dealloc": vulnerability_lines['resource_dealloc'],     # CWE-399
        "thread_lock": vulnerability_lines['thread_lock'],               # CWE-362
        "thread_unlock": vulnerability_lines['thread_unlock'],           # CWE-362
        "null_checks": vulnerability_lines['pointer_null_check'],        # CWE-476 (good practice)
        "bounds_checks": vulnerability_lines['array_bounds_check'],      # CWE-125, CWE-787 (good practice)
        "sensitive_calls": vulnerability_lines['sensitive_calls'],       # All sensitive API calls
        "null_assignments": vulnerability_lines['null_assignments'],     # CWE-476, CWE-416
        "variable_decls": vulnerability_lines['variable_declarations']   # General declarations
    }


# Helper function (existing)
def extract_nodes_with_location_info(nodes):
    """Extract nodes with location information"""
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    
    return node_indices, node_ids, line_numbers, node_id_to_line_number


# Updated main function to use enhanced PDG
def build_PDG(code_path: str, sensi_api_path: str, source_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    """
    Main function that calls the enhanced PDG builder
    
    Returns:
        PDG: NetworkX DiGraph representing the program dependence graph
        key_line_map: Dictionary mapping vulnerability types to line numbers
    """
    return build_enhanced_PDG(code_path, sensi_api_path, source_path)


if __name__ == "__main__":
    PDG, key_line_map = build_PDG(
        code_path="/home/linh/Documents/code/DeepWukong/data_test/csv/home/linh/Documents/code/DeepWukong/data_test/source-code/vulnerable_AP2DataPlot2D.h",
        sensi_api_path="path/to/sensi_api.txt",
        source_path="/home/linh/Documents/code/DeepWukong/data_test/source-code/vulnerable_AP2DataPlot2D.h"
    )

    print("PDG built successfully!")
    print(f"Number of nodes: {PDG.number_of_nodes()}")
    print(f"Number of edges: {PDG.number_of_edges()}")
    # Print key line mappings for vulnerabilities
    print("Key line mappings:")
    for key, lines in key_line_map.items():
        print(f"{key}: {sorted(lines)}")  
