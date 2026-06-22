import torch
import random
from torch_geometric.data import HeteroData
from collections import deque
from backend.app.data.graph_schema import NODE_MACHINE, NODE_CONVEYOR, NODE_SENSOR

def inject_synthetic_faults(hetero_data: HeteroData, root_fault_node: int, root_node_type: str, max_hops: int = 3, p_fault_base: float = 0.9) -> HeteroData:
    """
    Simulates a synthetic fault starting at root_fault_node.
    Propagates the fault to downstream nodes using BFS.
    Probability of fault decreases exponentially by hop distance.
    Assigns binary `.y` tensor to each node type.
    """
    # Initialize labels with 0
    for node_type in [NODE_MACHINE, NODE_CONVEYOR, NODE_SENSOR]:
        if node_type in hetero_data.node_types:
            num_nodes = hetero_data[node_type].num_nodes
            hetero_data[node_type].y = torch.zeros(num_nodes, dtype=torch.float)

    # Build adjacency list from edges
    adj = {}
    for edge_type in hetero_data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = hetero_data[edge_type].edge_index
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            
            src_node = (src_type, u)
            dst_node = (dst_type, v)
            if src_node not in adj:
                adj[src_node] = []
            adj[src_node].append(dst_node)

    # BFS from root
    visited = {}
    queue = deque()
    
    start_node = (root_node_type, root_fault_node)
    queue.append((start_node, 0))
    visited[start_node] = 0
    
    # Always set root to fault
    hetero_data[root_node_type].y[root_fault_node] = 1.0

    while queue:
        current_node, hop = queue.popleft()
        node_type, node_idx = current_node
        
        # Assign label probabilistically based on hop distance
        if hop > 0:
            prob = p_fault_base * (0.5 ** hop)
            is_fault = 1.0 if random.random() < prob else 0.0
            if is_fault == 1.0:
                hetero_data[node_type].y[node_idx] = 1.0
            
        if hop < max_hops:
            if current_node in adj:
                for neighbor in adj[current_node]:
                    if neighbor not in visited or visited[neighbor] > hop + 1:
                        visited[neighbor] = hop + 1
                        queue.append((neighbor, hop + 1))
                        
    return hetero_data
