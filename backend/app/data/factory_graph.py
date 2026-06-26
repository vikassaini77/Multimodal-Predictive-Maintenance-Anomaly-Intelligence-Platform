import networkx as nx
import torch
from torch_geometric.data import HeteroData
from .graph_schema import (
    NODE_MACHINE, NODE_CONVEYOR, NODE_SENSOR,
    EDGE_FEEDS_INTO, EDGE_FEEDS_INTO_REV, EDGE_MONITORS
)

def build_synthetic_factory_graph(num_lines: int = 3, machines_per_line: int = 4) -> nx.DiGraph:
    """
    Generates a realistic assembly-line graph topology using NetworkX.
    Nodes include machines, conveyors connecting them, and sensors monitoring machines.
    """
    G = nx.DiGraph()
    
    machine_idx = 0
    conveyor_idx = 0
    sensor_idx = 0
    
    for line in range(num_lines):
        prev_machine = None
        for m in range(machines_per_line):
            curr_machine = f"{NODE_MACHINE}_{machine_idx}"
            G.add_node(curr_machine, type=NODE_MACHINE, id=machine_idx)
            
            # Add a sensor for each machine
            curr_sensor = f"{NODE_SENSOR}_{sensor_idx}"
            G.add_node(curr_sensor, type=NODE_SENSOR, id=sensor_idx)
            G.add_edge(curr_sensor, curr_machine, type=EDGE_MONITORS)
            sensor_idx += 1
            
            if prev_machine is not None:
                curr_conveyor = f"{NODE_CONVEYOR}_{conveyor_idx}"
                G.add_node(curr_conveyor, type=NODE_CONVEYOR, id=conveyor_idx)
                
                G.add_edge(prev_machine, curr_conveyor, type=EDGE_FEEDS_INTO)
                G.add_edge(curr_conveyor, curr_machine, type=EDGE_FEEDS_INTO_REV)
                conveyor_idx += 1
                
            prev_machine = curr_machine
            machine_idx += 1
            
    return G

def convert_to_pyg_heterodata(nx_graph: nx.DiGraph, machine_embeddings: torch.Tensor = None, conveyor_embeddings: torch.Tensor = None, sensor_embeddings: torch.Tensor = None) -> HeteroData:
    """
    Maps NetworkX graph to PyTorch Geometric HeteroData format, attaching embeddings.
    """
    data = HeteroData()
    
    # Extract nodes by type
    nodes_by_type = {NODE_MACHINE: [], NODE_CONVEYOR: [], NODE_SENSOR: []}
    node_to_idx = {NODE_MACHINE: {}, NODE_CONVEYOR: {}, NODE_SENSOR: {}}
    
    for node, attrs in nx_graph.nodes(data=True):
        ntype = attrs['type']
        nidx = len(nodes_by_type[ntype])
        nodes_by_type[ntype].append(node)
        node_to_idx[ntype][node] = nidx
        
    # Set node features
    num_machines = len(nodes_by_type[NODE_MACHINE])
    if machine_embeddings is not None:
        assert machine_embeddings.shape[0] == num_machines, f"Expected {num_machines} embeddings, got {machine_embeddings.shape[0]}"
        data[NODE_MACHINE].x = machine_embeddings
    else:
        # Default fallback if no embeddings provided (e.g. testing)
        data[NODE_MACHINE].x = torch.zeros((num_machines, 256), dtype=torch.float)
        
    num_conveyors = len(nodes_by_type[NODE_CONVEYOR])
    if conveyor_embeddings is not None:
        assert conveyor_embeddings.shape[0] == num_conveyors
        data[NODE_CONVEYOR].x = conveyor_embeddings
    else:
        data[NODE_CONVEYOR].x = torch.zeros((num_conveyors, 256), dtype=torch.float)
        
    num_sensors = len(nodes_by_type[NODE_SENSOR])
    if sensor_embeddings is not None:
        assert sensor_embeddings.shape[0] == num_sensors
        data[NODE_SENSOR].x = sensor_embeddings
    else:
        data[NODE_SENSOR].x = torch.zeros((num_sensors, 256), dtype=torch.float)
        
    # Extract edges
    edges_by_type = {
        EDGE_FEEDS_INTO: ([], []),
        EDGE_FEEDS_INTO_REV: ([], []),
        EDGE_MONITORS: ([], [])
    }
    
    for u, v, attrs in nx_graph.edges(data=True):
        u_type = nx_graph.nodes[u]['type']
        v_type = nx_graph.nodes[v]['type']
        edge_type = attrs['type']
        
        u_idx = node_to_idx[u_type][u]
        v_idx = node_to_idx[v_type][v]
        
        edges_by_type[edge_type][0].append(u_idx)
        edges_by_type[edge_type][1].append(v_idx)
        
    for edge_type, (u_list, v_list) in edges_by_type.items():
        if len(u_list) > 0:
            edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
            data[edge_type].edge_index = edge_index
        else:
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            
    return data
