import pytest
import torch
import networkx as nx
from backend.app.data.graph_schema import (
    NODE_MACHINE, NODE_CONVEYOR, NODE_SENSOR,
    EDGE_FEEDS_INTO, EDGE_FEEDS_INTO_REV, EDGE_MONITORS
)
from backend.app.data.factory_graph import (
    build_synthetic_factory_graph,
    convert_to_pyg_heterodata
)

def test_build_synthetic_factory_graph():
    num_lines = 2
    machines_per_line = 3
    G = build_synthetic_factory_graph(num_lines, machines_per_line)
    
    # 3 machines, 3 sensors, 2 conveyors per line -> 8 nodes per line. 16 nodes total.
    assert isinstance(G, nx.DiGraph)
    
    machines = [n for n, d in G.nodes(data=True) if d['type'] == NODE_MACHINE]
    sensors = [n for n, d in G.nodes(data=True) if d['type'] == NODE_SENSOR]
    conveyors = [n for n, d in G.nodes(data=True) if d['type'] == NODE_CONVEYOR]
    
    assert len(machines) == num_lines * machines_per_line
    assert len(sensors) == num_lines * machines_per_line
    assert len(conveyors) == num_lines * (machines_per_line - 1)
    
    # Check no orphan nodes
    for node in G.nodes():
        assert G.degree(node) > 0

def test_convert_to_pyg_heterodata():
    G = build_synthetic_factory_graph(1, 2)
    # 2 machines -> we need 2 embeddings
    emb = torch.randn((2, 256))
    
    data = convert_to_pyg_heterodata(G, machine_embeddings=emb)
    
    assert data[NODE_MACHINE].x.shape == (2, 256)
    assert torch.allclose(data[NODE_MACHINE].x, emb)
    
    # 1 conveyor, 2 sensors
    assert data[NODE_CONVEYOR].x.shape == (1, 256)
    assert data[NODE_SENSOR].x.shape == (2, 256)
    
    assert EDGE_FEEDS_INTO in data.edge_types
    assert EDGE_FEEDS_INTO_REV in data.edge_types
    assert EDGE_MONITORS in data.edge_types
