import torch
from backend.app.models.gnn_layer import EquipmentTopologyGNN

def test_gnn_initialization():
    model = EquipmentTopologyGNN(256, 128, 64)
    assert model is not None

def test_equipment_topology_gnn():
    """Verify the GraphSAGE forward pass accurately processes graph nodes and edges."""
    in_channels, hidden, out_channels = 256, 128, 64
    num_nodes = 5
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    model = EquipmentTopologyGNN(in_channels, hidden, out_channels)
    with torch.no_grad():
        out = model(x, edge_index)
    assert out.shape == (num_nodes, out_channels), "GNN output dimension mismatch"
