import torch
from backend.app.models.gnn_layer import EquipmentTopologyGNN

def test_gnn_initialization():
    model = EquipmentTopologyGNN(256, 128, 64)
    assert model is not None
