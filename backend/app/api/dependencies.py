import torch
from functools import lru_cache
from backend.app.models.gnn_layer import HeteroEquipmentGNN, FaultPredictionHead
from backend.app.data.graph_schema import get_edge_types

class ModelContainer:
    def __init__(self):
        # We initialize with random weights for now
        self.gnn = HeteroEquipmentGNN(hidden_channels=128, out_channels=64, edge_types=get_edge_types())
        self.head = FaultPredictionHead(in_channels=64)
        
        self.gnn.eval()
        self.head.eval()

@lru_cache()
def get_model_container() -> ModelContainer:
    """
    Returns a singleton instance of the models to avoid reloading on each request.
    """
    return ModelContainer()
