import torch
from functools import lru_cache
from backend.app.config import settings
from backend.app.models.gnn_layer import HeteroEquipmentGNN, FaultPredictionHead
from backend.app.models.cross_attention import MultimodalFusionModel
from backend.app.models.scorer import AnomalyScorerHead
from backend.app.eval.calibration import ThresholdTuner
from backend.app.pipeline.inference import InferencePipeline
from backend.app.data.graph_schema import get_edge_types

class ModelContainer:
    def __init__(self):
        # We initialize with random weights for now
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn = HeteroEquipmentGNN(
            hidden_channels=settings.gnn_hidden_channels, 
            out_channels=settings.gnn_out_channels, 
            edge_types=get_edge_types()
        )
        self.head = FaultPredictionHead(in_channels=settings.gnn_out_channels)
        self.fusion = MultimodalFusionModel(
            sensor_input_dim=settings.sensor_input_dim, 
            embed_dim=settings.fused_embed_dim
        )
        self.scorer = AnomalyScorerHead(
            input_dim=settings.fused_embed_dim, 
            hidden_dim=settings.gnn_out_channels
        )
        
        self.tuner = ThresholdTuner()
        # Mock fitted threshold for pipeline
        self.tuner.optimal_threshold = 0.5
        self.tuner.is_fitted = True
        
        redis_url = f"redis://{settings.redis_host}:{settings.redis_port}/0"
        self.pipeline = InferencePipeline(
            fusion_model=self.fusion,
            gnn_model=self.gnn,
            prediction_head=self.head,
            scorer_head=self.scorer,
            threshold_tuner=self.tuner,
            device=self.device,
            redis_url=redis_url  # Will fallback to local_cache if not available
        )
        
        self.gnn.eval()
        self.head.eval()

@lru_cache()
def get_model_container() -> ModelContainer:
    """
    Returns a singleton instance of the models to avoid reloading on each request.
    """
    return ModelContainer()
