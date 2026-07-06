import os
import torch
import torch.nn as nn
import onnx
import onnxoptimizer
from backend.app.models.two_tower import SensorTower, VisualTower, TwoTowerAnomalyModel

# Ensure deployment dir exists
os.makedirs("backend/app/deployment/onnx_models", exist_ok=True)

class AnomalyScorerONNX(nn.Module):
    """Wraps the anomaly scoring logic into a single exportable graph."""
    def __init__(self, model: TwoTowerAnomalyModel):
        super().__init__()
        self.model = model
        
    def forward(self, sensor_data: torch.Tensor, visual_data: torch.Tensor):
        return self.model.predict_anomaly(sensor_data, visual_data)

def export_sensor_tower():
    print("Exporting SensorTower...")
    model = SensorTower(input_dim=2, embed_dim=256)
    model.eval()
    
    # Dummy input: (batch_size, input_dim, seq_len)
    dummy_input = torch.randn(1, 2, 50)
    
    onnx_path = "backend/app/deployment/onnx_models/sensor_tower.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["sensor_input"],
        output_names=["sensor_embedding"],
        dynamic_axes={
            "sensor_input": {0: "batch_size", 2: "seq_len"},
            "sensor_embedding": {0: "batch_size"}
        }
    )
    print(f"Saved to {onnx_path}")
    optimize_onnx(onnx_path)

def export_visual_tower():
    print("Exporting VisualTower...")
    model = VisualTower(embed_dim=256)
    model.eval()
    
    # Dummy input: (batch_size, channels, H, W)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    onnx_path = "backend/app/deployment/onnx_models/visual_tower.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["visual_input"],
        output_names=["visual_embedding"],
        dynamic_axes={
            "visual_input": {0: "batch_size"},
            "visual_embedding": {0: "batch_size"}
        }
    )
    print(f"Saved to {onnx_path}")
    optimize_onnx(onnx_path)

def export_fusion_head():
    print("Exporting Fusion Anomaly Scorer...")
    base_model = TwoTowerAnomalyModel(sensor_input_dim=2, embed_dim=256)
    model = AnomalyScorerONNX(base_model)
    model.eval()
    
    dummy_sensor = torch.randn(1, 2, 50)
    dummy_visual = torch.randn(1, 3, 224, 224)
    
    onnx_path = "backend/app/deployment/onnx_models/anomaly_scorer.onnx"
    torch.onnx.export(
        model, 
        (dummy_sensor, dummy_visual), 
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["sensor_input", "visual_input"],
        output_names=["anomaly_score"],
        dynamic_axes={
            "sensor_input": {0: "batch_size", 2: "seq_len"},
            "visual_input": {0: "batch_size"},
            "anomaly_score": {0: "batch_size"}
        }
    )
    print(f"Saved to {onnx_path}")
    optimize_onnx(onnx_path)

def optimize_onnx(onnx_path: str):
    """Applies ONNX graph optimizations (constant folding, node fusion)."""
    print(f"Optimizing {onnx_path}...")
    model = onnx.load(onnx_path)
    
    # List of passes to run
    passes = [
        "extract_constant_to_initializer",
        "eliminate_unused_initializer",
        "eliminate_deadend",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv"
    ]
    
    optimized_model = onnxoptimizer.optimize(model, passes)
    
    opt_path = onnx_path.replace(".onnx", "_opt.onnx")
    onnx.save(optimized_model, opt_path)
    print(f"Optimized model saved to {opt_path}")

if __name__ == "__main__":
    export_sensor_tower()
    export_visual_tower()
    export_fusion_head()
    print("All ONNX exports and optimizations complete!")
