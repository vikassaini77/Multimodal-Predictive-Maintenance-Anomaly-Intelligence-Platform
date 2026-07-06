import os
import torch
import numpy as np
import onnxruntime as ort
import pytest
import onnx
import onnxoptimizer
from backend.app.models.two_tower import SensorTower, VisualTower, TwoTowerAnomalyModel

ONNX_DIR = "backend/app/deployment/onnx_models"
os.makedirs(ONNX_DIR, exist_ok=True)

def optimize_onnx(onnx_path: str):
    model = onnx.load(onnx_path)
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
    return opt_path

def test_sensor_tower_parity():
    pt_model = SensorTower(input_dim=2, embed_dim=256)
    pt_model.eval()
    
    # Export it right here so weights match exactly
    dummy_input = torch.randn(1, 2, 50)
    onnx_path = os.path.join(ONNX_DIR, "test_sensor_tower.onnx")
    torch.onnx.export(
        pt_model, dummy_input, onnx_path, export_params=True, opset_version=17,
        do_constant_folding=True, input_names=["sensor_input"], output_names=["sensor_embedding"],
        dynamic_axes={"sensor_input": {0: "batch_size", 2: "seq_len"}, "sensor_embedding": {0: "batch_size"}}
    )
    opt_path = optimize_onnx(onnx_path)
    
    ort_session = ort.InferenceSession(opt_path, providers=['CPUExecutionProvider'])
    
    for batch_size in [1, 4]:
        for seq_len in [50]:
            dummy_input = torch.randn(batch_size, 2, seq_len)
            with torch.no_grad():
                pt_out = pt_model(dummy_input).numpy()
            ort_out = ort_session.run(None, {"sensor_input": dummy_input.numpy()})[0]
            np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)

def test_visual_tower_parity():
    pt_model = VisualTower(embed_dim=256)
    pt_model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(ONNX_DIR, "test_visual_tower.onnx")
    torch.onnx.export(
        pt_model, dummy_input, onnx_path, export_params=True, opset_version=17,
        do_constant_folding=True, input_names=["visual_input"], output_names=["visual_embedding"],
        dynamic_axes={"visual_input": {0: "batch_size"}, "visual_embedding": {0: "batch_size"}}
    )
    opt_path = optimize_onnx(onnx_path)
    
    ort_session = ort.InferenceSession(opt_path, providers=['CPUExecutionProvider'])
    
    for batch_size in [1, 2]:
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            pt_out = pt_model(dummy_input).numpy()
        ort_out = ort_session.run(None, {"visual_input": dummy_input.numpy()})[0]
        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)

def test_fusion_scorer_parity():
    from backend.app.deployment.export_onnx import AnomalyScorerONNX
    base_model = TwoTowerAnomalyModel(sensor_input_dim=2, embed_dim=256)
    pt_model = AnomalyScorerONNX(base_model)
    pt_model.eval()
    
    dummy_sensor = torch.randn(1, 2, 50)
    dummy_visual = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(ONNX_DIR, "test_anomaly_scorer.onnx")
    torch.onnx.export(
        pt_model, (dummy_sensor, dummy_visual), onnx_path, export_params=True, opset_version=17,
        do_constant_folding=True, input_names=["sensor_input", "visual_input"], output_names=["anomaly_score"],
        dynamic_axes={"sensor_input": {0: "batch_size", 2: "seq_len"}, "visual_input": {0: "batch_size"}, "anomaly_score": {0: "batch_size"}}
    )
    opt_path = optimize_onnx(onnx_path)
    
    ort_session = ort.InferenceSession(opt_path, providers=['CPUExecutionProvider'])
    
    for batch_size in [1, 2]:
        dummy_sensor = torch.randn(batch_size, 2, 50)
        dummy_visual = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            pt_out = pt_model(dummy_sensor, dummy_visual).numpy()
        ort_out = ort_session.run(None, {"sensor_input": dummy_sensor.numpy(), "visual_input": dummy_visual.numpy()})[0]
        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)
