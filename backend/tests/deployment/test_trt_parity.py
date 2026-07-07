import os
import torch
import numpy as np
import pytest

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    from backend.app.deployment.tensorrt_engine import TensorRTInferenceEngine
    from backend.app.deployment.tensorrt_builder import build_engine
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    trt = None

import onnxruntime as ort

ONNX_DIR = "backend/app/deployment/onnx_models"
os.makedirs(ONNX_DIR, exist_ok=True)

@pytest.fixture(scope="module")
def trt_engine_setup():
    if not HAS_TRT:
        pytest.skip("TensorRT is not installed")
        
    onnx_path = os.path.join(ONNX_DIR, "test_sensor_tower_opt.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip(f"ONNX model missing: {onnx_path}")
        
    trt_fp16_path = os.path.join(ONNX_DIR, "test_sensor_tower_fp16.plan")
    
    # Build FP16 engine if missing
    if not os.path.exists(trt_fp16_path):
        success = build_engine(onnx_path, trt_fp16_path, fp16_mode=True)
        assert success, "Failed to build FP16 engine"
        
    return onnx_path, trt_fp16_path

@pytest.mark.skipif(not HAS_TRT, reason="TensorRT not installed")
def test_trt_fp16_parity(trt_engine_setup):
    onnx_path, trt_fp16_path = trt_engine_setup
    
    # Load ONNX session
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Load TRT Engine
    trt_engine = TensorRTInferenceEngine(trt_fp16_path)
    
    # Test batch size 1 and 4
    for batch_size in [1, 4]:
        seq_len = 50
        dummy_input = torch.randn(batch_size, 2, seq_len)
        input_np = dummy_input.numpy()
        
        # ONNX baseline
        ort_out = ort_session.run(None, {"sensor_input": input_np})[0]
        
        # TRT FP16
        trt_out = trt_engine(sensor_input=input_np)["sensor_embedding"]
        
        # Assert parity within tolerance
        # FP16 can drift, so we use atol=5e-3 and rtol=1e-2
        np.testing.assert_allclose(ort_out, trt_out, rtol=1e-2, atol=5e-3)
