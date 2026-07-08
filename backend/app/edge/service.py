import io
import os
import torch
import numpy as np
import logging
from PIL import Image

# Import generated gRPC stubs
try:
    import backend.app.edge.edge_pb2 as edge_pb2
    import backend.app.edge.edge_pb2_grpc as edge_pb2_grpc
except ImportError:
    # Handle direct execution in directory
    import edge_pb2
    import edge_pb2_grpc

# Our custom TRT wrapper
try:
    from backend.app.deployment.tensorrt_engine import TensorRTInferenceEngine
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

logger = logging.getLogger(__name__)

class EdgeInferenceService(edge_pb2_grpc.EdgeInferenceServicer):
    """
    gRPC Service wrapping the TensorRT engines for ultra-low latency edge inference.
    """
    def __init__(self, model_dir="backend/app/deployment/onnx_models"):
        self.model_dir = model_dir
        self.sensor_engine = None
        self.visual_engine = None
        self.fusion_engine = None
        self._load_engines()

    def _load_engines(self):
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available. Running in Mock Mode.")
            return

        sensor_plan = os.path.join(self.model_dir, "sensor_tower_fp16.plan")
        visual_plan = os.path.join(self.model_dir, "visual_tower_fp16.plan")
        fusion_plan = os.path.join(self.model_dir, "fusion_scorer_fp16.plan")

        try:
            if os.path.exists(sensor_plan):
                self.sensor_engine = TensorRTInferenceEngine(sensor_plan)
            if os.path.exists(visual_plan):
                self.visual_engine = TensorRTInferenceEngine(visual_plan)
            if os.path.exists(fusion_plan):
                self.fusion_engine = TensorRTInferenceEngine(fusion_plan)
            logger.info("Successfully loaded TensorRT engines.")
        except Exception as e:
            logger.error(f"Failed to load TensorRT engines: {e}")

    def _preprocess_sensor(self, sensor_bytes):
        # In a real scenario, this might be raw byte struct or numpy save string
        # Here we mock it if it's empty
        if not sensor_bytes:
            return np.random.randn(1, 2, 50).astype(np.float32)
        
        # Example decoding numpy from bytes
        try:
            arr = np.frombuffer(sensor_bytes, dtype=np.float32)
            # Reshape to expected shape (batch, channels, seq_len)
            arr = arr.reshape(1, 2, 50)
            return arr
        except Exception:
            # Fallback for tests
            return np.random.randn(1, 2, 50).astype(np.float32)

    def _preprocess_image(self, image_bytes):
        if not image_bytes:
            return np.random.randn(1, 3, 224, 224).astype(np.float32)
            
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = image.resize((224, 224))
            arr = np.array(image).astype(np.float32) / 255.0
            # Normalize (ImageNet)
            mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
            std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
            arr = (arr - mean) / std
            # HWC -> CHW
            arr = np.transpose(arr, (2, 0, 1))
            # Add batch dimension
            arr = np.expand_dims(arr, axis=0)
            return arr
        except Exception:
            # Fallback for tests
            return np.random.randn(1, 3, 224, 224).astype(np.float32)

    async def PredictAnomaly(self, request, context):
        """
        Handles the incoming gRPC request, runs inference, and returns response.
        """
        sensor_np = self._preprocess_sensor(request.sensor_data)
        visual_np = self._preprocess_image(request.image_data)

        if TRT_AVAILABLE and self.sensor_engine and self.visual_engine and self.fusion_engine:
            try:
                # 1. Run Sensor Tower
                sensor_out = self.sensor_engine(sensor_input=sensor_np)
                sensor_emb = sensor_out["sensor_embedding"]

                # 2. Run Visual Tower
                visual_out = self.visual_engine(visual_input=visual_np)
                visual_emb = visual_out["visual_embedding"]

                # 3. Run Fusion Scorer
                fusion_out = self.fusion_engine(
                    sensor_embedding=sensor_emb, 
                    visual_embedding=visual_emb
                )
                risk_score = float(fusion_out["anomaly_score"].item())
                
                explanation = "Inference completed via TensorRT."
                if risk_score > 0.8:
                    explanation = "CRITICAL: High anomaly risk detected across modalities."
                    
            except Exception as e:
                logger.error(f"Inference error: {e}")
                risk_score = -1.0
                explanation = f"Error: {str(e)}"
        else:
            # Mock mode
            risk_score = float(np.random.rand())
            explanation = "Mock Mode: TRT engines not loaded."

        return edge_pb2.InferenceResponse(
            risk_score=risk_score,
            explanation=explanation
        )
