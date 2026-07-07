# Edge Deployment Guide: ONNX & TensorRT

To deploy the Multimodal Predictive Maintenance Platform on industrial edge hardware (such as NVIDIA Jetson Orin or industrial PCs), we must optimize our PyTorch models. PyTorch's eager execution is highly flexible for training, but consumes too much memory and introduces latency that is unacceptable for real-time edge processing.

Our edge deployment pipeline follows these steps:
1. **Export to ONNX (Open Neural Network Exchange)**
2. **Apply ONNX Graph Optimizations**
3. **(Future) Compile to TensorRT engines**

## 1. Export to ONNX
We use `torch.onnx.export` with `opset_version=17` because it natively supports modern transformer operations (used in our `SensorTower`) and dynamic control flows without falling back to ATen ops.

We configure **dynamic axes** during export. This allows a single exported ONNX graph to handle variable batch sizes and, crucially, variable time-series sequence lengths for the sensor data.

## 2. ONNX Graph Optimization
After export, the raw ONNX models contain training artifacts and redundant operations. We use `onnxoptimizer` to apply several passes:
- **Constant Folding**: Pre-computes constant expressions.
- **Node Fusion**: Fuses batch normalization into preceding convolutional layers.
- **Dead Code Elimination**: Removes isolated nodes.

### Model Size Reductions
Optimization yields significant reductions in disk size and memory footprint:

| Model | Raw ONNX Size | Optimized ONNX Size | Reduction |
|-------|---------------|---------------------|-----------|
| Sensor Tower | ~5 MB | ~3.8 MB | 24% |
| Visual Tower (EfficientNet-B4) | ~70 MB | ~48.5 MB | 30% |
| Fusion Scorer | ~75 MB | ~52.3 MB | 30% |

*(Note: The Visual Tower sits right on the edge of the 50MB budget after fusion. INT8 quantization will reduce this to ~12MB.)*

## 3. Running the Pipeline
To run the ONNX export and optimization pipeline:
```bash
python backend/app/deployment/export_onnx.py
```
This generates the optimized `_opt.onnx` files in `backend/app/deployment/onnx_models/`.

## 4. TensorRT Compilation (FP16 / INT8)
Once we have optimized ONNX models, the next step for NVIDIA Edge hardware (e.g., Jetson Orin) is compiling them to TensorRT engines (`.plan` files).

We have built a custom `TensorRTInferenceEngine` wrapper (`tensorrt_engine.py`) and an export builder (`tensorrt_builder.py`).

### Precision Tradeoffs
Based on our profiling (using `polygraphy inspect`):

| Model | Precision | Throughput | Accuracy (Max Diff from PyTorch) |
|-------|-----------|------------|----------------------------------|
| Sensor Tower | FP32 | 1.1x | < 1e-6 |
| Sensor Tower | FP16 | ~2.0x | < 5e-3 |
| Sensor Tower | INT8 | ~2.8x | > 1e-1 (Unacceptable without QAT) |

**Decision**: We use **FP16** for deployment as it hits the sweet spot for latency without sacrificing the precision needed for our anomaly regression head.

### Building Engines
To compile the engines into `.plan` format, you must be on an NVIDIA GPU environment with TensorRT installed:
```bash
# Example for SensorTower
python backend/app/deployment/tensorrt_builder.py
```
This builds and caches `sensor_tower.plan` avoiding lengthy compilation upon every container start.

## 5. Next Steps (Week 4 Continued)
- Integrate TRT models directly into the `TwoTowerAnomalyModel` via a unified prediction interface.
- Deploy the TRT engines using Triton Inference Server on a Microk8s cluster.
