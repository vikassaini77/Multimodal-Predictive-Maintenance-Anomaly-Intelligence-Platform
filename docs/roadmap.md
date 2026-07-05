# Project Roadmap

## Week 1: Data & Multimodal Deep Learning (Completed)
- Time-series synthetic generation (vibration, temp).
- MVTEC-AD visual anomaly dataset pipeline.
- ResNet50 + CNN Two-Tower Contrastive Model.
- GraphSAGE topology mapping for fault propagation.

## Week 2: RAG & Embeddings (Completed)
- `sentence-transformers` vectorization of text documents.
- `pgvector` indexing and retrieval.
- Gemini LLM generation with contextual grounding.

## Week 3: Autonomous Agentic Reasoning (Completed)
- ReAct Agent implementation for diagnostic logic.
- Native Gemini tool calling for fetching sensor and visual data.
- WebSocket chat API and memory management.
- Guardrails (Rate Limiting, Action Confirmation).
- Benchmarking (Task Success, Faithfulness).

## Week 4: Edge Deployment & Optimization (Upcoming)
- **TensorRT Optimization**: Export the PyTorch Two-Tower model and GraphSAGE to ONNX, then compile with NVIDIA TensorRT to drastically reduce latency and memory footprint.
- **Edge Architecture**: Package the optimized models into a lightweight C++ / Python inference server using Triton Inference Server.
- **Microk8s on Jetson**: Prepare deployment manifests for running the platform on a factory-floor edge device (e.g., NVIDIA Jetson Orin) via Microk8s.
- **Quantization**: Explore INT8 quantization to further speed up the ResNet50 visual tower without losing accuracy.
