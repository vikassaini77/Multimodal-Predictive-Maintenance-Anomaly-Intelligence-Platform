# IndustrialMind AI

A production-grade, multimodal predictive maintenance and anomaly intelligence platform targeting Bosch, Siemens Energy, Rockwell Automation, and NVIDIA India.

## Features
- **Two-Tower Model**: Sensor tower (1D-CNN + Transformer) + visual tower (EfficientNet-B4) fused via contrastive loss.
- **GNN Layer**: Equipment modeled as directed graphs, fault propagation via GraphSAGE.
- **ReAct Agent**: Powered by Gemini 2.5 Flash for anomaly diagnostics and alerting.
- **RAG Pipeline**: pgvector for maintenance manual retrieval.
- **Edge Deployment**: TensorRT FP16 support.
