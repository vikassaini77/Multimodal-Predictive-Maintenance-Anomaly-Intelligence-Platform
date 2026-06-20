# IndustrialMind AI

A production-grade, multimodal predictive maintenance and anomaly intelligence platform targeting Bosch, Siemens Energy, Rockwell Automation, and NVIDIA India.

## Features
- **Two-Tower Model**: Sensor tower (1D-CNN + Transformer) + visual tower (EfficientNet-B4) fused via contrastive loss.
- **GNN Layer**: Equipment modeled as directed graphs, fault propagation via GraphSAGE.
- **ReAct Agent**: Powered by Gemini 2.5 Flash for anomaly diagnostics and alerting.
- **RAG Pipeline**: pgvector for maintenance manual retrieval.
- **Edge Deployment**: TensorRT FP16 support.

## Week 1 Progress: Two-Tower Contrastive Training
We have successfully implemented the multimodal data alignment and Two-Tower model using NT-Xent loss.

### Experiments Table
| Experiment | Modality | Loss Type | Notes |
|---|---|---|---|
| 04_sensor_tower | Sensor-only | MSE | Baseline CMAPSS model |
| 05_visual_tower | Visual-only | Focal | EfficientNet on MVTec |
| 07_two_tower_v1 | Multimodal | NT-Xent | Joint training, Week 1 milestone |

### Week 2 Preparation
- Start graph modeling for equipment hierarchies (GNN layer).
- Retrieve maintenance manuals with pgvector for the RAG pipeline.
