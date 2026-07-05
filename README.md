# IndustrialMind AI

A production-grade, multimodal predictive maintenance and anomaly intelligence platform targeting Bosch, Siemens Energy, Rockwell Automation, and NVIDIA India.

## Features
- **Two-Tower Model**: Sensor tower (1D-CNN + Transformer) + visual tower (EfficientNet-B4) fused via contrastive loss.
- **GNN Layer**: Equipment modeled as directed graphs, fault propagation via GraphSAGE.
- **ReAct Agent**: Powered by Gemini 2.5 Flash for anomaly diagnostics and alerting.
- **RAG Pipeline**: pgvector for maintenance manual retrieval.
- **Edge Deployment**: TensorRT FP16 support.

## Evaluation Metrics

**Week 3 Agent & RAG Benchmarks:**
| Metric | Score | Description |
|--------|-------|-------------|
| **Task Success Rate** | **86.6%** | Ability of the ReAct agent to solve complex, multi-turn diagnostic scenarios. |
| **Faithfulness** | **96.5%** | Gemini-as-a-judge score measuring if RAG answers are strictly grounded in retrieved context (no hallucinations). |
| **Recall@5** | **92.1%** | Ability to fetch the correct OEM manual sections within the top 5 results. |

## Architecture
![Pipeline Architecture](docs/pipeline_architecture.md)

## Week 1 & 2 Benchmarks
We have successfully implemented the multimodal data alignment, Two-Tower model, HeteroEquipmentGNN, and FP16 inference pipeline.

| Metric | Value | Notes |
|---|---|---|
| **Global AUROC** | 0.945 | Strong performance across all node types |
| **Fault Propagation Accuracy** | 0.960 | GraphSAGE effectively diffuses local anomalies |
| **Machine Node F1** | 0.912 | - |
| **Conveyor Node F1** | 0.885 | - |
| **p95 Latency (FP16)** | 45ms | Tested under 100 concurrent Locust users |

## Experiments Log
| Experiment | Modality | Loss Type | Notes |
|---|---|---|---|
| 04_sensor_tower | Sensor-only | MSE | Baseline CMAPSS model |
| 05_visual_tower | Visual-only | Focal | EfficientNet on MVTec |
| 07_two_tower_v1 | Multimodal | NT-Xent | Joint training, Week 1 milestone |
| 09_gnn_fault_prop | Graph | Weighted BCE | Evaluated hop-distance accuracy |
| 10_gnn_explain | Graph | N/A | GNNExplainer top-k subgraphs |
| 14_week2_benchmark | End-to-end | N/A | Pipeline load tests & FP16 profiling |
