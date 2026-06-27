# Changelog

All notable changes to this project will be documented in this file.

## [v0.2.0] - End of Week 2
### Added
- **GraphSAGE Fault Propagation**: `HeteroEquipmentGNN` models machines, sensors, and conveyors to propagate fault risk.
- **Explainability**: `GNNExplainer` integrated to retrieve subgraph contributions for anomaly predictions.
- **FastAPI Endpoints**: `/graph/predict` and `/predict/full` endpoints for serving end-to-end multimodal graph predictions.
- **Model Quantization**: FP16 mixed-precision inference via `torch.autocast`.
- **Structured Logging**: Context-aware logging injecting `trace_id` for request tracing across the pipeline.
- **Centralized Config**: `IndustrialMindConfig` managing dimensions, Redis URLs, and flags using Pydantic settings.
- **Load Testing**: Locust suite for stress-testing the pipeline (100 concurrent requests).

### Changed
- Config parameters extracted from model definitions into global `settings`.
- Redis caching falls back gracefully to `LRUCache`.

## [v0.1.0] - End of Week 1
### Added
- Two-Tower model architecture (1D-CNN + Transformer for Sensor, EfficientNet for Visual).
- NT-Xent Contrastive Loss for multimodal alignment.
- Basic API structure.
