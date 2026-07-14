# Experimental Notebooks Index

Over the 4-week development lifecycle, we conducted 21 structured experiments. This index tracks the iterative improvements from basic baselines to our final multimodal ReAct agent architecture.

| Week | Notebook | Focus Area | Key Outcome / Metric |
|------|----------|------------|----------------------|
| **1** | `01_baseline_cnn.ipynb` | Sensor Anomaly Detection | AUROC: 0.81 (1D CNN) |
| **1** | `02_vision_resnet.ipynb` | Visual Defect Detection | AUROC: 0.85 (ResNet50) |
| **1** | `03_multimodal_fusion.ipynb` | Late Fusion (Sensor + Vision) | AUROC: 0.89 |
| **1** | `04_contrastive_learning.ipynb` | Contrastive Multimodal Embeddings | AUROC: 0.94 |
| **1** | `05_edge_quantization.ipynb` | INT8 Quantization (TensorRT) | Latency reduced to 45ms |
| **2** | `06_topology_graph_init.ipynb` | Factory Graph Construction | NetworkX graph initialized |
| **2** | `07_gcn_baseline.ipynb` | Basic Graph Convolution | F1: 0.76 on Fault Propagation |
| **2** | `08_graphsage_training.ipynb` | GraphSAGE Implementation | F1: 0.88 on Fault Propagation |
| **2** | `09_temporal_graphsage.ipynb` | Adding Time-series to GraphSAGE | F1: 0.92 |
| **2** | `10_gnn_explainability.ipynb` | GNNExplainer for Subgraphs | Explanations match ground truth |
| **3** | `11_pgvector_setup.ipynb` | Vector DB Initialization | Latency: <10ms for ANN |
| **3** | `12_oem_manual_chunking.ipynb` | PDF Parsing & Chunking | 500-token semantic chunks |
| **3** | `13_rag_retrieval_eval.ipynb` | BM25 + Dense Hybrid Search | Recall@5: 96% |
| **3** | `14_llm_prompt_tuning.ipynb` | Zero-shot vs Few-shot prompts | Few-shot yields better formatting |
| **3** | `15_react_agent_loop.ipynb` | ReAct (Reason+Act) Loop | Agent successfully uses tools |
| **4** | `16_fastapi_websockets.ipynb` | Real-time Streaming API | Sub-second token delivery |
| **4** | `17_frontend_react_dashboard.ipynb` | UI Prototyping | Dashboard layout confirmed |
| **4** | `18_end_to_end_integration.ipynb` | Full Pipeline Test | Latency: 1.2s total |
| **4** | `19_docker_compose_setup.ipynb` | Orchestration | All services boot together |
| **4** | `20_monitoring_grafana.ipynb` | Prometheus Metrics | Dashboards operational |
| **4** | `21_final_evaluation.ipynb` | Final V1.0 Benchmark | **AUROC: 0.965, F1: 0.942** |

*Note: Notebooks are archived in the `scratch/` directory for reference.*
