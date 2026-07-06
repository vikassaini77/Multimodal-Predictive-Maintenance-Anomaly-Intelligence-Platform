import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = """# Day 13: Pipeline Integration & Profiling
End-to-end integration of the Anomaly Detection Pipeline including caching and batch dynamic padding.
"""

code_1 = """import torch
import time
import cProfile
import pstats
from torch.profiler import profile, record_function, ProfilerActivity

from backend.app.api.dependencies import get_model_container
from backend.app.schemas.graph import FullPredictRequest, GraphInput, NodeInput, EdgeInput

# Initialize Pipeline
models = get_model_container()
pipeline = models.pipeline
print("Pipeline Initialized")
"""

markdown_2 = """## End-to-End Smoke Test (50 Scenarios)"""

code_2 = """# Mock Graph
graph_input = GraphInput(
    nodes=[
        NodeInput(id="M1", type="machine", features=[0.1]*64),
        NodeInput(id="S1", type="sensor", features=[0.2]*64)
    ],
    edges=[
        EdgeInput(source="S1", target="M1", type="monitors")
    ]
)

success = 0
failed = 0
latencies = []

print("Running 50 smoke tests...")
for i in range(50):
    try:
        req = FullPredictRequest(
            machine_id=f"M1",
            timestamp=float(i), # different timestamps bypass cache
            sensor_data=[[0.5]*50 for _ in range(14)], # 14 channels, seq 50
            visual_data=[[[0.1]*224 for _ in range(224)] for _ in range(3)],
            graph=graph_input
        )
        
        start = time.time()
        # Direct method call for testing
        sensor_tensor = torch.tensor(req.sensor_data, dtype=torch.float32)
        visual_tensor = torch.tensor(req.visual_data, dtype=torch.float32)
        
        # Dummy hetero_graph
        from backend.app.data.factory_graph import convert_to_pyg_heterodata
        import networkx as nx
        nx_graph = nx.DiGraph()
        nx_graph.add_node("M1", type="machine")
        nx_graph.add_node("S1", type="sensor")
        nx_graph.add_edge("S1", "M1", type=("sensor", "monitors", "machine"))
        hetero_graph = convert_to_pyg_heterodata(nx_graph)
        
        res = pipeline.predict("M1", req.timestamp, sensor_tensor, visual_tensor, hetero_graph)
        latencies.append(time.time() - start)
        success += 1
    except Exception as e:
        print(e)
        failed += 1

print(f"Smoke Test Results: {success} Passed, {failed} Failed")
print(f"Average Latency: {sum(latencies)/len(latencies)*1000:.2f} ms")
"""

markdown_3 = """## Profiling"""

code_3 = """# PyTorch Profiler
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("e2e_inference"):
        pipeline.predict("M1", 999.0, sensor_tensor, visual_tensor, hetero_graph)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# The Visual Tower (Conv2D operations) is typically the bottleneck
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_code_cell(code_3)
]

with open('experiments/13_pipeline_integration.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
