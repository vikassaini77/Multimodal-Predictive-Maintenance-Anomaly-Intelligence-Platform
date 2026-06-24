import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = """# Day 11: Cross-Modal Attention Fusion
In this notebook, we instantiate the `MultimodalFusionModel` and visualize the attention weights and gating behavior.
"""

code_1 = """import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from backend.app.models.cross_attention import MultimodalFusionModel

# 1. Initialize the Fusion Model
sensor_input_dim = 14  # Example CMAPSS dimension
model = MultimodalFusionModel(sensor_input_dim=sensor_input_dim, embed_dim=256)
model.eval()
print("Model Initialized!")
"""

markdown_2 = """## Testing Missing Modality Gating
Let's see how the learnable gate handles missing visual data.
"""

code_2 = """# Mock data
sensor_data = torch.randn(2, sensor_input_dim, 50)
visual_data = torch.randn(2, 3, 224, 224)

# Missing Visual Modality
out_missing_vis, aux_missing_vis = model(sensor_data=sensor_data, visual_data=None)
print("Missing Visual - Gate Weights (Sensor, Visual):")
print(aux_missing_vis['gate_weights'].detach().numpy())

# Missing Sensor Modality
out_missing_sen, aux_missing_sen = model(sensor_data=None, visual_data=visual_data)
print("\\nMissing Sensor - Gate Weights (Sensor, Visual):")
print(aux_missing_sen['gate_weights'].detach().numpy())
"""

markdown_3 = """## Visualizing Cross-Attention Weights
When both modalities are present, we compute the bidirectional cross-attention weights.
"""

code_3 = """# Both Modalities Present
out_full, aux_full = model(sensor_data=sensor_data, visual_data=visual_data)
gate_weights = aux_full['gate_weights'].detach().numpy()
print("Full Modality - Gate Weights:")
print(gate_weights)

attn_weights = aux_full['attn_weights']
s2v = attn_weights['s2v'].detach().numpy()[0] # Batch 0
v2s = attn_weights['v2s'].detach().numpy()[0] # Batch 0

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(s2v, ax=axes[0], cmap='viridis')
axes[0].set_title("Sensor queries Visual (S2V) Attention")
axes[0].set_xlabel("Visual Seq Length")
axes[0].set_ylabel("Sensor Seq Length")

sns.heatmap(v2s, ax=axes[1], cmap='viridis')
axes[1].set_title("Visual queries Sensor (V2S) Attention")
axes[1].set_xlabel("Sensor Seq Length")
axes[1].set_ylabel("Visual Seq Length")

plt.tight_layout()
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_code_cell(code_3)
]

with open('experiments/11_cross_attention_fusion.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
