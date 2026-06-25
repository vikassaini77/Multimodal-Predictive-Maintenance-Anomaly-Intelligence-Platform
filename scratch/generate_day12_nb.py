import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = """# Day 12: Anomaly Scorer Evaluation
End-to-end evaluation of the anomaly scorer head, Platt scaling calibration, and threshold tuning on the Precision-Recall curve.
"""

code_1 = """import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from backend.app.models.scorer import AnomalyScorerHead, MahalanobisDistance
from backend.app.eval.calibration import PlattScaler, ThresholdTuner
from backend.app.eval.metrics import bootstrap_confidence_intervals

# Generate Mock Data
np.random.seed(42)
torch.manual_seed(42)

N_TRAIN = 1000
N_VAL = 500
FEATURE_DIM = 256

# Training Data (Normal vs Anomaly)
train_features = torch.randn(N_TRAIN, FEATURE_DIM)
train_labels = torch.randint(0, 2, (N_TRAIN,))
# Make anomalies have a shifted mean
train_features[train_labels == 1] += 0.5

# Validation Data
val_features = torch.randn(N_VAL, FEATURE_DIM)
val_labels = torch.randint(0, 2, (N_VAL,))
val_features[val_labels == 1] += 0.5

print(f"Generated {N_TRAIN} training samples and {N_VAL} validation samples.")
"""

markdown_2 = """## 1. Train Anomaly Scorer Head"""

code_2 = """# Initialize Scorer
scorer = AnomalyScorerHead(input_dim=FEATURE_DIM, hidden_dim=64)
optimizer = torch.optim.Adam(scorer.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

# Quick training loop
scorer.train()
for epoch in range(10):
    optimizer.zero_grad()
    logits = scorer(train_features)
    loss = criterion(logits, train_labels.float())
    loss.backward()
    optimizer.step()

scorer.eval()
with torch.no_grad():
    val_logits = scorer(val_features).numpy()
    val_probs = scorer(val_features, return_probs=True).numpy()

print("Scorer trained.")
"""

markdown_3 = """## 2. Platt Scaling Calibration"""

code_3 = """scaler = PlattScaler()
scaler.fit(val_logits, val_labels.numpy())
calibrated_probs = scaler.predict_proba(val_logits)

plt.figure(figsize=(10, 4))
plt.hist(val_probs[val_labels.numpy()==0], bins=50, alpha=0.5, label='Normal (Raw Sigmoid)')
plt.hist(val_probs[val_labels.numpy()==1], bins=50, alpha=0.5, label='Anomaly (Raw Sigmoid)')
plt.title("Raw Sigmoid Probabilities")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.hist(calibrated_probs[val_labels.numpy()==0], bins=50, alpha=0.5, label='Normal (Calibrated)')
plt.hist(calibrated_probs[val_labels.numpy()==1], bins=50, alpha=0.5, label='Anomaly (Calibrated)')
plt.title("Calibrated Probabilities (Platt Scaling)")
plt.legend()
plt.show()
"""

markdown_4 = """## 3. Threshold Tuning"""

code_4 = """tuner = ThresholdTuner()
tuner_results = tuner.fit(calibrated_probs, val_labels.numpy())

best_threshold = tuner_results['best_threshold']
print(f"Optimal Threshold (Max F1): {best_threshold:.4f}")
print(f"Validation F1 at Optimal Threshold: {tuner_results['best_f1']:.4f}")

# Apply threshold
final_preds = tuner.predict(calibrated_probs)
"""

markdown_5 = """## 4. Bootstrap Confidence Intervals"""

code_5 = """# AUROC CI
auroc_lower, auroc_upper = bootstrap_confidence_intervals(
    y_true=val_labels.numpy(),
    y_pred=calibrated_probs,
    metric_func=roc_auc_score
)
print(f"AUROC: {roc_auc_score(val_labels.numpy(), calibrated_probs):.4f} (95% CI: [{auroc_lower:.4f}, {auroc_upper:.4f}])")

# F1 CI
def f1_wrapper(y_true, y_pred_prob):
    y_pred_bin = (y_pred_prob >= best_threshold).astype(int)
    return f1_score(y_true, y_pred_bin)

f1_lower, f1_upper = bootstrap_confidence_intervals(
    y_true=val_labels.numpy(),
    y_pred=calibrated_probs,
    metric_func=f1_wrapper
)
print(f"F1 Score: {tuner_results['best_f1']:.4f} (95% CI: [{f1_lower:.4f}, {f1_upper:.4f}])")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(markdown_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(markdown_5),
    nbf.v4.new_code_cell(code_5)
]

with open('experiments/12_anomaly_scorer_eval.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Day 12 Notebook created successfully!")
