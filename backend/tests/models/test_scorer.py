import torch
import numpy as np
import pytest
from backend.app.models.scorer import AnomalyScorerHead, MahalanobisDistance
from backend.app.eval.calibration import PlattScaler, ThresholdTuner

def test_anomaly_scorer_head():
    scorer = AnomalyScorerHead(input_dim=256, hidden_dim=64)
    x = torch.randn(10, 256)
    
    # Test raw logits
    logits = scorer(x, return_probs=False)
    assert logits.shape == (10,)
    
    # Test probabilities
    probs = scorer(x, return_probs=True)
    assert probs.shape == (10,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

def test_mahalanobis_distance():
    mahalanobis = MahalanobisDistance(feature_dim=10)
    
    # Needs fit before forward
    with pytest.raises(RuntimeError):
        mahalanobis(torch.randn(5, 10))
        
    # Needs more samples than features
    with pytest.raises(ValueError):
        mahalanobis.fit(torch.randn(5, 10))
        
    normal_samples = torch.randn(100, 10)
    mahalanobis.fit(normal_samples)
    assert mahalanobis.is_fitted
    
    # Center should be roughly 0 for standard normal
    assert torch.allclose(mahalanobis.centroid, torch.zeros(10), atol=0.5)
    
    # Test inference
    distances = mahalanobis(torch.randn(5, 10))
    assert distances.shape == (5,)
    assert torch.all(distances >= 0.0) # Distances must be positive

def test_platt_scaler():
    scaler = PlattScaler()
    raw_scores = np.linspace(-5, 5, 100)
    # y_true tracks raw scores with some noise
    y_true = (raw_scores + np.random.normal(0, 1, 100) > 0).astype(int)
    
    with pytest.raises(RuntimeError):
        scaler.predict_proba(raw_scores)
        
    scaler.fit(raw_scores, y_true)
    assert scaler.is_fitted
    
    probs = scaler.predict_proba(raw_scores)
    assert probs.shape == (100,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    
    # Monotonicity test: higher logit should yield higher probability
    # Logistic regression naturally guarantees this
    assert np.all(np.diff(probs) >= -1e-6)

def test_threshold_tuner():
    tuner = ThresholdTuner()
    
    y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1, 1])
    
    with pytest.raises(RuntimeError):
        tuner.predict(y_prob)
        
    results = tuner.fit(y_prob, y_true)
    assert tuner.is_fitted
    
    assert "best_threshold" in results
    assert "best_f1" in results
    
    # Given the clear separation at 0.6, the threshold should be <= 0.6 and > 0.4
    assert 0.4 < results["best_threshold"] <= 0.6
    
    preds = tuner.predict(y_prob)
    assert np.array_equal(preds, np.array([0, 0, 1, 1, 1]))
