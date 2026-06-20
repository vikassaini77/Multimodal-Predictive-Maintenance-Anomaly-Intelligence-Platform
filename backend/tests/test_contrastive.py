import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.models.two_tower import TwoTowerAnomalyModel, NTXentLoss

def test_contrastive_loss():
    # 1. Setup Model
    batch_size = 4
    sensor_channels = 14
    seq_length = 50
    image_size = 224
    embed_dim = 256
    
    model = TwoTowerAnomalyModel(sensor_input_dim=sensor_channels, embed_dim=embed_dim)
    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 2. Dummy Data
    sensor_data = torch.randn(batch_size, sensor_channels, seq_length)
    visual_data = torch.randn(batch_size, 3, image_size, image_size)
    
    # 3. Before Training Step: Measure similarities
    model.eval()
    with torch.no_grad():
        out = model(sensor_data, visual_data)
        sensor_emb_pre = out['sensor_embeddings']
        visual_emb_pre = out['visual_embeddings']
        
        # Positive pair similarity (diagonal)
        sim_pre = torch.matmul(sensor_emb_pre, visual_emb_pre.T)
        pos_sim_pre = sim_pre.diag().mean().item()
        
        # Negative pair similarity (off-diagonal)
        mask = torch.eye(batch_size, dtype=torch.bool)
        neg_sim_pre = sim_pre[~mask].mean().item()
        
    print(f"Before Training: Mean Pos Sim: {pos_sim_pre:.4f}, Mean Neg Sim: {neg_sim_pre:.4f}")
    
    # Range should be between -1 and 1 since L2 normalized
    assert -1.0 <= pos_sim_pre <= 1.0
    assert -1.0 <= neg_sim_pre <= 1.0
    
    # 4. Training Step
    model.train()
    optimizer.zero_grad()
    out = model(sensor_data, visual_data)
    
    loss = criterion(out['sensor_embeddings'], out['visual_embeddings'], out['temperature'])
    loss.backward()
    
    # Check gradients flow
    assert model.sensor_tower.conv[0].weight.grad is not None
    assert model.visual_tower.projection_head[0].weight.grad is not None
    assert model.temperature.grad is not None
    
    optimizer.step()
    
    # 5. After Training Step: Measure similarities
    model.eval()
    with torch.no_grad():
        out = model(sensor_data, visual_data)
        sensor_emb_post = out['sensor_embeddings']
        visual_emb_post = out['visual_embeddings']
        
        sim_post = torch.matmul(sensor_emb_post, visual_emb_post.T)
        pos_sim_post = sim_post.diag().mean().item()
        neg_sim_post = sim_post[~mask].mean().item()
        
    print(f"After 1 Step: Mean Pos Sim: {pos_sim_post:.4f}, Mean Neg Sim: {neg_sim_post:.4f}")
    
    # The contrastive loss should pull positives closer than negatives,
    # or at least increase positive similarity relative to negative similarity.
    assert (pos_sim_post - neg_sim_post) > (pos_sim_pre - neg_sim_pre), \
        "Contrastive loss did not increase the margin between positives and negatives."
    print("Test Passed: Contrastive Loss works and gradients flow correctly.")

if __name__ == "__main__":
    test_contrastive_loss()
