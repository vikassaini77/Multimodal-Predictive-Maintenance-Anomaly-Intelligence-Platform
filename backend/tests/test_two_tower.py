import torch
import sys
import os

# Add the backend directory to the path so we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.models.two_tower import TwoTowerAnomalyModel

def test_model_shapes():
    print("Initializing the Two-Tower Model...")
    
    # 1. Define dimensions
    batch_size = 8
    sensor_channels = 14  # e.g., 14 different sensors from NASA CMAPSS
    seq_length = 50       # e.g., 50 time steps of data
    image_size = 224      # Standard image size for EfficientNet
    embed_dim = 256       # The final embedding size we want
    
    model = TwoTowerAnomalyModel(sensor_input_dim=sensor_channels, embed_dim=embed_dim, freeze_visual=True)
    model.eval() # Set to evaluation mode for testing
    
    # 2. Create Dummy Data (Random Tensors)
    # This simulates what a real batch of data will look like
    print("Generating fake sensor and visual data...")
    dummy_sensor_data = torch.randn(batch_size, sensor_channels, seq_length)
    dummy_visual_data = torch.randn(batch_size, 3, image_size, image_size)
    
    # 3. Pass data through the model
    print("Passing data through the forward pass...")
    try:
        output = model(dummy_sensor_data, dummy_visual_data)
        
        # 4. Verify the outputs
        print("\n--- TEST SUCCESS ---")
        print(f"Sensor Embedding Shape: {output['sensor_embeddings'].shape} (Expected: [{batch_size}, {embed_dim}])")
        print(f"Visual Embedding Shape: {output['visual_embeddings'].shape} (Expected: [{batch_size}, {embed_dim}])")
        
        # Test inference method
        anomaly_score = model.predict_anomaly(dummy_sensor_data, dummy_visual_data)
        print(f"Anomaly Score Shape: {anomaly_score.shape} (Expected: [{batch_size}])")
        
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model_shapes()
