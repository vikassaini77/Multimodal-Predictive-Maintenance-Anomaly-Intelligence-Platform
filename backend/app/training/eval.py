import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

class EmbeddingVisualizer:
    """
    Visualizes joint embeddings from the Two-Tower Contrastive Model using t-SNE.
    Colors points by their anomaly label to verify if the model separates
    normal and anomalous representations in the joint space.
    """
    def __init__(self, model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        
    def extract_embeddings(self):
        self.model.eval()
        
        sensor_embs = []
        visual_embs = []
        labels_list = []
        
        with torch.no_grad():
            for sensor_data, visual_data, labels in self.dataloader:
                sensor_data = sensor_data.to(self.device)
                visual_data = visual_data.to(self.device)
                
                out = self.model(sensor_data, visual_data)
                
                sensor_embs.append(out['sensor_embeddings'].cpu().numpy())
                visual_embs.append(out['visual_embeddings'].cpu().numpy())
                labels_list.append(labels.numpy())
                
        sensor_embs = np.concatenate(sensor_embs, axis=0)
        visual_embs = np.concatenate(visual_embs, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return sensor_embs, visual_embs, labels
        
    def plot_tsne(self, save_path=None):
        print("Extracting embeddings for t-SNE...")
        sensor_embs, visual_embs, labels = self.extract_embeddings()
        
        # Combine sensor and visual embeddings to project them together
        # We will label them to distinguish modality, or just plot joint points
        # To show they are aligned, we can plot them in the same space
        all_embs = np.vstack([sensor_embs, visual_embs])
        
        # Labels for normal/anomaly
        all_labels = np.concatenate([labels, labels])
        
        # Labels for modality (0 for sensor, 1 for visual)
        modality_labels = np.concatenate([np.zeros(len(labels)), np.ones(len(labels))])
        
        print("Running t-SNE (this might take a minute)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embs)-1))
        embs_2d = tsne.fit_transform(all_embs)
        
        sensor_2d = embs_2d[:len(sensor_embs)]
        visual_2d = embs_2d[len(sensor_embs):]
        
        plt.figure(figsize=(10, 8))
        
        # Plot Normal Sensor (Circle, Blue)
        idx_normal = labels == 0
        plt.scatter(sensor_2d[idx_normal, 0], sensor_2d[idx_normal, 1], 
                    c='blue', marker='o', alpha=0.6, label='Sensor (Normal)')
                    
        # Plot Anomaly Sensor (Circle, Red)
        idx_anomaly = labels == 1
        plt.scatter(sensor_2d[idx_anomaly, 0], sensor_2d[idx_anomaly, 1], 
                    c='red', marker='o', alpha=0.6, label='Sensor (Anomaly)')
                    
        # Plot Normal Visual (Triangle, Light Blue)
        plt.scatter(visual_2d[idx_normal, 0], visual_2d[idx_normal, 1], 
                    c='cyan', marker='^', alpha=0.6, label='Visual (Normal)')
                    
        # Plot Anomaly Visual (Triangle, Orange)
        plt.scatter(visual_2d[idx_anomaly, 0], visual_2d[idx_anomaly, 1], 
                    c='orange', marker='^', alpha=0.6, label='Visual (Anomaly)')
                    
        # Optionally draw lines between matched pairs
        for i in range(len(sensor_2d)):
            plt.plot([sensor_2d[i, 0], visual_2d[i, 0]], 
                     [sensor_2d[i, 1], visual_2d[i, 1]], 
                     'k-', alpha=0.05)
                     
        plt.title('t-SNE Visualization of Joint Sensor-Visual Embeddings')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            
        plt.show()
