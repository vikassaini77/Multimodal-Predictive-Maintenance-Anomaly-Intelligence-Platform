import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer
from typing import List
from backend.app.schemas.graph import NodeExplanation

class GraphFaultExplainer:
    def __init__(self, gnn_model: nn.Module, head_model: nn.Module):
        self.gnn = gnn_model
        self.head = head_model
        
        # Wrapper to combine GNN and head for Explainer
        class FullModel(nn.Module):
            def __init__(self, gnn, head):
                super().__init__()
                self.gnn = gnn
                self.head = head
                
            def forward(self, x_dict, edge_index_dict):
                x_dict = self.gnn(x_dict, edge_index_dict)
                # Explaining 'machine' nodes
                return self.head(x_dict['machine']).squeeze(-1)
                
        self.full_model = FullModel(self.gnn, self.head)
        
        self.explainer = Explainer(
            model=self.full_model,
            algorithm=GNNExplainer(epochs=50),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )

    def explain_prediction(self, data: HeteroData, node_index: int, top_k: int = 3) -> List[NodeExplanation]:
        """
        Explain the fault prediction for a specific machine node.
        Returns the top-K contributing neighbor machines/sensors/conveyors.
        """
        explanation = self.explainer(data.x_dict, data.edge_index_dict, index=node_index)
        
        results = []
        node_mask_dict = explanation.node_mask_dict
        
        for n_type, mask in node_mask_dict.items():
            if mask is not None:
                scores = mask.squeeze().tolist()
                if isinstance(scores, float):
                    scores = [scores]
                for idx, score in enumerate(scores):
                    if score > 0.05: # Filter
                        results.append(NodeExplanation(
                            node_id=str(idx), 
                            node_type=n_type,
                            importance_score=score
                        ))
        
        results.sort(key=lambda x: x.importance_score, reverse=True)
        return results[:top_k]
