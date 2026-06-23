from fastapi import APIRouter, Depends, HTTPException
import torch
import networkx as nx
from typing import List

from backend.app.schemas.graph import GraphInput, RiskPrediction
from backend.app.api.dependencies import get_model_container, ModelContainer
from backend.app.data.factory_graph import convert_to_pyg_heterodata
from backend.app.eval.explainer import GraphFaultExplainer

router = APIRouter()

@router.post("/predict", response_model=List[RiskPrediction])
async def predict_graph_faults(
    payload: GraphInput,
    models: ModelContainer = Depends(get_model_container)
):
    """
    Accepts a heterogeneous graph state, runs GraphSAGE, and returns fault risks 
    along with subgraph explanations.
    """
    if not payload.nodes:
        raise HTTPException(status_code=400, detail="Graph must contain nodes")

    nx_graph = nx.DiGraph()
    
    machine_embeddings = []
    conveyor_embeddings = []
    sensor_embeddings = []
    
    machine_ids = []
    conveyor_ids = []
    sensor_ids = []

    for node in payload.nodes:
        nx_graph.add_node(node.id, type=node.type)
        if node.type == "machine":
            machine_embeddings.append(node.features)
            machine_ids.append(node.id)
        elif node.type == "conveyor":
            conveyor_embeddings.append(node.features)
            conveyor_ids.append(node.id)
        elif node.type == "sensor":
            sensor_embeddings.append(node.features)
            sensor_ids.append(node.id)

    for edge in payload.edges:
        source_type = nx_graph.nodes[edge.source]['type']
        target_type = nx_graph.nodes[edge.target]['type']
        edge_tuple = (source_type, edge.type, target_type)
        nx_graph.add_edge(edge.source, edge.target, type=edge_tuple)

    try:
        data = convert_to_pyg_heterodata(
            nx_graph,
            torch.tensor(machine_embeddings) if machine_embeddings else None,
            torch.tensor(conveyor_embeddings) if conveyor_embeddings else None,
            torch.tensor(sensor_embeddings) if sensor_embeddings else None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process graph structure: {str(e)}")

    with torch.no_grad():
        node_embeddings = models.gnn(data.x_dict, data.edge_index_dict)
        if 'machine' not in node_embeddings:
            return []
            
        machine_logits = models.head(node_embeddings['machine']).squeeze(-1)
        # Ensure it's 1D
        if machine_logits.dim() == 0:
            machine_logits = machine_logits.unsqueeze(0)
        machine_probs = torch.sigmoid(machine_logits)

    explainer = GraphFaultExplainer(models.gnn, models.head)
    predictions = []
    
    probs_list = machine_probs.tolist()
    if isinstance(probs_list, float):
        probs_list = [probs_list]
        
    for idx, prob in enumerate(probs_list):
        is_fault = prob > 0.5
        explanations = []
        if is_fault:
            explanations = explainer.explain_prediction(data, idx, top_k=3)
            for exp in explanations:
                n_type = exp.node_type
                n_idx = int(exp.node_id)
                if n_type == "machine" and n_idx < len(machine_ids):
                    exp.node_id = machine_ids[n_idx]
                elif n_type == "conveyor" and n_idx < len(conveyor_ids):
                    exp.node_id = conveyor_ids[n_idx]
                elif n_type == "sensor" and n_idx < len(sensor_ids):
                    exp.node_id = sensor_ids[n_idx]
        
        predictions.append(RiskPrediction(
            node_id=machine_ids[idx],
            node_type="machine",
            fault_probability=prob,
            is_fault=is_fault,
            top_contributing_neighbors=explanations if explanations else None
        ))
        
    return predictions
