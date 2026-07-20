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

from backend.app.schemas.graph import FullPredictRequest, FullPredictResponse, AsyncPredictResponse, JobStatusResponse
from backend.app.worker import predict_full_pipeline_task
from celery.result import AsyncResult

@router.post("/predict/full", response_model=AsyncPredictResponse)
async def predict_full_pipeline(
    payload: FullPredictRequest
):
    """
    End-to-End Pipeline: Enqueues a Celery task for async inference
    """
    try:
        task = predict_full_pipeline_task.delay(payload.model_dump())
        return AsyncPredictResponse(job_id=task.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue task: {str(e)}")

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Check the status of an async inference job and return results if ready.
    """
    try:
        task_result = AsyncResult(job_id)
        
        status = task_result.status
        result_data = None
        
        if status == 'SUCCESS':
            result_data = FullPredictResponse(**task_result.result)
            
        return JobStatusResponse(
            job_id=job_id,
            status=status.lower(),
            result=result_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch job status: {str(e)}")
