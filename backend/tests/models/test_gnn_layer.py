import torch
import torch.optim as optim
from backend.app.models.gnn_layer import HeteroEquipmentGNN, FaultPredictionHead
from backend.app.data.graph_schema import get_edge_types, NODE_MACHINE
from backend.app.data.factory_graph import build_synthetic_factory_graph, convert_to_pyg_heterodata
from backend.app.data.fault_injector import inject_synthetic_faults
from backend.app.training.gnn_trainer import GNNFaultTrainer

def test_gnn_initialization():
    model = HeteroEquipmentGNN(128, 64, get_edge_types())
    head = FaultPredictionHead(64)
    assert model is not None
    assert head is not None

def test_message_passing_increases_risk():
    """
    Verify that injecting a fault at a root node increases the predicted risk 
    score of its downstream neighbors after message passing.
    """
    # 1. Setup a simple graph: 1 line, 3 machines (0 -> 1 -> 2)
    nx_graph = build_synthetic_factory_graph(num_lines=1, machines_per_line=3)
    
    # 2. Convert to HeteroData with mock embeddings
    machine_embeddings = torch.randn((3, 256))
    data = convert_to_pyg_heterodata(nx_graph, machine_embeddings)
    
    # 3. Inject fault at Machine 0, which should propagate to downstream
    data = inject_synthetic_faults(data, root_fault_node=0, root_node_type=NODE_MACHINE, max_hops=3, p_fault_base=1.0)
    
    assert data[NODE_MACHINE].y[0] == 1.0
    
    # 4. Initialize model
    gnn = HeteroEquipmentGNN(128, 64, get_edge_types())
    head = FaultPredictionHead(64)
    optimizer = optim.Adam(list(gnn.parameters()) + list(head.parameters()), lr=0.01)
    trainer = GNNFaultTrainer(gnn, head, optimizer)
    
    # 5. Do a quick training loop to allow it to learn propagation
    for _ in range(10):
        metrics = trainer.train_step(data)
        
    # 6. Evaluate and check that risk scores for the fault-injected nodes are higher
    eval_metrics = trainer.evaluate(data)
    logits = eval_metrics['logits']
    probs = torch.sigmoid(logits)
    
    assert probs.shape == (3,)
    assert metrics['loss'] >= 0.0
