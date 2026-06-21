import matplotlib.pyplot as plt
import networkx as nx
from backend.app.data.graph_schema import NODE_MACHINE, NODE_CONVEYOR, NODE_SENSOR

def visualize_factory_graph(nx_graph: nx.DiGraph, anomalous_nodes=None):
    """
    Visualizes the factory graph using matplotlib and networkx.draw.
    Color-codes nodes by machine type and highlights anomalous nodes.
    """
    if anomalous_nodes is None:
        anomalous_nodes = []
        
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(nx_graph, seed=42)
    
    # Extract nodes by type
    machine_nodes = [n for n, d in nx_graph.nodes(data=True) if d['type'] == NODE_MACHINE]
    conveyor_nodes = [n for n, d in nx_graph.nodes(data=True) if d['type'] == NODE_CONVEYOR]
    sensor_nodes = [n for n, d in nx_graph.nodes(data=True) if d['type'] == NODE_SENSOR]
    
    # Determine colors for machines (red if anomalous, blue otherwise)
    machine_colors = ['red' if n in anomalous_nodes else 'lightblue' for n in machine_nodes]
    
    # Draw nodes
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=machine_nodes, 
                           node_color=machine_colors, node_size=800, node_shape='s', label='Machines')
    
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=conveyor_nodes, 
                           node_color='gray', node_size=400, node_shape='o', label='Conveyors')
                           
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=sensor_nodes, 
                           node_color='lightgreen', node_size=300, node_shape='^', label='Sensors')
    
    # Draw edges
    nx.draw_networkx_edges(nx_graph, pos, arrows=True, arrowsize=15, alpha=0.7)
    
    # Draw labels
    labels = {n: n.split('_')[-1] for n in nx_graph.nodes()}
    nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=10, font_family="sans-serif")
    
    plt.title("Equipment Factory Topology Graph")
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
