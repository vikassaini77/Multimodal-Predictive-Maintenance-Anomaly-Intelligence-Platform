"""
Equipment Graph Schema Definition

Defines the node types and edge types for the factory topology.
"""

# Node Types
NODE_MACHINE = "machine"
NODE_CONVEYOR = "conveyor"
NODE_SENSOR = "sensor"

# Edge Types
# Tuples defining (source_node_type, edge_relation, target_node_type)
EDGE_FEEDS_INTO = (NODE_MACHINE, "feeds_into", NODE_CONVEYOR)
EDGE_FEEDS_INTO_REV = (NODE_CONVEYOR, "feeds_into", NODE_MACHINE)
EDGE_MONITORS = (NODE_SENSOR, "monitors", NODE_MACHINE)

def get_node_types():
    """Returns a list of all valid node types."""
    return [NODE_MACHINE, NODE_CONVEYOR, NODE_SENSOR]

def get_edge_types():
    """Returns a list of all valid edge types in PyG standard triplet format."""
    return [EDGE_FEEDS_INTO, EDGE_FEEDS_INTO_REV, EDGE_MONITORS]
