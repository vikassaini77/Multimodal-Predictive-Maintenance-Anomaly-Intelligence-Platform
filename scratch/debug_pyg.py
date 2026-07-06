import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

data = HeteroData()
data['machine'].x = torch.randn(3, 256)
data['conveyor'].x = torch.randn(2, 256)
data['sensor'].x = torch.randn(3, 256)

data['machine', 'feeds_into', 'conveyor'].edge_index = torch.tensor([[0, 1], [0, 1]])
data['conveyor', 'feeds_into_rev', 'machine'].edge_index = torch.tensor([[0, 1], [1, 2]])
data['sensor', 'monitors', 'machine'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])

conv = HeteroConv({
    ('machine', 'feeds_into', 'conveyor'): SAGEConv((-1, -1), 64, aggr='mean'),
    ('conveyor', 'feeds_into_rev', 'machine'): SAGEConv((-1, -1), 64, aggr='mean'),
    ('sensor', 'monitors', 'machine'): SAGEConv((-1, -1), 64, aggr='mean')
}, aggr='sum')

out = conv(data.x_dict, data.edge_index_dict)
print(out.keys())
