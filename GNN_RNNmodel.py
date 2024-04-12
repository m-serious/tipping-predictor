import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_max_pool, GATConv, SAGEConv
from torch.nn import LSTM, GRU, TransformerEncoder
import torch_geometric.nn as pyg_nn
# from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
#     return GCNConv(input_dim, out_dim)



class GIN_GRU(nn.Module):
    def __init__(self, node_features, hidden_dim, out_dim, num_layers):
        super(GIN_GRU, self).__init__()
        
#         self.recurrent_1 = GConvGRU(node_features, int(node_features/2), 3)
#         self.recurrent_2 = GConvGRU(int(node_features/2), int(node_features/4), 3)

#         self.recurrent_1 = LSTM(node_features, hidden_dim, 2)
#         self.recurrent_2 = LSTM(hidden_dim, 45, 1)

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
#         self.fc = nn.Sequential( nn.Linear(hidden_dim, hidden_dim), nn.ReLU() )
        self.bn_last = nn.Linear( 64, 1 )

        for i in range(num_layers-1):
            if i == 0:
                self.layers.append(make_gin_conv(node_features, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.layers.append(make_gin_conv(hidden_dim, int(hidden_dim/4)))
        self.batch_norms.append(nn.BatchNorm1d(int(hidden_dim/4)))
        
        self.recurrent = GRU(int(hidden_dim/4), out_dim, 4, dropout=0.1)
#         self.recurrent_2 = GRU(int(hidden_dim/2), out_dim, 1)

        project_dim = hidden_dim * num_layers
        self.project = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU())


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch 
#         print(x.shape)

        z = x
        zs = []
#         print(z, edge_index, batch)
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)  # , edge_weight
            z = F.relu(z)
            z = bn(z)
#             z = F.dropout(z, p=0.1, training=self.training)
            zs.append(z)
#         gs = [global_max_pool(z, batch) for z in zs]
#         g = torch.cat(gs, dim=1)
        z = global_max_pool(z, batch)        

        z = z.reshape(-1, 1, int(self.hidden_dim/4))
        z = torch.transpose(z,0,1)
      
        z = self.recurrent(z) 
        # print(x.shape)
#         z = F.relu(z)
#         x = F.dropout(x, training=self.training)
#         z = self.recurrent_2(z[0]) 
        # print(x.shape)
#         z = F.relu(z) 
        
        z = torch.transpose(z[0],0,1)
        z = z.reshape(-1,self.out_dim)
        
        
        
#         g = global_mean_pool(z, batch)
#         z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        
#         g = self.fc(g)
        g = self.project(z)
#         output = self.bn_last(torch.cat((g, p.reshape((len(p),1))),1))
        output = self.bn_last(g)
#         print(torch.sum(output))
        
        return output[:, 0]
    
