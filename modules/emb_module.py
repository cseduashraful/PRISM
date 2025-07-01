import math
from torch_geometric.nn import TransformerConv
import torch
import pdb

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        breakpoint()
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        # pdb.set_trace()
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TimeEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        class NormalLinear(torch.nn.Linear):
            def reset_parameters(self):
                stdv = 1.0 / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.out_channels)

    def forward(self, x, last_update, t):
        rel_t = last_update - t
        embeddings = x * (1 + self.embedding_layer(rel_t.to(x.dtype).unsqueeze(1)))

        return embeddings


# class JodieEmbedding(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, msg_dim, time_enc):
#         super().__init__()
#         self.time_enc = time_enc
#         edge_dim = msg_dim + time_enc.out_channels
#         self.conv = TransformerConv(
#             in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
#         )

#     def forward(self, x, last_update, edge_index, t, msg, root_ts):
#         rel_t = last_update[edge_index[0]] - t
#         rel_t_enc = self.time_enc(rel_t.to(x.dtype))
#         # pdb.set_trace()
#         edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
#         return self.conv(x, edge_index, edge_attr)

class IdentityEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()

    def forward(self, x, last_update, edge_index, t, msg):
        return x
    


# class JODIETimeEmbedding(torch.nn.Module):

#     def __init__(self, dim_out):
#         super(JODIETimeEmbedding, self).__init__()
#         self.dim_out = dim_out

#         class NormalLinear(torch.nn.Linear):
#         # From Jodie code
#             def reset_parameters(self):
#                 stdv = 1. / math.sqrt(self.weight.size(1))
#                 self.weight.data.normal_(0, stdv)
#                 if self.bias is not None:
#                     self.bias.data.normal_(0, stdv)

#         self.time_emb = NormalLinear(1, dim_out)
    
#     def forward(self, h, mem_ts, ts):
#         time_diff = (ts - mem_ts) / (ts + 1)
#         rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
#         return rst
            