"""
Message Aggregator Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter
from torch_scatter import scatter_max, scatter_softmax, scatter_sum


class LastAggregator(torch.nn.Module):
    def __init__(self, emb_dim: int = 100):
        super().__init__()
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def __init__(self, emb_dim: int = 100):
        super().__init__()
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        # breakpoint()
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")
    

# import torch
# from torch import Tensor
# from torch_scatter import scatter_softmax, scatter_sum

class AttentionAggregator(torch.nn.Module):
    """
    Aggregates messages using learned attention over incoming messages per node.
    """
    def __init__(self, emb_dim: int = 100):
        super().__init__()
        self.att_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 1),
        )
        # self.att_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(emb_dim, emb_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(emb_dim, 1),
        # )
        # self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int) -> Tensor:
        """
        Args:
            msg: [num_messages, emb_dim] - messages to aggregate
            index: [num_messages] - node indices for scatter
            t: [num_messages] - timestamps (currently not used, but can be included)
            dim_size: number of nodes

        Returns:
            Aggregated node embeddings [dim_size, emb_dim]
        """
        # Step 1: Compute attention scores
        att_score = self.att_mlp(msg).squeeze(-1)  # [num_messages]

        # Step 2: Normalize attention scores over neighbors
        att_weight = scatter_softmax(att_score, index, dim=0)

        # Step 3: Weighted sum aggregation
        msg_weighted = msg * att_weight.unsqueeze(-1)  # [num_messages, emb_dim]
        out = scatter_sum(msg_weighted, index, dim=0, dim_size=dim_size)

        # att_score = self.att_mlp(msg).squeeze(-1) / self.temperature.clamp(min=1e-6)
        # att_weight = scatter_softmax(att_score, index, dim=0)
        # msg_weighted = msg * att_weight.unsqueeze(-1)
        # out = scatter_sum(msg_weighted, index, dim=0, dim_size=dim_size)
        return out


class AttentionAggregator_v2(torch.nn.Module):
    """
    Aggregates messages using learned attention over incoming messages per node.
    """
    def __init__(self, emb_dim: int = 100):
        super().__init__()
        # self.att_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(emb_dim, 1),
        # )
        self.att_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, 1),
        )
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int) -> Tensor:
        """
        Args:
            msg: [num_messages, emb_dim] - messages to aggregate
            index: [num_messages] - node indices for scatter
            t: [num_messages] - timestamps (currently not used, but can be included)
            dim_size: number of nodes

        Returns:
            Aggregated node embeddings [dim_size, emb_dim]
        """
        # # Step 1: Compute attention scores
        # att_score = self.att_mlp(msg).squeeze(-1)  # [num_messages]

        # # Step 2: Normalize attention scores over neighbors
        # att_weight = scatter_softmax(att_score, index, dim=0)

        # # Step 3: Weighted sum aggregation
        # msg_weighted = msg * att_weight.unsqueeze(-1)  # [num_messages, emb_dim]
        # out = scatter_sum(msg_weighted, index, dim=0, dim_size=dim_size)

        att_score = self.att_mlp(msg).squeeze(-1) / self.temperature.clamp(min=1e-6)
        att_weight = scatter_softmax(att_score, index, dim=0)
        msg_weighted = msg * att_weight.unsqueeze(-1)
        out = scatter_sum(msg_weighted, index, dim=0, dim_size=dim_size)
        return out

        # return out




