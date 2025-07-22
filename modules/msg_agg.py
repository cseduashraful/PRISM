"""
Message Aggregator Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter
from torch_scatter import scatter_max, scatter_softmax, scatter_sum, scatter_add


class LastAggregator(torch.nn.Module):
    def __init__(self, emb_dim: int = 100):
        super().__init__()
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator_old(torch.nn.Module):
    def __init__(self, emb_dim: int = 100):
        super().__init__()
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int, inverse_indices = None):
        # breakpoint()
        if inverse_indices is not None:
            return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")
        else:
            return scatter(msg[inverse_indices], index, dim=0, dim_size=dim_size, reduce="mean")

import mapped_scatter
class MeanAggregator(torch.nn.Module):
    def __init__(self, emb_dim: int = 100):
        super().__init__()

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int, inverse_indices: Tensor = None):
        if inverse_indices is None:
            return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")
        else:
            # msg[inverse_indices] is not materialized; we do the equivalent operation manually
            # breakpoint()
            b = torch.zeros(dim_size, msg.size(1), device=msg.device)

            # Call kernel (in-place update)
            mapped_scatter.scatter_add_mapped(msg, inverse_indices, index, b)
            ones = torch.ones(index.size(), dtype=msg.dtype, device=msg.device)
            count = scatter_sum(ones, index, 0, None, dim_size)
            # count[count < 1] = 
            b = b / count.unsqueeze(-1).clamp(min=1)
            # breakpoint()


            # msg_expanded_sum = scatter_add(msg, inverse_indices, dim=0, dim_size=index.size(0))  # [N, D]
            # count = torch.bincount(inverse_indices, minlength=index.size(0)).clamp(min=1).unsqueeze(-1)  # [N, 1]
            # msg_avg = msg_expanded_sum / count  # average over inverse_indices group

            # # Now scatter to final destinations using index
            # b = scatter_add(msg_avg, index, dim=0, dim_size=dim_size)
            # a = scatter(msg[inverse_indices], index, dim=0, dim_size=dim_size, reduce="mean")
            # if  torch.allclose(a, b, atol=1e-6, rtol=1e-5):
            #     print("okay")
            # else:
            #     breakpoint()

            return   b# [dim_size, D]



#aggr = self.aggr_module(msg_unique[inverse_indices], ux, b_t[inverse_indices], n_id.size(0))


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
        # Step 1: Compute attention 
        # print("Index: ", index)
        # breakpoint()
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




class TransformerAggregator(torch.nn.Module):
    def __init__(self, emb_dim: int = 100, nhead: int = 2, dim_feedforward: int = 100, num_layers: int = 1):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int) -> Tensor:
        if msg.size(0) == 0:
        # Return zero embeddings if no messages to process
            return msg.new_zeros((dim_size, msg.size(-1)))
        # 1. Count messages per index (since index is sorted)
        lengths = torch.bincount(index, minlength=dim_size)

        # 2. Pack msg into padded sequences
        max_len = lengths.max().item()
        padded = msg.new_zeros((dim_size, max_len, msg.size(-1)))
        mask = torch.ones((dim_size, max_len), dtype=torch.bool, device=msg.device)

        curr = 0
        for i in range(dim_size):
            l = lengths[i]
            if l > 0:
                padded[i, :l] = msg[curr:curr + l]
                mask[i, :l] = False  # Valid entries
                curr += l

        # 3. Apply transformer
        out = self.encoder(padded, src_key_padding_mask=mask)

        # 4. Pool final embeddings (e.g., take last non-masked)
        last_indices = lengths - 1
        final_out = out[torch.arange(dim_size), last_indices.clamp(min=0)]

        return final_out
