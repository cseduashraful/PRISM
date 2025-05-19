import copy
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
from torch.nn import GRUCell, RNNCell, Linear

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter

from modules.time_enc import TimeEncoder

# import pdb
# import time
import mem_update_graph
from torch_scatter import scatter_max

TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]




class DAATGNMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "gru",
        layer: int = 3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.layer = layer

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        # self.gru = GRUCell(message_module.out_channels, memory_dim)
        if memory_updater_cell == "gru":  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError(
                "Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'."
            )

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer("last_update", last_update)
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, "reset_parameters"):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, "reset_parameters"):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()
        
    def mem_graph(self, ei_src, ei_dst, pos_node_s, pos_node_d):
        batch_size = pos_node_s.size(0)
        return mem_update_graph.mem_graph(
            ei_src.to(torch.int64).contiguous(),
            ei_dst.to(torch.int64).contiguous(),
            pos_node_s.to(torch.int64).contiguous(),
            pos_node_d.to(torch.int64).contiguous(),
            batch_size
        )

    def prep(self, ei_src, ei_dst, pos_node_s, pos_node_d):
        batch_size = pos_node_s.size(0)
        recent_indices = []
        # breakpoint()
        for i in range(ei_src.size(0)):
            target_node = ei_src[i].item()
            max_idx = ei_dst[i].item() % batch_size

            found_idx = -1
            for j in reversed(range(min(max_idx, pos_node_s.size(0)))):
                if pos_node_s[j].item() == target_node:
                    found_idx = j
                    break
                if pos_node_d[j].item() == target_node:
                    found_idx = j+batch_size
                    break
            if found_idx == -1:
                breakpoint()
            recent_indices.append(found_idx)

        return torch.tensor(recent_indices, device=ei_dst.device)

        # return None

    def forward(self, n_id, b_edge_index, b_t, b_raw_msg, b_isrc) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        memory, last_update = self._get_updated_memory(n_id)
            # return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        
        memory = memory[self._assoc[n_id]]
        for _ in range(self.layer-1):
            memory, last_update_n =  self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        return self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)

        # if self.training:
        #     memory, last_update = self._get_updated_memory(n_id)
        #     # return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        
        #     memory = memory[self._assoc[n_id]]
        #     for _ in range(self.layer-1):
        #         memory, last_update_n =  self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        #     return self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        # else:
        #     nn_id  = n_id.unique()
        #     self._assoc[nn_id] = torch.arange(nn_id.size(0), device=nn_id.device)

        #     memory, last_update = self.memory[nn_id], self.last_update[nn_id]

        #     memory = memory[self._assoc[n_id]]
        #     for _ in range(self.layer-1):
        #         memory, last_update_n =  self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        #     return self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)

            # return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)

        # return memory, last_update

    def update_state_v2(self, b_edge_index, ei_src, bs, src, pos_dst, t, msg, n_id, last_update, z ):
        used = ei_src.unique()
        all = b_edge_index[1, :].unique()
        not_used = all[~torch.isin(all, used)]
        is_src = not_used<bs
        not_used = not_used % bs

        s_store_indx = not_used[is_src]
        s_store_src = src[s_store_indx]
        s_store_dst = pos_dst[s_store_indx]
        s_store_t = t[s_store_indx]
        s_store_msg = msg[s_store_indx]

        d_store_indx = not_used[~is_src]
        d_store_src = pos_dst[d_store_indx]
        d_store_dst = src[d_store_indx]
        d_store_t = t[d_store_indx]
        d_store_msg = msg[d_store_indx]

        unique_nid, inverse = torch.unique(n_id, return_inverse=True)
        max_val, argmax_idx = scatter_max(last_update, inverse, dim=0)
        valid_mask = max_val > 0
        final_indices = argmax_idx[valid_mask]
        m_last_update = last_update[final_indices]
        m_nid = n_id[final_indices]
        m_memory = z[final_indices]

        self.memory[m_nid] = m_memory
        self.last_update[m_nid] = m_last_update

        
        self._update_msg_store(s_store_src, s_store_dst, s_store_t, s_store_msg, self.msg_s_store)
        self._update_msg_store(d_store_src, d_store_dst, d_store_t, d_store_msg, self.msg_d_store)

        # if not self.training:
        #     self._update_memory(n_id)





    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`."""
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _intra_batch_compute_msg(self, all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm, msg_module):
        msrc_s = b_edge_index[1][b_isrc]
        # breakpoint()
        src_s = all_n_id[msrc_s]
        dst_s = all_n_id[b_edge_index[0][b_isrc]]
        raw_msg_s = b_raw_msg[b_isrc]
        t_s = b_t[b_isrc]
        t_rel_s = t_s - last_update[self._assoc[src_s]]
        t_enc_s = self.time_enc(t_rel_s.to(raw_msg_s.dtype))
        msg_s = msg_module(bm[self._assoc[src_s]], bm[self._assoc[dst_s]], raw_msg_s, t_enc_s)

        return msg_s, t_s, src_s, dst_s, msrc_s
    
    def _intra_batch_compute_msg_v2(self, all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm, msg_module):
        msrc_s = b_edge_index[1][b_isrc]
        msrc_d = b_edge_index[0][b_isrc]
        src_s = all_n_id[msrc_s]
        dst_s = all_n_id[b_edge_index[0][b_isrc]]
        raw_msg_s = b_raw_msg[b_isrc]
        t_s = b_t[b_isrc]
        t_rel_s = t_s - last_update[self._assoc[src_s]]
        t_enc_s = self.time_enc(t_rel_s.to(raw_msg_s.dtype))
        msg_s = msg_module(bm[msrc_s], bm[msrc_d], raw_msg_s, t_enc_s)

        return msg_s, t_s, src_s, dst_s, msrc_s



    def _apply_intra_batch_info_v2(self, all_n_id, old_mem, last_update, b_edge_index, b_t, b_raw_msg, b_isrc):
        # breakpoint()
        # print(self._assoc[all_n_id])
        # print(all_n_id.max())
        # print(self._assoc[all_n_id].max())
        # print(bm.shape)

        # old_mem = bm[self._assoc[all_n_id]]

        # breakpoint()

        msg_s, t_s, src_s, dst_s, msrc_s  = self._intra_batch_compute_msg_v2(all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, old_mem, self.msg_s_module)
        msg_d, t_d, src_d, dst_d, msrc_d  = self._intra_batch_compute_msg_v2(all_n_id, b_edge_index, ~b_isrc, b_raw_msg, b_t, last_update, old_mem, self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([msrc_s, msrc_d], dim=0).long()
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        # breakpoint()
        aggr = self.aggr_module(msg, idx, t, all_n_id.size(0))
        # breakpoint()

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, old_mem)
        dim_size = memory.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")
        # breakpoint()

        # # Get local copy of updated `last_update`.
        # dim_size = self.last_update.size(0)
        # last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]
        # breakpoint()
        return memory, last_update


    def _apply_intra_batch_info(self, all_n_id, bm, last_update, b_edge_index, b_t, b_raw_msg, b_isrc):
        # breakpoint()
        # print(self._assoc[all_n_id])
        # print(all_n_id.max())
        # print(self._assoc[all_n_id].max())
        # print(bm.shape)

        old_mem = bm[self._assoc[all_n_id]]

        # breakpoint()

        msg_s, t_s, src_s, dst_s, msrc_s  = self._intra_batch_compute_msg(all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm, self.msg_s_module)
        msg_d, t_d, src_d, dst_d, msrc_d  = self._intra_batch_compute_msg(all_n_id, b_edge_index, ~b_isrc, b_raw_msg, b_t, last_update, bm, self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([msrc_s, msrc_d], dim=0).long()
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        # breakpoint()
        aggr = self.aggr_module(msg, idx, t, all_n_id.size(0))
        # breakpoint()

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, old_mem)
        dim_size = memory.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")
        # breakpoint()

        # # Get local copy of updated `last_update`.
        # dim_size = self.last_update.size(0)
        # last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]
        # breakpoint()
        return memory, last_update





    def _get_updated_memory(self, all_n_id: Tensor) -> Tuple[Tensor, Tensor]:
        n_id  = all_n_id.unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module
        )

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]

        return memory, last_update

    def _update_msg_store(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
        msg_store: TGNMessageStoreType,
    ):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])
        # breakpoint()

    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
    ):
        data = [msg_store[i] for i in n_id.tolist()]
        # breakpoint()
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        # breakpoint()
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            # breakpoint()
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)


















































































class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "gru",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        # self.gru = GRUCell(message_module.out_channels, memory_dim)
        if memory_updater_cell == "gru":  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError(
                "Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'."
            )

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer("last_update", last_update)
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, "reset_parameters"):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, "reset_parameters"):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`."""
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module
        )

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]

        return memory, last_update

    def _update_msg_store(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
        msg_store: TGNMessageStoreType,
    ):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
    ):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)


class DyRepMemory(torch.nn.Module):
    r"""
    Based on intuitions from TGN Memory...
    Differences with the original TGN Memory:
        - can use source or destination embeddings in message generation
        - can use a RNN or GRU module as the memory updater

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
        memory_updater_type (str): specifies whether the memory updater is GRU or RNN
        use_src_emb_in_msg (bool): whether to use the source embeddings 
            in generation of messages
        use_dst_emb_in_msg (bool): whether to use the destination embeddings 
            in generation of messages
    """
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable, memory_updater_type: str,
                 use_src_emb_in_msg: bool = False, use_dst_emb_in_msg: bool = False):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)

        assert memory_updater_type in ['gru', 'rnn'], "Memor updater can be either `rnn` or `gru`."
        if memory_updater_type == 'gru':  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_type == 'rnn':  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError("Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'.")
        
        self.use_src_emb_in_msg = use_src_emb_in_msg
        self.use_dst_emb_in_msg = use_dst_emb_in_msg

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer('last_update', last_update)
        self.register_buffer('_assoc', torch.empty(num_nodes,
                                                   dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, 
                     embeddings: Tensor = None, assoc: Tensor = None):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`."""
        n_id = torch.cat([src, dst]).unique()
        
        if self.training:
            self._update_memory(n_id, embeddings, assoc)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id, embeddings, assoc)

    def _reset_message_store(self):
        i = self.memory.new_empty((0, ), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor, embeddings: Tensor = None, assoc: Tensor = None):
        memory, last_update = self._get_updated_memory(n_id, embeddings, assoc)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor, embeddings: Tensor = None, assoc: Tensor = None) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store,
                                                     self.msg_s_module, embeddings, assoc)                                          

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store,
                                                     self.msg_d_module, embeddings, assoc)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce='max')[n_id]

        return memory, last_update

    def _update_msg_store(self, src: Tensor, dst: Tensor, t: Tensor,
                          raw_msg: Tensor, msg_store: TGNMessageStoreType):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable, 
                     embeddings: Tensor = None, assoc: Tensor = None):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        # source nodes: retrieve embeddings
        source_memory = self.memory[src]
        if self.use_src_emb_in_msg and embeddings != None:
            if src.size(0) > 0:
                curr_src, curr_src_idx = [], []
                for s_idx, s in enumerate(src):
                    if s in n_id:
                        curr_src.append(s.item())
                        curr_src_idx.append(s_idx)

                source_memory[curr_src_idx] = embeddings[assoc[curr_src]]

        # destination nodes: retrieve embeddings
        destination_memory = self.memory[dst]
        if self.use_dst_emb_in_msg and embeddings != None:
            if dst.size(0) > 0:
                curr_dst, curr_dst_idx = [], []
                for d_idx, d in enumerate(dst):
                    if d in n_id:
                        curr_dst.append(d.item())
                        curr_dst_idx.append(d_idx)
                destination_memory[curr_dst_idx] = embeddings[assoc[curr_dst]]
            
        msg = msg_module(source_memory, destination_memory, raw_msg, t_enc)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(
                torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)





class APANEncoder(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h_prev: Tensor, mailbox: Tensor) -> Tensor:
        query = h_prev.unsqueeze(1)
        key_value = mailbox
        attn_out, _ = self.attn(query, key_value, key_value)
        attn_out = attn_out.squeeze(1)
        out = self.norm1(attn_out + h_prev)
        out = self.norm2(self.feedforward(out) + out)
        return out

class APANMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "gru",
        max_mailbox_size: int = 10,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.max_mailbox_size = max_mailbox_size

        self.msg_module = message_module
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)

        if memory_updater_cell == "gru":
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError("Undefined memory updater. Use 'gru' or 'rnn'.")

        self.mail_proj = nn.Linear(memory_dim * 2 + raw_msg_dim, raw_msg_dim)
        self.position_emb = nn.Embedding(max_mailbox_size, raw_msg_dim)
        self.encoder = APANEncoder(raw_msg_dim, num_heads)

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes, dtype=torch.long))
        self._reset_message_store()

    @property
    def device(self):
        return self.memory.device

    def reset_parameters(self):
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.mail_proj.reset_parameters()
        self.position_emb.reset_parameters()
        self.reset_state()

    def reset_state(self):
        self.memory.zero_()
        self.last_update.zero_()
        self._reset_message_store()

    def _reset_message_store(self):
        self.mailbox = {i: deque(maxlen=self.max_mailbox_size) for i in range(self.num_nodes)}

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        memory = self.memory[n_id]
        last_update = self.last_update[n_id]
        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        n_id = dst.unique()
        self._update_memory(n_id)
        self._propagate_mail(src, dst, raw_msg)

    def _update_memory(self, n_id: Tensor):
        mailbox_tensor = self.get_mailbox_tensor(n_id)
        h_prev = self.memory[n_id]
        h_new = self.encoder(h_prev, mailbox_tensor)
        self.memory[n_id] = h_new
        self.last_update[n_id] = torch.max(self.last_update[n_id], torch.tensor(0).to(self.device))

    def _propagate_mail(self, src: Tensor, dst: Tensor, raw_msg: Tensor):
        for s, d, e_feat in zip(src.tolist(), dst.tolist(), raw_msg):
            z_src = self.memory[s]
            z_dst = self.memory[d]
            mail_input = torch.cat([z_src, e_feat, z_dst]).unsqueeze(0)
            mail = self.mail_proj(mail_input).squeeze(0).detach().clone()
            self.mailbox[d].append(mail)

    def get_mailbox_tensor(self, n_id: Tensor) -> Tensor:
        B, D, K = len(n_id), self.raw_msg_dim, self.max_mailbox_size
        out = self.memory.new_zeros(B, K, D)

        for i, nid in enumerate(n_id.tolist()):
            mails = list(self.mailbox[nid])
            for j, msg in enumerate(mails):
                if j >= K: break
                out[i, j] = msg + self.position_emb(torch.tensor(j, device=self.device))
        return out