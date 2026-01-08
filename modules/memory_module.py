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
from torch_scatter import scatter_add

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
        # breakpoint()
        return mem_update_graph.mem_graph(
            ei_src.to(torch.int64).contiguous(),
            ei_dst.to(torch.int64).contiguous(),
            pos_node_s.to(torch.int64).contiguous(),
            pos_node_d.to(torch.int64).contiguous(),
            batch_size
        )

    # def prep(self, ei_src, ei_dst, pos_node_s, pos_node_d):
    #     batch_size = pos_node_s.size(0)
    #     recent_indices = []
    #     # breakpoint()
    #     for i in range(ei_src.size(0)):
    #         target_node = ei_src[i].item()
    #         max_idx = ei_dst[i].item() % batch_size

    #         found_idx = -1
    #         for j in reversed(range(min(max_idx, pos_node_s.size(0)))):
    #             if pos_node_s[j].item() == target_node:
    #                 found_idx = j
    #                 break
    #             if pos_node_d[j].item() == target_node:
    #                 found_idx = j+batch_size
    #                 break
    #         if found_idx == -1:
    #             breakpoint()
    #         recent_indices.append(found_idx)

    #     return torch.tensor(recent_indices, device=ei_dst.device)

        # return None

    def forward(self, n_id, b_edge_index, b_t, b_raw_msg, b_isrc, delivery_addr = None) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        memory, last_update = self._get_updated_memory(n_id)
            # return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        # breakpoint()
        memory = memory[self._assoc[n_id]]
        init_mem = None # memory
        for _ in range(self.layer-1):
            memory, last_update_n =  self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc, init_mem = init_mem, delivery_addr = delivery_addr)
        return self._apply_intra_batch_info_v2(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc, init_mem = init_mem, delivery_addr = delivery_addr)

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

    def update_state_v2(self, all_nodes, ei_src, bs, src, pos_dst, t, msg, n_id, last_update, z ):
        used = ei_src.unique()
        all = all_nodes.unique()
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





    # def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
    #     """Updates the memory with newly encountered interactions
    #     :obj:`(src, dst, t, raw_msg)`."""
    #     n_id = torch.cat([src, dst]).unique()

    #     if self.training:
    #         self._update_memory(n_id)
    #         self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
    #         self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
    #     else:
    #         self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
    #         self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
    #         self._update_memory(n_id)

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

    # def _intra_batch_compute_msg(self, all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm, msg_module):
    #     msrc_s = b_edge_index[1][b_isrc]
    #     # breakpoint()
    #     src_s = all_n_id[msrc_s]
    #     dst_s = all_n_id[b_edge_index[0][b_isrc]]
    #     raw_msg_s = b_raw_msg[b_isrc]
    #     t_s = b_t[b_isrc]
    #     t_rel_s = t_s - last_update[self._assoc[src_s]]
    #     t_enc_s = self.time_enc(t_rel_s.to(raw_msg_s.dtype))
    #     msg_s = msg_module(bm[self._assoc[src_s]], bm[self._assoc[dst_s]], raw_msg_s, t_enc_s)

    #     return msg_s, t_s, src_s, dst_s, msrc_s
    
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



    def _apply_intra_batch_info_v2(self, all_n_id, old_mem, last_update, b_edge_index, b_t, b_raw_msg, b_isrc, init_mem = None, delivery_addr = None):
        msg_s, t_s, src_s, dst_s, msrc_s  = self._intra_batch_compute_msg_v2(all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, old_mem, self.msg_s_module)
        msg_d, t_d, src_d, dst_d, msrc_d  = self._intra_batch_compute_msg_v2(all_n_id, b_edge_index, ~b_isrc, b_raw_msg, b_t, last_update, old_mem, self.msg_d_module)

        # Aggregate messages.
        if delivery_addr is not None:
            msrc_s = delivery_addr[b_isrc]
            msrc_d = delivery_addr[~b_isrc]
        
        idx = torch.cat([msrc_s, msrc_d], dim=0).long()
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        # breakpoint()
        aggr = self.aggr_module(msg, idx, t, all_n_id.size(0))
        # breakpoint()

        # Get local copy of updated memory.
        if init_mem is None:
            memory = self.memory_updater(aggr, old_mem)
        else:
            memory = self.memory_updater(aggr, init_mem)
        dim_size = memory.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")
        # breakpoint()
        return memory, last_update


    # def _apply_intra_batch_info(self, all_n_id, bm, last_update, b_edge_index, b_t, b_raw_msg, b_isrc):
    #     # breakpoint()
    #     # print(self._assoc[all_n_id])
    #     # print(all_n_id.max())
    #     # print(self._assoc[all_n_id].max())
    #     # print(bm.shape)

    #     old_mem = bm[self._assoc[all_n_id]]

    #     # breakpoint()

    #     msg_s, t_s, src_s, dst_s, msrc_s  = self._intra_batch_compute_msg(all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm, self.msg_s_module)
    #     msg_d, t_d, src_d, dst_d, msrc_d  = self._intra_batch_compute_msg(all_n_id, b_edge_index, ~b_isrc, b_raw_msg, b_t, last_update, bm, self.msg_d_module)

    #     # Aggregate messages.
    #     idx = torch.cat([msrc_s, msrc_d], dim=0).long()
    #     msg = torch.cat([msg_s, msg_d], dim=0)
    #     t = torch.cat([t_s, t_d], dim=0)
    #     # breakpoint()
    #     aggr = self.aggr_module(msg, idx, t, all_n_id.size(0))
    #     # breakpoint()

    #     # Get local copy of updated memory.
    #     memory = self.memory_updater(aggr, old_mem)
    #     dim_size = memory.size(0)
    #     last_update = scatter(t, idx, 0, dim_size, reduce="max")
    #     # breakpoint()

    #     # # Get local copy of updated `last_update`.
    #     # dim_size = self.last_update.size(0)
    #     # last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]
    #     # breakpoint()
    #     return memory, last_update





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
        # breakpoint()
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
        # breakpoint()
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])
        # for node, s, d, ts, msg in zip(src.tolist(), src, dst, t, raw_msg):
        #     msg_store[node] = (s, d, ts, msg)

        # # breakpoint()
        # # return {
        # #     src[i].item(): (src[i], dst[i], t[i], raw_msg[i])
        # #     for i in range(src.size(0))
        # # }
        # src_cpu = src.cpu()  # avoid repeated .item() GPU→CPU syncs
        # for i in range(src.size(0)):
        #     node_id = src_cpu[i].item()
        #     msg_store[node_id] = (src[i], dst[i], t[i], raw_msg[i])

    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
    ):
        data = [msg_store[i] for i in n_id.tolist()]
        # breakpoint()
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0).to(self.last_update[src].device)
        raw_msg = torch.cat(raw_msg, dim=0).to(self.last_update[src].device)
        # breakpoint()
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        # breakpoint()
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src.to(t.device), dst

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




# class APANMemory_old(torch.nn.Module):
#     def __init__(
#         self,
#         num_nodes: int,
#         raw_msg_dim: int,
#         memory_dim: int,
#         time_dim: int,
#         message_module: Callable,
#         aggregator_module: Callable,
#         memory_updater_cell: str = "gru",
#         mailbox_size: int = 10,
#         # get_neighbors: Optional[Callable[[Tensor], Dict[int, List[int]]]] = None,
#     ):
#         super().__init__()

#         self.num_nodes = num_nodes
#         self.raw_msg_dim = raw_msg_dim
#         self.memory_dim = memory_dim
#         self.time_dim = time_dim
#         self.mailbox_size = mailbox_size
#         # self.get_neighbors = get_neighbors

#         self.msg_module = message_module
#         self.aggr_module = aggregator_module
#         self.time_enc = TimeEncoder(time_dim)

#         if memory_updater_cell == "gru":
#             self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
#         elif memory_updater_cell == "rnn":
#             self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
#         else:
#             raise ValueError("Invalid memory updater type.")

#         self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
#         self.register_buffer("last_update", torch.empty(num_nodes, dtype=torch.long))
#         self.register_buffer('_assoc', torch.empty(num_nodes,
#                                                    dtype=torch.long))


#         # Message store: same shape and logic as TGN
#         i = torch.empty((0,), dtype=torch.long)
#         msg = torch.empty((0, raw_msg_dim))
#         self.msg_store = {j: (i, i, i, msg, i) for j in range(num_nodes)}

#         # self._assoc = torch.empty(num_nodes, dtype=torch.long)

#         self.reset_parameters()

#     def reset_parameters(self):
#         if hasattr(self.msg_module, "reset_parameters"):
#             self.msg_module.reset_parameters()
#         if hasattr(self.aggr_module, "reset_parameters"):
#             self.aggr_module.reset_parameters()
#         self.time_enc.reset_parameters()
#         self.memory_updater.reset_parameters()
#         self.reset_state()

#     def reset_state(self):
#         zeros(self.memory)
#         zeros(self.last_update)
#         self._reset_message_store()

#     def detach(self):
#         """Detaches the memory from gradient computation."""
#         self.memory.detach_()
        
#     def _reset_message_store(self):
#         i = self.memory.new_empty((0,), dtype=torch.long)
#         msg = self.memory.new_empty((0, self.raw_msg_dim))
#         self.msg_store = {j: (i, i, i, msg, i) for j in range(self.num_nodes)}

#     def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
#         # breakpoint()
#         memory, last_update = self._get_updated_memory(n_id)
#         # if self.training:
#         #     memory, last_update = self._get_updated_memory(n_id)
#         # else:
#         #     memory, last_update = self.memory[n_id], self.last_update[n_id]
#         return memory, last_update

#     def fill_store(self, src, dst, t, raw_msg, all_neighbors, msg_store):
#         for i in range(len(src)):
#             ux, v = src[i].item(), dst[i].item()
#             timestamp = t[i].unsqueeze(0)
#             msg = raw_msg[i].unsqueeze(0)

#             neighbors = all_neighbors[ux]#self.get_neighbors(u, v, n_id, edge_index)
#             for nbr in neighbors:
#                 if nbr == ux:
#                     continue
#                 old_src, old_dst, old_t, old_msg, old_dla = msg_store[nbr]
#                 new_src = torch.cat([old_src, torch.tensor([ux], device=src.device)])
#                 new_dst = torch.cat([old_dst, torch.tensor([v], device=src.device)])
#                 new_t = torch.cat([old_t, timestamp])
#                 new_msg = torch.cat([old_msg, msg])
#                 new_dla = torch.cat([old_dla, torch.tensor([nbr], device=src.device)])

#                 # Keep only the last `mailbox_size` messages
#                 if new_msg.size(0) > self.mailbox_size:
#                     new_src = new_src[-self.mailbox_size:]
#                     new_dst = new_dst[-self.mailbox_size:]
#                     new_t = new_t[-self.mailbox_size:]
#                     new_msg = new_msg[-self.mailbox_size:]
#                     new_dla = new_dla[-self.mailbox_size:]

#                 msg_store[nbr] = (new_src, new_dst, new_t, new_msg, new_dla)



#     def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, all_neighbors, n_ids):
#         # model['memory'].update_state(src, pos_dst, t, msg, neighbors)
#         # if self.get_neighbors is None:
#         #     raise ValueError("get_neighbors function must be provided.")
#         # all_neighbors = self.get_neighbors(u, v, n_id, edge_index)
#         # breakpoint()
#         self._update_memory(n_ids)
#         self.fill_store(src, dst, t, raw_msg, all_neighbors, self.msg_store)
#         self.fill_store(dst, src, t, raw_msg, all_neighbors, self.msg_store)
#         # breakpoint()

#         # for i in range(len(src)):
#         #     ux, v = src[i].item(), dst[i].item()
#         #     timestamp = t[i].unsqueeze(0)
#         #     msg = raw_msg[i].unsqueeze(0)

#         #     neighbors = all_neighbors[ux]#self.get_neighbors(u, v, n_id, edge_index)
#         #     for nbr in neighbors:
#         #         if nbr == ux:
#         #             continue
#         #         old_src, old_dst, old_t, old_msg = self.msg_store[nbr]
#         #         new_src = torch.cat([old_src, torch.tensor([ux], device=src.device)])
#         #         new_dst = torch.cat([old_dst, torch.tensor([v], device=src.device)])
#         #         new_t = torch.cat([old_t, timestamp])
#         #         new_msg = torch.cat([old_msg, msg])

#         #         # Keep only the last `mailbox_size` messages
#         #         if new_msg.size(0) > self.mailbox_size:
#         #             new_src = new_src[-self.mailbox_size:]
#         #             new_dst = new_dst[-self.mailbox_size:]
#         #             new_t = new_t[-self.mailbox_size:]
#         #             new_msg = new_msg[-self.mailbox_size:]

#         #         self.msg_store[nbr] = (new_src, new_dst, new_t, new_msg)
#         # print(src)
#         # breakpoint()
#         # print(dst)
#         # n_id = torch.tensor([i for i, (s, _, _, _) in self.msg_store.items() if s.numel() > 0], device=src.device)
#         # if len(n_id) > 0:
#         #     self._update_memory(n_id)

#     def _update_memory(self, n_id: Tensor):
#         memory, last_update = self._get_updated_memory(n_id)
#         self.memory[n_id] = memory
#         self.last_update[n_id] = last_update
#         for i in n_id.tolist():
#             self.msg_store[i] = tuple(tensor[:0] for tensor in self.msg_store[i])

#     def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
#         # breakpoint()
#         self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
#         msg, t, src, dst = self._compute_msg(n_id)

#         # print("n_id: ", n_id)
#         # print("src: ", src)

#         aggr = self.aggr_module(msg, self._assoc[src], t, n_id.size(0))
#         updated_memory = self.memory_updater(aggr, self.memory[n_id])
#         last_update = scatter(t, src, 0, self.last_update.size(0), reduce="max")[n_id]
#         # breakpoint()
#         return updated_memory, last_update

#     def _compute_msg(self, n_id: Tensor):
#         # breakpoint()
#         data = [self.msg_store[i] for i in n_id.tolist()]
#         # breakpoint()
#         src, dst, t, raw_msg, ux = list(zip(*data))
#         src = torch.cat(src, dim=0)
#         dst = torch.cat(dst, dim=0)
#         ux = torch.cat(ux, dim=0)
#         t = torch.cat(t, dim=0)
#         raw_msg = torch.cat(raw_msg, dim=0)
#         t_rel = t - self.last_update[ux]
#         t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
#         msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
#         return msg, t, ux, dst
    
#     def train(self, mode: bool = True):
#         """Sets the module in training mode."""
#         if self.training and not mode:
#             # Flush message store to memory in case we just entered eval mode.
#             self._update_memory(
#                 torch.arange(self.num_nodes, device=self.memory.device))
#             self._reset_message_store()
#         super().train(mode)


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
        mailbox_size: int = 10,
        num_head: int = 2,
        data = None,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.mailbox_size = mailbox_size
        self.data = data

        self.msg_module = message_module
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)

        # self.msg_s_store = {}
        # self.msg_d_store = {}
        self.m_counts = torch.zeros(num_nodes, dtype=torch.long, device='cuda:0')
        self.m_store = torch.full((num_nodes, self.mailbox_size), -1, dtype=torch.long, device='cuda:0')
        self.m_store_dir = torch.full((num_nodes, self.mailbox_size), False, dtype=torch.bool, device='cuda:0')


        if memory_updater_cell == "gru":
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "transformer":
            self.memory_updater = TransformerMemoryUpdater(
                input_dim=message_module.out_channels,
                memory_dim=memory_dim,
                nhead=num_head,           # Optional: make configurable
                num_layers=1
            )
        else:
            raise ValueError("Invalid memory updater type.")

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.reset_parameters()

    def _init_message_store(self):
        device = self.memory.device
        self.msg_src = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        self.msg_dst = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        self.msg_t = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        self.msg_dla = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        self.msg_raw = torch.zeros((self.num_nodes, self.mailbox_size, self.raw_msg_dim), device=device)
        self.msg_counts = torch.zeros(self.num_nodes, dtype=torch.long, device=device)

    def reset_parameters(self):
        if hasattr(self.msg_module, "reset_parameters"):
            self.msg_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)
        self._init_message_store()

    def detach(self):
        self.memory.detach_()

    def _reset_message_store(self):
        self.msg_src.fill_(-1)
        self.msg_dst.fill_(-1)
        self.msg_t.fill_(-1)
        self.msg_dla.fill_(-1)
        self.msg_raw.zero_()
        self.msg_counts.zero_()

    def forward(self, n_id: Tensor, data=None) -> Tuple[Tensor, Tensor]:
        memory, last_update = self._get_updated_memory(n_id, data =  data)
        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, all_neighbors, n_ids, npadded = None, eid_start = 0, assoc = None, data = None):
        self._update_memory(n_ids, data = data)
        self.m_counts[n_ids] = 0
        self._fill_vector_store(src, dst, eid_start, npadded, assoc)
        # self._fill_tensor_store_old(src, dst, t, raw_msg, all_neighbors)
        # self._fill_tensor_store_old(dst, src, t, raw_msg, all_neighbors)


    def _fill_vector_store(self, src, dst, eid_start, npadded, assoc):
        # breakpoint()
        msg_id = torch.arange(eid_start, eid_start+src.size(0), device=src.device)
        # self.msg_s_store[]
        npadded_idx = assoc[src]
        npadded_idx_dst =  assoc[dst]
        src_npadded = npadded[npadded_idx]
        dst_npadded = npadded[npadded_idx_dst]

        msg_id_expanded = msg_id.unsqueeze(1).expand_as(src_npadded)  # (N, K)
        flat_nodes = src_npadded.flatten()
        flat_msg_ids = msg_id_expanded.flatten()
        msg_id_expanded_dst = msg_id.unsqueeze(1).expand_as(dst_npadded)  # (N, K)
        flat_nodes_dst = dst_npadded.flatten()
        flat_msg_ids_dst = msg_id_expanded.flatten()

        # Remove -1 entries
        valid = flat_nodes != -1
        flat_nodes = flat_nodes[valid]       # (M,)
        flat_msg_ids = flat_msg_ids[valid]   # (M,)
        valid_dst = flat_nodes_dst != -1
        flat_nodes_dst = flat_nodes_dst[valid_dst]       # (M,)
        flat_msg_ids_dst = flat_msg_ids_dst[valid_dst]   # (M,)



        # Sort by flat_nodes so we can group
        sorted_nodes, sort_idx = torch.sort(flat_nodes)
        sorted_msg_ids = flat_msg_ids[sort_idx]
        # breakpoint()

        pairs = torch.stack([flat_nodes, flat_msg_ids], dim=1)  # shape: (N, 2)
        new_pairs = torch.stack([dst, msg_id], dim=1) 
        pairs = torch.cat([pairs, new_pairs], dim=0)
        pairs = torch.unique(pairs, dim=0)

        pairs_dst = torch.stack([flat_nodes_dst, flat_msg_ids_dst], dim=1)  # shape: (N, 2)
        new_pairs_dst = torch.stack([src, msg_id], dim=1) 
        pairs_dst = torch.cat([pairs_dst, new_pairs_dst], dim=0)
        pairs_dst = torch.unique(pairs_dst, dim=0)
        # breakpoint()

        # # Group by node: find boundaries
        # unique_nodes, counts = torch.unique_consecutive(sorted_nodes, return_counts=True)
        # msg_store_keys = unique_nodes
        # msg_store_ptr = torch.cat([torch.tensor([0], device=counts.device), counts.cumsum(0)])  # (num_keys+1,)
        # msg_store_values = sorted_msg_ids  # flat tensor of all message IDs
        # breakpoint()
        sdir = torch.full_like(pairs[:,1], True, dtype=torch.bool)
        ddir = torch.full_like(pairs_dst[:,1], False, dtype=torch.bool)

        combined_pairs = torch.cat([pairs, pairs_dst], dim=0)
        combined_dirs = torch.cat([sdir, ddir], dim=0)
        
        sorted_vals, sorted_idx = torch.sort(combined_pairs[:, 0] * (combined_pairs[:, 0].max() + 1) + combined_pairs[:, 1])
        pairs = combined_pairs[sorted_idx]
        dirs = combined_dirs[sorted_idx]
        # breakpoint()


        node_ids = pairs[:, 0]
        msg_ids = pairs[:, 1]

        # Step 1: Sort by node_id to group duplicates
        sorted_vals, sorted_idx = torch.sort(node_ids, stable=True)
        sorted_nodes = node_ids[sorted_idx]
        sorted_msgs = msg_ids[sorted_idx]
        sorted_dirs = dirs[sorted_idx]

        # Step 2: Get per-node insertion offsets
        # inverse_idx gives the group assignment; counts how many per node
        unique_nodes, inverse_idx, counts = torch.unique_consecutive(sorted_nodes, return_inverse=True, return_counts=True)

        # How many previous messages each instance has seen within its group
        intra_offsets = torch.arange(len(sorted_nodes), device=pairs.device) - torch.cumsum(
            torch.bincount(inverse_idx, minlength=unique_nodes.size(0)), dim=0
        ).repeat_interleave(counts) + counts.repeat_interleave(counts)

        # Step 3: Compute true write positions in circular buffer
        # Each node’s global counter (before update)
        global_start = self.m_counts[sorted_nodes]
        write_idx = (global_start + intra_offsets) % self.mailbox_size

        # Step 4: Insert messages into store
        self.m_store[sorted_nodes, write_idx] = sorted_msgs
        self.m_store_dir[sorted_nodes, write_idx] = sorted_dirs#torch.full_like(sorted_msgs, isSrc, dtype=torch.bool)
        # breakpoint()

        # Step 5: Update msg_counts per node (add how many were added)
        # breakpoint()
        self.m_counts.index_add_(0, unique_nodes, counts)
        # breakpoint()



    def _fill_tensor_store_new(self, src, dst, t, raw_msg, all_neighbors):
        device = src.device
        B = src.size(0)

        neighbors_list = []
        repeat_index = []

        for i in range(B):
            u = src[i].item()
            if u >= len(all_neighbors):
                continue

            neighbors_u = all_neighbors[u]
            if not isinstance(neighbors_u, torch.Tensor):
                neighbors_u = torch.tensor(neighbors_u, device=device, dtype=torch.long)

            # skip self-loop
            neighbors_u = neighbors_u[neighbors_u != u]
            if neighbors_u.numel() == 0:
                continue

            neighbors_list.append(neighbors_u)
            repeat_index.append(torch.full((neighbors_u.numel(),), i, device=device, dtype=torch.long))

        if not neighbors_list:
            return  # nothing to write

        all_nbrs = torch.cat(neighbors_list)  # All neighbors getting messages
        idx = torch.cat(repeat_index)         # Source index for each neighbor

        # Compute message data
        src_rep = src[idx]
        dst_rep = dst[idx]
        t_rep = t[idx]
        raw_msg_rep = raw_msg[idx]

        # Step 1: get current counts
        counts = self.msg_counts[all_nbrs]

        # Step 2: compute write index (mailbox position)
        write_idx = counts % self.mailbox_size

        # Step 3: write to mailbox
        self.msg_src[all_nbrs, write_idx] = src_rep
        self.msg_dst[all_nbrs, write_idx] = dst_rep
        self.msg_t[all_nbrs, write_idx] = t_rep
        self.msg_raw[all_nbrs, write_idx] = raw_msg_rep
        self.msg_dla[all_nbrs, write_idx] = all_nbrs

        # Step 4: simulate atomic add via scatter_add
        one_tensor = torch.ones_like(all_nbrs)
        scatter_add(src=one_tensor, index=all_nbrs, out=self.msg_counts)


    def _fill_tensor_store(self, src, dst, t, raw_msg, all_neighbors):
        device = src.device
        B = src.size(0)

        neighbors_list = []
        repeat_index = []

        for i in range(B):
            u = src[i].item()
            if u >= len(all_neighbors):  # safeguard
                continue

            neighbors_u = all_neighbors[u]
            if not isinstance(neighbors_u, torch.Tensor):
                neighbors_u = torch.tensor(neighbors_u, device=device, dtype=torch.long)

            # skip if empty
            if neighbors_u.numel() == 0:
                continue

            # remove self-loop
            neighbors_u = neighbors_u[neighbors_u != u]
            if neighbors_u.numel() == 0:
                continue

            neighbors_list.append(neighbors_u)
            repeat_index.append(torch.full((neighbors_u.numel(),), i, device=device, dtype=torch.long))

        if not neighbors_list:
            return  # nothing to do

        all_nbrs = torch.cat(neighbors_list)
        idx = torch.cat(repeat_index)

        src_rep = src[idx]
        dst_rep = dst[idx]
        t_rep = t[idx]
        raw_msg_rep = raw_msg[idx]

        counts = self.msg_counts[all_nbrs]
        write_idx = counts % self.mailbox_size

        self.msg_src[all_nbrs, write_idx] = src_rep
        self.msg_dst[all_nbrs, write_idx] = dst_rep
        self.msg_t[all_nbrs, write_idx] = t_rep
        self.msg_raw[all_nbrs, write_idx] = raw_msg_rep
        self.msg_dla[all_nbrs, write_idx] = all_nbrs

        self.msg_counts[all_nbrs] += 1



    def _fill_tensor_store_old(self, src, dst, t, raw_msg, all_neighbors):
        for i in range(src.size(0)):
            u = src[i].item()
            v = dst[i].item()
            neighbors = all_neighbors[u]
            for nbr in neighbors:
                if nbr == u:
                    continue
                count = self.msg_counts[nbr].item()
                write_idx = count % self.mailbox_size
                self.msg_src[nbr, write_idx] = u
                self.msg_dst[nbr, write_idx] = v
                self.msg_t[nbr, write_idx] = t[i]
                self.msg_raw[nbr, write_idx] = raw_msg[i]
                self.msg_dla[nbr, write_idx] = nbr
                self.msg_counts[nbr] += 1

    def _update_memory(self, n_id: Tensor, data=None):
        memory, last_update = self._get_updated_memory(n_id, data=data)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update
        self.msg_counts[n_id] = 0

    def _get_updated_memory_old(self, n_id: Tensor, data = None) -> Tuple[Tensor, Tensor]:
        # breakpoint()
        return self._get_updated_memory(n_id, data)
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        msg, t, src, dst = self._compute_msg(n_id)
        aggr = self.aggr_module(msg, self._assoc[src], t, n_id.size(0))
        # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        if isinstance(self.memory_updater, (GRUCell, RNNCell)):
            updated_memory = self.memory_updater(aggr, self.memory[n_id])
        else:
            x = torch.stack([self.memory[n_id], aggr], dim=1)
            updated_memory = self.memory_updater(x)
            # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        last_update = scatter(t, src, 0, self.last_update.size(0), reduce="max")[n_id]
        return updated_memory, last_update

    def _get_updated_memory(self, n_id, data):
        # breakpoint()
        if data is not None:
            eids = self.m_store[n_id].flatten()
            dirs = self.m_store_dir[n_id].flatten()
            n_id_flat = n_id.unsqueeze(1).expand(-1, 10).reshape(-1)

            mask = eids != -1

            valid_eids = eids[mask]
            valid_dirs = dirs[mask]
            valid_ux = n_id_flat[mask]

            msg = data[valid_eids].to(n_id.device)
            # Swap where dirs is False
            lmask = ~valid_dirs  # inverse of dirs (i.e., where it's False)
            src = torch.where(lmask, msg.dst, msg.src)
            dst = torch.where(lmask, msg.src, msg.dst)
            t = msg.t
            ux = valid_ux
            raw_msg = msg.msg
            self.buffer_msg = {
                'src': src,
                'dst': dst,
                't': t,
                'ux': ux,
                'raw_msg': raw_msg,
            }
        else:
            src = self.buffer_msg['src']
            dst = self.buffer_msg['dst']
            t = self.buffer_msg['t']
            ux = self.buffer_msg['ux']
            raw_msg = self.buffer_msg['raw_msg']
        
        mask = (t >= 0)
        src, dst, t, raw_msg, ux = src[mask], dst[mask], t[mask], raw_msg[mask], ux[mask]

        t_rel = t - self.last_update[ux.to(self.last_update.device)]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        aggr = self.aggr_module(msg, self._assoc[ux], t, n_id.size(0))
        # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        if isinstance(self.memory_updater, (GRUCell, RNNCell)):
            updated_memory = self.memory_updater(aggr, self.memory[n_id])
        else:
            x = torch.stack([self.memory[n_id], aggr], dim=1)
            updated_memory = self.memory_updater(x)
            # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        last_update = scatter(t, ux, 0, self.last_update.size(0), reduce="max")[n_id]
        return updated_memory, last_update
        



    def _compute_msg(self, n_id: Tensor):
        src = self.msg_src[n_id].flatten()
        dst = self.msg_dst[n_id].flatten()
        t = self.msg_t[n_id].flatten()
        raw_msg = self.msg_raw[n_id].reshape(-1, self.raw_msg_dim)
        ux = self.msg_dla[n_id].flatten()

        mask = (t >= 0)
        src, dst, t, raw_msg, ux = src[mask], dst[mask], t[mask], raw_msg[mask], ux[mask]

        t_rel = t - self.last_update[ux.to(self.last_update.device)]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, ux, dst

    def train(self, mode: bool = True):
        # if self.training and not mode:
        #     self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
        #     self._reset_message_store()
        super().train(mode)



class DA_APANMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "gru",
        mailbox_size: int = 10,
        num_head: int = 2,
        data = None,
        layer: int = 1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.mailbox_size = mailbox_size
        self.layer = layer
        self.data = data

        self.msg_module = message_module
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)

        # self.msg_s_store = {}
        # self.msg_d_store = {}
        self.m_counts = torch.zeros(num_nodes, dtype=torch.long, device='cuda:0')
        self.m_store = torch.full((num_nodes, self.mailbox_size), -1, dtype=torch.long, device='cuda:0')
        self.m_store_dir = torch.full((num_nodes, self.mailbox_size), False, dtype=torch.bool, device='cuda:0')


        if memory_updater_cell == "gru":
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "transformer":
            self.memory_updater = TransformerMemoryUpdater(
                input_dim=message_module.out_channels,
                memory_dim=memory_dim,
                nhead=num_head,           # Optional: make configurable
                num_layers=1
            )
        else:
            raise ValueError("Invalid memory updater type.")

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.reset_parameters()

    def _init_message_store(self):
        device = self.memory.device
        # self.msg_src = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        # self.msg_dst = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        # self.msg_t = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        # self.msg_dla = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        # self.msg_raw = torch.zeros((self.num_nodes, self.mailbox_size, self.raw_msg_dim), device=device)
        # self.msg_counts = torch.zeros(self.num_nodes, dtype=torch.long, device=device)
    # def mem_graph(self, ei_src, ei_dst, pos_node_s, pos_node_d):
    #     batch_size = pos_node_s.size(0)
    #     return mem_update_graph.apan_mem_graph(
    #         ei_src.to(torch.int64).contiguous(),
    #         ei_dst.to(torch.int64).contiguous(),
    #         pos_node_s.to(torch.int64).contiguous(),
    #         pos_node_d.to(torch.int64).contiguous(),
    #         batch_size
    #     )
    def mem_graph(self, od_updated, bs, max_seen_eid):
        tmp = od_updated[:, 0] % bs
        od_updated = torch.cat([od_updated, tmp.unsqueeze(1)], dim=1)
        N = od_updated.size(0)

        # Estimate maximum possible sizes (same as N)
        mem_quad = torch.zeros((4, N), dtype=torch.long, device=od_updated.device)
        store_quad = torch.zeros((4, N), dtype=torch.long, device=od_updated.device)

        mem_counter = torch.zeros(1, dtype=torch.int32, device=od_updated.device)
        store_counter = torch.zeros(1, dtype=torch.int32, device=od_updated.device)

        #.to(torch.int64).contiguous()
        mem_update_graph.apan_mem_graph(
            od_updated.contiguous(),
            mem_quad,
            store_quad,
            bs,
            max_seen_eid,
            mem_counter,
            store_counter
        )

        mem_quad_trimmed = mem_quad[:, :mem_counter.item()]
        store_quad_trimmed = store_quad[:, :store_counter.item()]

        return mem_quad_trimmed, store_quad_trimmed

    def reset_parameters(self):
        if hasattr(self.msg_module, "reset_parameters"):
            self.msg_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)
        self._init_message_store()

    def detach(self):
        self.memory.detach_()

    # def _reset_message_store(self):
    #     self.msg_src.fill_(-1)
    #     self.msg_dst.fill_(-1)
    #     self.msg_t.fill_(-1)
    #     self.msg_dla.fill_(-1)
    #     self.msg_raw.zero_()
    #     self.msg_counts.zero_()

    def forward_naive(self, n_id: Tensor, data=None) -> Tuple[Tensor, Tensor]:
        memory, last_update = self._get_updated_memory(n_id, data =  data)
        return memory, last_update
    
    def forward(self, n_id: Tensor, mem_graph, t, raw_msg, unique_keys, inverse_indices, data=None) -> Tuple[Tensor, Tensor]:
        memory, last_update = self._get_updated_memory(n_id, data = data)
        if mem_graph[0].size(0) == 0:
            # breakpoint()
            return memory, last_update
        for i in range(self.layer-1):
            memory, last_update = self._apply_intra_batch_info(mem_graph, n_id, t, raw_msg, memory, unique_keys, inverse_indices)
        return  self._apply_intra_batch_info(mem_graph, n_id, t, raw_msg, memory, unique_keys, inverse_indices)
    
    def _apply_intra_batch_info(self, mem_graph, n_id, b_t, b_raw_msg, old_mem, unique_keys, inverse_indices):
        # breakpoint()
        # msg, t, ux, dst  = self._intra_batch_compute_msg( mem_graph, n_id, b_t, b_raw_msg)

        src = n_id[unique_keys[:, 0]]
        dst = n_id[unique_keys[:, 1]]

        t_rel = b_t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(b_raw_msg.dtype))

        msg_unique = self.msg_module(self.memory[src], self.memory[dst], b_raw_msg, t_enc)

        ux = mem_graph[2]
        # breakpoint()
        # aggr = scatter(msg_unique[inverse_indices], ux, dim=0, dim_size=n_id.size(0), reduce="sum")
        # aggr = self.aggr_module(msg_unique[inverse_indices], ux, b_t[inverse_indices], n_id.size(0))
        aggr = self.aggr_module(msg_unique, ux, b_t, n_id.size(0), inverse_indices = inverse_indices)

        # breakpoint()














        # src = n_id[mem_graph[0]]
        # dst = n_id[mem_graph[1]]
        # ux = mem_graph[2]
        # t_rel = b_t - self.last_update[src]
        # t_enc = self.time_enc(t_rel.to(b_raw_msg.dtype))
        # msg = self.msg_module(self.memory[src], self.memory[dst], b_raw_msg, t_enc)


        # aggr = self.aggr_module(msg, ux, b_t, n_id.size(0))

        # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        if isinstance(self.memory_updater, (GRUCell, RNNCell)):
            updated_memory = self.memory_updater(aggr, old_mem)
        else:
            x = torch.stack([old_mem, aggr], dim=1)
            updated_memory = self.memory_updater(x)
            # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        # breakpoint()
        last_update = scatter(b_t[inverse_indices], ux, 0, n_id.size(0), reduce="max")
        # breakpoint()
        return updated_memory, last_update

    def _intra_batch_compute_msg(self, mem_graph, n_id, t, raw_msg):
        # x, xinv = torch.unique(mem_graph[3], dim=0, return_inverse=True)


        src = n_id[mem_graph[0]]
        dst = n_id[mem_graph[1]]
        ux = mem_graph[2]
        t_rel = t - self.last_update[n_id[ux]]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, ux, dst

    def update_state_naive(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, all_neighbors, n_ids, npadded = None, eid_start = 0, assoc = None, data = None):
        self._update_memory(n_ids, data = data)
        self.m_counts[n_ids] = 0
        self._fill_vector_store(src, dst, eid_start, npadded, assoc)


    # (n_id, z, last_update, store_quad, dataset['data'].t[store_eid].to(device), dataset['data'].msg[store_eid])

    def update_state(self, n_id, z, last_update, store_quad, dirs):
        unique_nid, inverse_indices = torch.unique(n_id, return_inverse=True)
        max_vals, max_indices = scatter_max(last_update, inverse_indices, dim=0)
        self.memory[unique_nid] = z[max_indices]
        self.last_update[unique_nid] = max_vals
        self.m_counts[unique_nid] = 0

        # breakpoint()

        store_quad = store_quad.T
        
        pairs =store_quad[:, 2:]
        # breakpoint()

        triads = torch.cat([pairs, dirs.unsqueeze(1).long()], dim=1)
        unique_triads = torch.unique(triads, dim=0)
        pairs = unique_triads[:, :2]
        dirs = unique_triads[:,2].bool()

        node_ids = pairs[:,0]#store_quad[:,2]
        msg_ids = pairs[:,1]

        # Step 1: Sort by node_id to group duplicates
        sorted_vals, sorted_idx = torch.sort(node_ids, stable=True)
        sorted_nodes = node_ids[sorted_idx]
        sorted_msgs = msg_ids[sorted_idx]
        # breakpoint()
        sorted_dirs = dirs[sorted_idx]

        # Step 2: Get per-node insertion offsets
        # inverse_idx gives the group assignment; counts how many per node
        unique_nodes, inverse_idx, counts = torch.unique_consecutive(sorted_nodes, return_inverse=True, return_counts=True)

        # How many previous messages each instance has seen within its group
        intra_offsets = torch.arange(len(sorted_nodes), device=pairs.device) - torch.cumsum(
            torch.bincount(inverse_idx, minlength=unique_nodes.size(0)), dim=0
        ).repeat_interleave(counts) + counts.repeat_interleave(counts)

        # Step 3: Compute true write positions in circular buffer
        # Each node’s global counter (before update)
        global_start = self.m_counts[sorted_nodes]
        write_idx = (global_start + intra_offsets) % self.mailbox_size

        # Step 4: Insert messages into store
        self.m_store[sorted_nodes, write_idx] = sorted_msgs
        self.m_store_dir[sorted_nodes, write_idx] = sorted_dirs#torch.full_like(sorted_msgs, isSrc, dtype=torch.bool)
        # breakpoint()

        # Step 5: Update msg_counts per node (add how many were added)
        # breakpoint()
        self.m_counts.index_add_(0, unique_nodes, counts)
        # breakpoint()



        # sorted_idx = torch.argsort(store_quad[:, 3])
        # store_quad_sorted = store_quad[sorted_idx]
        # store_t_sorted = store_t[sorted_idx]

        # sorted_idx_cpu = sorted_idx.cpu()
        # store_msg_sorted = store_msg[sorted_idx_cpu]
        # breakpoint()
        # for i in range(store_quad_sorted.shape[0]):
        #     nbr = store_quad_sorted[i, 2]
        #     count = self.msg_counts[nbr].item()
        #     write_idx = count % self.mailbox_size
        #     self.msg_src[nbr, write_idx] = n_id[store_quad_sorted[i, 0]]
        #     self.msg_dst[nbr, write_idx] = n_id[store_quad_sorted[i, 1]]
        #     self.msg_t[nbr, write_idx] = store_t_sorted[i]
        #     self.msg_raw[nbr, write_idx] = store_msg_sorted[i]
        #     self.msg_dla[nbr, write_idx] = nbr
        #     self.msg_counts[nbr] += 1
    

    def _fill_vector_store(self, src, dst, eid_start, npadded, assoc):
        # breakpoint()
        msg_id = torch.arange(eid_start, eid_start+src.size(0), device=src.device)
        # self.msg_s_store[]
        npadded_idx = assoc[src]
        npadded_idx_dst =  assoc[dst]
        src_npadded = npadded[npadded_idx]
        dst_npadded = npadded[npadded_idx_dst]

        msg_id_expanded = msg_id.unsqueeze(1).expand_as(src_npadded)  # (N, K)
        flat_nodes = src_npadded.flatten()
        flat_msg_ids = msg_id_expanded.flatten()
        msg_id_expanded_dst = msg_id.unsqueeze(1).expand_as(dst_npadded)  # (N, K)
        flat_nodes_dst = dst_npadded.flatten()
        flat_msg_ids_dst = msg_id_expanded.flatten()

        # Remove -1 entries
        valid = flat_nodes != -1
        flat_nodes = flat_nodes[valid]       # (M,)
        flat_msg_ids = flat_msg_ids[valid]   # (M,)
        valid_dst = flat_nodes_dst != -1
        flat_nodes_dst = flat_nodes_dst[valid_dst]       # (M,)
        flat_msg_ids_dst = flat_msg_ids_dst[valid_dst]   # (M,)



        # Sort by flat_nodes so we can group
        sorted_nodes, sort_idx = torch.sort(flat_nodes)
        sorted_msg_ids = flat_msg_ids[sort_idx]
        # breakpoint()

        pairs = torch.stack([flat_nodes, flat_msg_ids], dim=1)  # shape: (N, 2)
        new_pairs = torch.stack([dst, msg_id], dim=1) 
        pairs = torch.cat([pairs, new_pairs], dim=0)
        pairs = torch.unique(pairs, dim=0)

        pairs_dst = torch.stack([flat_nodes_dst, flat_msg_ids_dst], dim=1)  # shape: (N, 2)
        new_pairs_dst = torch.stack([src, msg_id], dim=1) 
        pairs_dst = torch.cat([pairs_dst, new_pairs_dst], dim=0)
        pairs_dst = torch.unique(pairs_dst, dim=0)
        
        sdir = torch.full_like(pairs[:,1], True, dtype=torch.bool)
        ddir = torch.full_like(pairs_dst[:,1], False, dtype=torch.bool)

        combined_pairs = torch.cat([pairs, pairs_dst], dim=0)
        combined_dirs = torch.cat([sdir, ddir], dim=0)
        
        sorted_vals, sorted_idx = torch.sort(combined_pairs[:, 0] * (combined_pairs[:, 0].max() + 1) + combined_pairs[:, 1])
        pairs = combined_pairs[sorted_idx]
        dirs = combined_dirs[sorted_idx]

        
        # breakpoint()


        node_ids = pairs[:, 0]
        msg_ids = pairs[:, 1]

        # Step 1: Sort by node_id to group duplicates
        sorted_vals, sorted_idx = torch.sort(node_ids, stable=True)
        sorted_nodes = node_ids[sorted_idx]
        sorted_msgs = msg_ids[sorted_idx]
        sorted_dirs = dirs[sorted_idx]

        # Step 2: Get per-node insertion offsets
        # inverse_idx gives the group assignment; counts how many per node
        unique_nodes, inverse_idx, counts = torch.unique_consecutive(sorted_nodes, return_inverse=True, return_counts=True)

        # How many previous messages each instance has seen within its group
        intra_offsets = torch.arange(len(sorted_nodes), device=pairs.device) - torch.cumsum(
            torch.bincount(inverse_idx, minlength=unique_nodes.size(0)), dim=0
        ).repeat_interleave(counts) + counts.repeat_interleave(counts)

        # Step 3: Compute true write positions in circular buffer
        # Each node’s global counter (before update)
        global_start = self.m_counts[sorted_nodes]
        write_idx = (global_start + intra_offsets) % self.mailbox_size

        # Step 4: Insert messages into store
        self.m_store[sorted_nodes, write_idx] = sorted_msgs
        self.m_store_dir[sorted_nodes, write_idx] = sorted_dirs#torch.full_like(sorted_msgs, isSrc, dtype=torch.bool)
        # breakpoint()

        # Step 5: Update msg_counts per node (add how many were added)
        # breakpoint()
        self.m_counts.index_add_(0, unique_nodes, counts)
        # breakpoint()




    def _update_memory(self, n_id: Tensor, data=None):
        memory, last_update = self._get_updated_memory(n_id, data=data)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update
        self.msg_counts[n_id] = 0

    # def _get_updated_memory_old(self, n_id: Tensor, data = None) -> Tuple[Tensor, Tensor]:
    #     # breakpoint()
    #     return self._get_updated_memory(n_id, data)

    def _get_updated_memory(self, n_id, data):
        # breakpoint()
        if data is not None:
            eids = self.m_store[n_id].flatten()
            dirs = self.m_store_dir[n_id].flatten()
            n_id_flat = n_id.unsqueeze(1).expand(-1, 10).reshape(-1)

            mask = eids != -1

            valid_eids = eids[mask]
            valid_dirs = dirs[mask]
            valid_ux = n_id_flat[mask]

            # breakpoint()

            msg = data[valid_eids.cpu()].to(n_id.device)
            # Swap where dirs is False
            lmask = ~valid_dirs  # inverse of dirs (i.e., where it's False)
            src = torch.where(lmask, msg.dst, msg.src)
            dst = torch.where(lmask, msg.src, msg.dst)
            t = msg.t
            ux = valid_ux
            raw_msg = msg.msg
            self.buffer_msg = {
                'src': src,
                'dst': dst,
                't': t,
                'ux': ux,
                'raw_msg': raw_msg,
            }
        else:
            src = self.buffer_msg['src']
            dst = self.buffer_msg['dst']
            t = self.buffer_msg['t']
            ux = self.buffer_msg['ux']
            raw_msg = self.buffer_msg['raw_msg']
        
        mask = (t >= 0)
        src, dst, t, raw_msg, ux = src[mask], dst[mask], t[mask], raw_msg[mask], ux[mask]

        t_rel = t - self.last_update[ux.to(self.last_update.device)]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        aggr = self.aggr_module(msg, self._assoc[ux], t, n_id.size(0))
        # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        if isinstance(self.memory_updater, (GRUCell, RNNCell)):
            updated_memory = self.memory_updater(aggr, self.memory[n_id])
        else:
            x = torch.stack([self.memory[n_id], aggr], dim=1)
            updated_memory = self.memory_updater(x)
            # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        last_update = scatter(t, ux, 0, self.last_update.size(0), reduce="max")[n_id]
        return updated_memory, last_update
        



    def _compute_msg(self, n_id: Tensor):
        src = self.msg_src[n_id].flatten()
        dst = self.msg_dst[n_id].flatten()
        t = self.msg_t[n_id].flatten()
        raw_msg = self.msg_raw[n_id].reshape(-1, self.raw_msg_dim)
        ux = self.msg_dla[n_id].flatten()

        mask = (t >= 0)
        src, dst, t, raw_msg, ux = src[mask], dst[mask], t[mask], raw_msg[mask], ux[mask]

        t_rel = t - self.last_update[ux.to(self.last_update.device)]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, ux, dst

    def train(self, mode: bool = True):
        # if self.training and not mode:
        #     self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
        #     self._reset_message_store()
        super().train(mode)



class TransformerMemoryUpdater(torch.nn.Module):
    def __init__(self, input_dim, memory_dim, nhead=2, num_layers=1):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = torch.nn.Linear(input_dim, memory_dim)

    def forward(self, msg: Tensor, memory: Tensor) -> Tensor:
        # msg: [batch_size, dim], memory: [batch_size, dim]
        x = torch.stack([memory, msg], dim=1)  # [B, 2, D]
        out = self.encoder(x)[:, -1]  # Use updated token
        return self.linear(out)
    
















class DAAAPANMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "gru",
        mailbox_size: int = 10,
        num_head: int = 2,
        layer: int = 3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.mailbox_size = mailbox_size
        self.layer = layer

        self.msg_module = message_module
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)

        if memory_updater_cell == "gru":
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "rnn":
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "transformer":
            self.memory_updater = TransformerMemoryUpdater(
                input_dim=message_module.out_channels,
                memory_dim=memory_dim,
                nhead=num_head,           # Optional: make configurable
                num_layers=1
            )
        else:
            raise ValueError("Invalid memory updater type.")

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.reset_parameters()

    def _init_message_store(self):
        device = self.memory.device
        self.msg_src = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        self.msg_dst = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        self.msg_t = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        self.msg_dla = torch.full((self.num_nodes, self.mailbox_size), -1, dtype=torch.long, device=device)
        self.msg_raw = torch.zeros((self.num_nodes, self.mailbox_size, self.raw_msg_dim))
        self.msg_counts = torch.zeros(self.num_nodes, dtype=torch.long, device=device)

    def reset_parameters(self):
        if hasattr(self.msg_module, "reset_parameters"):
            self.msg_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)
        self._init_message_store()

    def detach(self):
        self.memory.detach_()

    def _reset_message_store(self):
        self.msg_src.fill_(-1)
        self.msg_dst.fill_(-1)
        self.msg_t.fill_(-1)
        self.msg_dla.fill_(-1)
        self.msg_raw.zero_()
        self.msg_counts.zero_()

    def mem_graph(self, od_updated, bs, max_seen_eid):
        return mem_update_graph.mem_graph_apan(
            od_updated.to(torch.int64).contiguous(),
            bs,
            max_seen_eid
        )


    def forward(self, n_id: Tensor, mem_graph, t, raw_msg) -> Tuple[Tensor, Tensor]:
        memory, last_update = self._get_updated_memory(n_id)
        for i in range(self.layer-1):
            memory, last_update = self._apply_intra_batch_info(mem_graph, n_id, t, raw_msg)
        return  self._apply_intra_batch_info(mem_graph, n_id, t, raw_msg)
    
    def _apply_intra_batch_info(self, mem_graph, n_id, b_t, b_raw_msg):
        msg, t, ux, dst  = self._intra_batch_compute_msg( mem_graph, n_id, b_t, b_raw_msg)

        aggr = self.aggr_module(msg, ux, t, n_id.size(0))

        # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        if isinstance(self.memory_updater, (GRUCell, RNNCell)):
            updated_memory = self.memory_updater(aggr, self.memory[n_id])
        else:
            x = torch.stack([self.memory[n_id], aggr], dim=1)
            updated_memory = self.memory_updater(x)
            # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        last_update = scatter(t, ux, 0, n_id.size(0), reduce="max")
        # breakpoint()
        return updated_memory, last_update

        # return None, None

    def update_state_old(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, all_neighbors, n_ids):
        self._update_memory(n_ids)
        self._fill_tensor_store(src, dst, t, raw_msg, all_neighbors)
        self._fill_tensor_store(dst, src, t, raw_msg, all_neighbors)
    

    def update_state_vectorized(self, n_id, z, last_update, store_quad, store_t, store_msg):
        # Step 1: Update memory and last update
        unique_nid, inverse_indices = torch.unique(n_id, return_inverse=True)
        max_vals, max_indices = scatter_max(last_update, inverse_indices, dim=0)
        self.memory[unique_nid] = z[max_indices]
        self.last_update[unique_nid] = max_vals
        self.msg_counts[unique_nid] = 0  # reset counts

        # Step 2: Sort store_quad by time (column 3)
        store_quad = store_quad.T  # now shape [4, N]
        time_col = store_quad[3]
        sorted_idx = torch.argsort(time_col)

        store_quad_sorted = store_quad[:, sorted_idx]  # shape [4, N]
        store_t_sorted = store_t[sorted_idx]
        store_msg_sorted = store_msg[sorted_idx.cpu()]  # assuming msg on CPU

        # Step 3: Vectorize mailbox write
        src_idx = n_id[store_quad_sorted[0]]  # shape [N]
        dst_idx = n_id[store_quad_sorted[1]]
        dla_idx = store_quad_sorted[2]        # already node ids
        t_vals = store_t_sorted               # shape [N]
        msg_vals = store_msg_sorted           # shape [N, msg_dim]

        # Get current counts and compute write indices
        count_vals = self.msg_counts[dla_idx]
        write_idx = count_vals % self.mailbox_size

        # Write to mailboxes
        self.msg_src[dla_idx, write_idx] = src_idx
        self.msg_dst[dla_idx, write_idx] = dst_idx
        self.msg_t[dla_idx, write_idx] = t_vals
        self.msg_raw[dla_idx, write_idx] = msg_vals
        self.msg_dla[dla_idx, write_idx] = dla_idx

        # Increment counts
        self.msg_counts[dla_idx] += 1


    def update_state(self, n_id, z, last_update, store_quad, store_t, store_msg):
        unique_nid, inverse_indices = torch.unique(n_id, return_inverse=True)
        max_vals, max_indices = scatter_max(last_update, inverse_indices, dim=0)
        self.memory[unique_nid] = z[max_indices]
        self.last_update[unique_nid] = max_vals
        self.msg_counts[unique_nid] = 0

        store_quad = store_quad.T
        sorted_idx = torch.argsort(store_quad[:, 3])
        store_quad_sorted = store_quad[sorted_idx]
        store_t_sorted = store_t[sorted_idx]

        sorted_idx_cpu = sorted_idx.cpu()
        store_msg_sorted = store_msg[sorted_idx_cpu]
        # breakpoint()
        for i in range(store_quad_sorted.shape[0]):
            nbr = store_quad_sorted[i, 2]
            count = self.msg_counts[nbr].item()
            write_idx = count % self.mailbox_size
            self.msg_src[nbr, write_idx] = n_id[store_quad_sorted[i, 0]]
            self.msg_dst[nbr, write_idx] = n_id[store_quad_sorted[i, 1]]
            self.msg_t[nbr, write_idx] = store_t_sorted[i]
            self.msg_raw[nbr, write_idx] = store_msg_sorted[i]
            self.msg_dla[nbr, write_idx] = nbr
            self.msg_counts[nbr] += 1


        
        # print("Not Implemented Yet")
        # breakpoint()

    def _fill_tensor_store(self, src, dst, t, raw_msg, all_neighbors):
        for i in range(src.size(0)):
            u = src[i].item()
            v = dst[i].item()
            neighbors = all_neighbors[u]
            for nbr in neighbors:
                if nbr == u:
                    continue
                count = self.msg_counts[nbr].item()
                write_idx = count % self.mailbox_size
                self.msg_src[nbr, write_idx] = u
                self.msg_dst[nbr, write_idx] = v
                self.msg_t[nbr, write_idx] = t[i]
                self.msg_raw[nbr, write_idx] = raw_msg[i]
                self.msg_dla[nbr, write_idx] = nbr
                self.msg_counts[nbr] += 1

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update
        self.msg_counts[n_id] = 0

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        msg, t, src, dst = self._compute_msg(n_id)
        aggr = self.aggr_module(msg, self._assoc[src], t, n_id.size(0))
        # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        if isinstance(self.memory_updater, (GRUCell, RNNCell)):
            updated_memory = self.memory_updater(aggr, self.memory[n_id])
        else:
            x = torch.stack([self.memory[n_id], aggr], dim=1)
            updated_memory = self.memory_updater(x)
            # updated_memory = self.memory_updater(aggr, self.memory[n_id])
        last_update = scatter(t, src, 0, self.last_update.size(0), reduce="max")[n_id]
        return updated_memory, last_update

    def _compute_msg(self, n_id: Tensor):
        src = self.msg_src[n_id].flatten()
        dst = self.msg_dst[n_id].flatten()
        t = self.msg_t[n_id].flatten()
        rm = self.msg_raw[n_id.cpu()].to(dst.device)
        raw_msg = rm.reshape(-1, self.raw_msg_dim)
        ux = self.msg_dla[n_id].flatten()

        mask = (t >= 0)
        src, dst, t, raw_msg, ux = src[mask], dst[mask], t[mask], raw_msg[mask], ux[mask]

        t_rel = t - self.last_update[ux.to(self.last_update.device)]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        # print(src.max())
        # print(dst.max())
        # print(src.min())
        # print(dst.min())
        # breakpoint()
        msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, ux, dst

    def _intra_batch_compute_msg(self, mem_graph, n_id, t, raw_msg):
        # src = self.msg_src[n_id].flatten()
        # dst = self.msg_dst[n_id].flatten()
        # t = self.msg_t[n_id].flatten()
        # raw_msg = self.msg_raw[n_id].reshape(-1, self.raw_msg_dim)
        # ux = self.msg_dla[n_id].flatten()

        # mask = (t >= 0)
        # src, dst, t, raw_msg, ux = src[mask], dst[mask], t[mask], raw_msg[mask], ux[mask]

        src = n_id[mem_graph[0]]
        dst = n_id[mem_graph[1]]
        ux = mem_graph[2]
        # e_id = mem_graph[3]

        # t = b_t[e_id]
        # raw_msg = b_raw_msg[e_id]
        

        t_rel = t - self.last_update[n_id[ux]]

        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, ux, dst

    def train(self, mode: bool = True):
        if self.training and not mode:
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)
