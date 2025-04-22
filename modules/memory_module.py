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

TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]



class DATGNMemory(torch.nn.Module):
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

    


    def forward(self, assoc, src: Tensor, pos_dst: Tensor, neg_dst: Tensor,
                b_nid: Tensor, b_edge_index: Tensor, b_eid: Tensor, 
                srcs: list, pos_dsts: list, neg_dsts: list, eids: list, t: list, raw_msg: list):


        """
        b0_XX: all the nodes required in the entire batch
        Need to calculate specific properties for each block
        How to get 
        """
        # if not self.training:
        #     pdb.set_trace()

        #batch updated memory
        memory, last_update = self._get_updated_memory(b_nid)
        msm_assoc = torch.arange(0, b_nid.shape[0], dtype=torch.int64, device=memory.device)
        lum_assoc = torch.arange(0, b_nid.shape[0], dtype=torch.int64, device=memory.device)

        local_store_s = torch.full((b_nid.shape[0],), -1, device=memory.device)
        local_store_d = torch.full((b_nid.shape[0],), -1, device=memory.device)
        local_store_mask_src = torch.full((b_nid.shape[0],), False, device=memory.device)
        local_store_mask_dst = torch.full((b_nid.shape[0],), False, device=memory.device)
        

        #block 0 proecessing
        
        # print(srcs)

        b_nid_after_blocking = b_nid.clone()
        


        ##### everything should be mapped following the order of how node memory's are stored******
        src_indices_lst = []
        pos_dst_indices_lst = []
        neg_dst_indices_lst = []
        edge_indices_lst = []
        eid_lst = []


        for bx in range(len(srcs)):
            #prepare mfg: append sampled neighborhood
            src_index = msm_assoc[assoc[srcs[bx]]]
            pos_dst_index = msm_assoc[assoc[pos_dsts[bx]]]
            neg_dst_index  = msm_assoc[assoc[neg_dsts[bx]]]


            # #maps last used memory ## not correct
            # lum_assoc[assoc[srcs[bx]]] = src_index
            # lum_assoc[assoc[pos_dsts[bx]]] = pos_dst_index
            # lum_assoc[assoc[neg_dsts[bx]]] = neg_dst_index


            


            src_indices_lst.append(src_index)
            pos_dst_indices_lst.append(pos_dst_index)
            neg_dst_indices_lst.append(neg_dst_index)

            src_node_curr_memory = memory[src_index]
            src_node_last_update = last_update[src_index]
            pos_dst_node_curr_memory = memory[pos_dst_index]
            pos_dst_node_last_update = last_update[pos_dst_index]
            # neg_dst_node_curr_memory = memory[neg_dst_index]
            # neg_dst_node_last_index = last_update[neg_dst_index]


            bx_pos_nids = torch.cat([srcs[bx], pos_dsts[bx]]).unique()
            bx_nids = torch.cat([bx_pos_nids, neg_dsts[bx]]).unique()
            bx_nids_index = assoc[bx_nids] #nodes involved in the current block

            msm_bx_nids_index = msm_assoc[assoc[bx_pos_nids]]
            bx_nids_memory = memory[msm_bx_nids_index]

            

            #neighborhood of the block nodes at the start of the batch
            indices = torch.nonzero(torch.isin(b_edge_index[1], bx_nids_index)).squeeze(1)
            bx_edge_index = msm_assoc[b_edge_index[:,indices]]

            #CODES FOR TRACTED UPDATE
            # pdb.set_trace()
            used_mem_indices = b_edge_index[:,indices].view(-1).unique()
            lum_assoc[used_mem_indices] = msm_assoc[used_mem_indices]
            local_store_mask_src[used_mem_indices] = False
            local_store_mask_dst[used_mem_indices] = False 


            # if len(bx_edge_index.shape)==1:
            #     pdb.set_trace()


            edge_indices_lst.append(bx_edge_index)
            bx_eid = b_eid[indices] #actual_eid, no further transformation needed
            eid_lst.append(bx_eid)
            # pdb.set_trace()



            #calculate updated memory of b_pos_nids anpos_dst_node_last_updated append at the tail of memory vector
            
            new_memory, new_last_update = self._get_updated_memory_local(bx_pos_nids, bx_nids_memory, srcs[bx], pos_dsts[bx],
                                           src_node_curr_memory, pos_dst_node_curr_memory,
                                           src_node_last_update, pos_dst_node_last_update,
                                           t[bx], raw_msg[bx])

            #update msm_assoc as new versions of memory will be calculated for the nodes in bx_pos_nids
            memory = torch.cat([memory, new_memory], dim=0)
            last_update = torch.cat([last_update, new_last_update], dim=0)



            msm_assoc[assoc[bx_pos_nids]] = torch.arange(b_nid_after_blocking.shape[0], 
                                                  b_nid_after_blocking.shape[0]+bx_pos_nids.shape[0], 
                                                  device=msm_assoc.device, dtype=torch.int64)
            b_nid_after_blocking = torch.cat([b_nid_after_blocking, bx_pos_nids])


            #CODES FOR TRACTED UPDATE
            #updates local store with new pending updates
            local_store_s[assoc[srcs[bx]]] = eids[bx]#pos_dsts[bx]
            local_store_d[assoc[pos_dsts[bx]]] = eids[bx]#srcs[bx]
            local_store_mask_src[assoc[srcs[bx]]] = True
            local_store_mask_dst[assoc[pos_dsts[bx]]] = True


            
            #append with new edges (with original mapping)

            _new_b_edge_index = torch.stack((torch.cat((assoc[srcs[bx]], assoc[pos_dsts[bx]])), torch.cat((assoc[pos_dsts[bx]], assoc[srcs[bx]]))), dim=0)
            b_edge_index = torch.cat((b_edge_index, _new_b_edge_index), dim=1)
            b_eid = torch.cat([b_eid, eids[bx], eids[bx]])


        
        # print(edge_indices_lst)
        el_r = torch.cat(edge_indices_lst, dim=1)
        eid_r = torch.cat(eid_lst, dim=0)
        s_r = torch.cat(src_indices_lst, dim=0)
        p_d_r = torch.cat(pos_dst_indices_lst, dim=0)
        n_d_r = torch.cat(neg_dst_indices_lst, dim=0)
        # pdb.set_trace()

        #node_ids_with updated_memory ---done
        #pending_d_store update
        #pending_s_store_update

        #CODES FOR TRACTED UPDATE
        s_store_eid = local_store_s[local_store_mask_src]#(b_nid[local_store_mask_src], local_store_s[local_store_mask_src])
        d_store_eid = local_store_d[local_store_mask_dst]#(b_nid[local_store_mask_dst], local_store_d[local_store_mask_dst])
        
        
        return memory, last_update, el_r, eid_r, s_r, p_d_r, n_d_r, lum_assoc, s_store_eid, d_store_eid

    def update_memory_DATGN(self, nid, memory, lum_assoc):
        # pdb.set_trace()
        self.memory[nid] = memory[lum_assoc]

    def update_state_DATGN(
            self, nid: Tensor, memory, lum_assoc,
            s_src: Tensor, s_dst: Tensor, s_t: Tensor, s_raw_msg: Tensor,
            d_src: Tensor, d_dst: Tensor, d_t: Tensor, d_raw_msg: Tensor,
        ):
        self.memory[nid] = memory[lum_assoc]
        # self.update_memory_DATGN(nid, memory, lum_assoc)
        self._update_msg_store(s_src, s_dst, s_t, s_raw_msg, self.msg_s_store)
        self._update_msg_store(d_dst, d_src, d_t, d_raw_msg, self.msg_d_store)


    def forward_legacy(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        # memory, last_update = self._get_updated_memory(n_id)
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


    def _get_updated_memory_local(
            self,
            n_id,
            n_id_mem,
            src,
            dst,
            src_memory,
            dst_memory,
            src_last_update,
            dst_last_update,
            raw_t,
            raw_msg,


    ):
        # n_id_2 = torch.cat([src, dst])
        # if n_id.shape != n_id_2.shape:
        #     pdb.set_trace
        # if not torch.equal(n_id_2, n_id):
        #     pdb.set_trace()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        
        msg_s, t_s, src_s, dst_s = self._compute_msg_local(
            raw_t, 
            src_last_update,
            src,
            dst,
            raw_msg,
            src_memory,
            dst_memory,
            self.msg_s_module,
        )

        msg_d, t_d, src_d, dst_d = self._compute_msg_local(
            raw_t, 
            dst_last_update,
            dst,
            src,
            raw_msg,
            dst_memory,
            src_memory,
            self.msg_d_module,
        )

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        # pdb.set_trace()
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # aggr =  torch.cat([aggr[self._assoc[src_s], aggr[self._assoc[src_s]]])

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, n_id_mem)

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]

        return memory, last_update


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

    

    def _compute_msg_local(
            self,
            src_t, 
            src_last_update,
            src,
            dst,
            raw_msg,
            src_memory,
            dst_memory,
            msg_module,
            
    ):
        t_rel = src_t - src_last_update
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = msg_module(src_memory, dst_memory, raw_msg, t_enc)
        return msg, src_t, src, dst


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
















class TGNMemory(torch.nn.Module):
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

    def forward(self, n_id, b_edge_index, b_t, b_raw_msg, b_isrc) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
            return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)
        else:
            nn_id  = n_id.unique()
            self._assoc[nn_id] = torch.arange(nn_id.size(0), device=nn_id.device)
            memory, last_update = self.memory[nn_id], self.last_update[nn_id]
            return self._apply_intra_batch_info(n_id, memory, last_update, b_edge_index, b_t, b_raw_msg, b_isrc)

        # return memory, last_update

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

    def _intra_batch_compute_msg(self, all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm):
        msrc_s = b_edge_index[1][b_isrc]
        src_s = all_n_id[msrc_s]
        dst_s = all_n_id[b_edge_index[0][b_isrc]]
        raw_msg_s = b_raw_msg[b_isrc]
        t_s = b_t[b_isrc]
        t_rel_s = t_s - last_update[self._assoc[src_s]]
        t_enc_s = self.time_enc(t_rel_s.to(raw_msg_s.dtype))
        msg_s = self.msg_s_module(bm[self._assoc[src_s]], bm[self._assoc[dst_s]], raw_msg_s, t_enc_s)

        return msg_s, t_s, src_s, dst_s, msrc_s



    def _apply_intra_batch_info(self, all_n_id, bm, last_update, b_edge_index, b_t, b_raw_msg, b_isrc):
        # breakpoint()
        # print(self._assoc[all_n_id])
        # print(all_n_id.max())
        # print(self._assoc[all_n_id].max())
        # print(bm.shape)

        old_mem = bm[self._assoc[all_n_id]]

        # breakpoint()

        msg_s, t_s, src_s, dst_s, msrc_s  = self._intra_batch_compute_msg(all_n_id, b_edge_index, b_isrc, b_raw_msg, b_t, last_update, bm)
        msg_d, t_d, src_d, dst_d, msrc_d  = self._intra_batch_compute_msg(all_n_id, b_edge_index, ~b_isrc, b_raw_msg, b_t, last_update, bm)

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






class DyRepMemory(torch.nn.Module):

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









class DATGNMemoryv2(torch.nn.Module):
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
        self.draft = 0

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

    
    def get_occurrences(self, a, b):
        sorted_a, sorted_idx = torch.sort(a)

        # Find the start index of each b element in sorted_a
        search_idx = torch.searchsorted(sorted_a, b)

        # Collect all matching indices using a mask
        indices_list = [
            sorted_idx[search_idx[i] + torch.nonzero(sorted_a[search_idx[i]:] == b[i], as_tuple=True)[0]]
            for i in range(len(b))
        ]
        flat_indices = torch.cat(indices_list)

        # Step 5: Compute the cumulative start positions
        range_mapping = torch.cat((torch.tensor([0]), torch.cumsum(torch.tensor([len(x) for x in indices_list]), dim=0)))
        # breakpoint()
        return flat_indices, range_mapping.to(self.device)
    
    def get_occurences_for_batch(self, b, flat_indices, range_mapping, block_end):
        start_positions = range_mapping[b]
        end_positions = range_mapping[b + 1]

        # Step 2: Efficiently generate all index ranges in a single tensor operation
        range_lengths = end_positions - start_positions  # Lengths of each range
        all_indices = torch.arange(start_positions.min(), end_positions.max(), device=self.device)  # Generate full range
        mask = (all_indices[:, None] >= start_positions) & (all_indices[:, None] < end_positions)
        selected_indices = flat_indices[all_indices[mask.any(dim=1)]]

        return selected_indices[selected_indices<block_end]


    def forward(self, assoc, src: Tensor, pos_dst: Tensor, neg_dst: Tensor,
                b_nid: Tensor, b_edge_index: Tensor, b_eid: Tensor, 
                srcs: list, pos_dsts: list, neg_dsts: list, eids: list, t: list, raw_msg: list):


        """
        b0_XX: all the nodes required in the entire batch
        Need to calculate specific properties for each block
        How to get 
        """
        # if not self.training:
        #     pdb.set_trace()

        #batch updated memory
        memory, last_update = self._get_updated_memory(b_nid)
        msm_assoc = torch.arange(0, b_nid.shape[0], dtype=torch.int64, device=memory.device)
        lum_assoc = torch.arange(0, b_nid.shape[0], dtype=torch.int64, device=memory.device)

        local_store_s = torch.full((b_nid.shape[0],), -1, device=memory.device)
        local_store_d = torch.full((b_nid.shape[0],), -1, device=memory.device)
        local_store_mask_src = torch.full((b_nid.shape[0],), False, device=memory.device)
        local_store_mask_dst = torch.full((b_nid.shape[0],), False, device=memory.device)
        

        #block 0 proecessing
        
        # print(srcs)

        b_nid_after_blocking = b_nid.clone()
        


        ##### everything should be mapped following the order of how node memory's are stored******
        src_indices_lst = []
        pos_dst_indices_lst = []
        neg_dst_indices_lst = []
        edge_indices_lst = []
        eid_lst = []


        edge_index_list = [
            torch.stack((torch.cat((assoc[srcs[bx]], assoc[pos_dsts[bx]])), 
                        torch.cat((assoc[pos_dsts[bx]], assoc[srcs[bx]]))), dim=0)
            for bx in range(len(srcs))  # Assuming srcs and pos_dsts have the same length
        ]

        # b_eid = torch.cat([b_eid, eids[bx], eids[bx]])
        



        # Compute the sizes of each edge_index_bx (number of columns in each)
        sizes = torch.tensor([edge.shape[1] for edge in edge_index_list])

        # Compute cumulative sizes
        cumulative_sizes = torch.cat((torch.tensor([0]), sizes.cumsum(dim=0)))+b_edge_index.shape[1]

        edge_index_batched = torch.cat(edge_index_list, dim=1) 
        updated_edge_index = torch.cat([b_edge_index, edge_index_batched], dim=1)


        new_eid_list = [torch.cat([eid_t, eid_t]) for eid_t in eids]
        new_eids = torch.cat(new_eid_list)
        updated_eids = torch.cat([b_eid, new_eids])

        flat_indices, range_mapping = self.get_occurrences(updated_edge_index[1], assoc[b_nid])

        # breakpoint()

        for bx in range(len(srcs)):
            #prepare mfg: append sampled neighborhood
            src_index = msm_assoc[assoc[srcs[bx]]]
            pos_dst_index = msm_assoc[assoc[pos_dsts[bx]]]
            neg_dst_index  = msm_assoc[assoc[neg_dsts[bx]]]
            src_indices_lst.append(src_index)
            pos_dst_indices_lst.append(pos_dst_index)
            neg_dst_indices_lst.append(neg_dst_index)

            src_node_curr_memory = memory[src_index]
            src_node_last_update = last_update[src_index]
            pos_dst_node_curr_memory = memory[pos_dst_index]
            pos_dst_node_last_update = last_update[pos_dst_index]
            # neg_dst_node_curr_memory = memory[neg_dst_index]
            # neg_dst_node_last_index = last_update[neg_dst_index]


            bx_pos_nids = torch.cat([srcs[bx], pos_dsts[bx]]).unique()
            bx_nids = torch.cat([bx_pos_nids, neg_dsts[bx]]).unique()
            bx_nids_index = assoc[bx_nids] #nodes involved in the current block

            msm_bx_nids_index = msm_assoc[assoc[bx_pos_nids]]
            bx_nids_memory = memory[msm_bx_nids_index]

            indices = torch.nonzero(torch.isin(updated_edge_index[1,:cumulative_sizes[bx]], bx_nids_index)).squeeze(1)
            bx_edge_index = msm_assoc[updated_edge_index[:,indices]]
            
            used_mem_indices = updated_edge_index[:,indices].view(-1).unique()
            # breakpoint()

            lum_assoc[used_mem_indices] = msm_assoc[used_mem_indices]
            local_store_mask_src[used_mem_indices] = False
            local_store_mask_dst[used_mem_indices] = False 

            edge_indices_lst.append(bx_edge_index)

            # bx_eid = b_eid[indices] #actual_eid, no further transformation needed
            bx_eid = updated_eids[indices]
            # breakpoint()
            eid_lst.append(bx_eid)
            # pdb.set_trace()



            #calculate updated memory of b_pos_nids anpos_dst_node_last_updated append at the tail of memory vector
            
            new_memory, new_last_update = self._get_updated_memory_local(bx_pos_nids, bx_nids_memory, srcs[bx], pos_dsts[bx],
                                           src_node_curr_memory, pos_dst_node_curr_memory,
                                           src_node_last_update, pos_dst_node_last_update,
                                           t[bx], raw_msg[bx])

            #update msm_assoc as new versions of memory will be calculated for the nodes in bx_pos_nids
            memory = torch.cat([memory, new_memory], dim=0)
            last_update = torch.cat([last_update, new_last_update], dim=0)



            msm_assoc[assoc[bx_pos_nids]] = torch.arange(b_nid_after_blocking.shape[0], 
                                                  b_nid_after_blocking.shape[0]+bx_pos_nids.shape[0], 
                                                  device=msm_assoc.device, dtype=torch.int64)
            b_nid_after_blocking = torch.cat([b_nid_after_blocking, bx_pos_nids])


            #CODES FOR TRACTED UPDATE
            #updates local store with new pending updates
            local_store_s[assoc[srcs[bx]]] = eids[bx]#pos_dsts[bx]
            local_store_d[assoc[pos_dsts[bx]]] = eids[bx]#srcs[bx]
            local_store_mask_src[assoc[srcs[bx]]] = True
            local_store_mask_dst[assoc[pos_dsts[bx]]] = True


        
        # print(edge_indices_lst)
        el_r = torch.cat(edge_indices_lst, dim=1)
        eid_r = torch.cat(eid_lst, dim=0)
        s_r = torch.cat(src_indices_lst, dim=0)
        p_d_r = torch.cat(pos_dst_indices_lst, dim=0)
        n_d_r = torch.cat(neg_dst_indices_lst, dim=0)
        # pdb.set_trace()

        #CODES FOR TRACTED UPDATE
        s_store_eid = local_store_s[local_store_mask_src]#(b_nid[local_store_mask_src], local_store_s[local_store_mask_src])
        d_store_eid = local_store_d[local_store_mask_dst]#(b_nid[local_store_mask_dst], local_store_d[local_store_mask_dst])
        
        
        return memory, last_update, el_r, eid_r, s_r, p_d_r, n_d_r, lum_assoc, s_store_eid, d_store_eid

    def update_memory_DATGN(self, nid, memory, lum_assoc):
        # pdb.set_trace()
        self.memory[nid] = memory[lum_assoc]

    def update_state_DATGN(
            self, nid: Tensor, memory, lum_assoc,
            s_src: Tensor, s_dst: Tensor, s_t: Tensor, s_raw_msg: Tensor,
            d_src: Tensor, d_dst: Tensor, d_t: Tensor, d_raw_msg: Tensor,
        ):
        self.memory[nid] = memory[lum_assoc]
        # self.update_memory_DATGN(nid, memory, lum_assoc)
        self._update_msg_store(s_src, s_dst, s_t, s_raw_msg, self.msg_s_store)
        self._update_msg_store(d_dst, d_src, d_t, d_raw_msg, self.msg_d_store)


    def forward_legacy(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        # memory, last_update = self._get_updated_memory(n_id)
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


    def _get_updated_memory_local(
            self,
            n_id,
            n_id_mem,
            src,
            dst,
            src_memory,
            dst_memory,
            src_last_update,
            dst_last_update,
            raw_t,
            raw_msg,


    ):
        # n_id_2 = torch.cat([src, dst])
        # if n_id.shape != n_id_2.shape:
        #     pdb.set_trace
        # if not torch.equal(n_id_2, n_id):
        #     pdb.set_trace()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        
        msg_s, t_s, src_s, dst_s = self._compute_msg_local(
            raw_t, 
            src_last_update,
            src,
            dst,
            raw_msg,
            src_memory,
            dst_memory,
            self.msg_s_module,
        )

        msg_d, t_d, src_d, dst_d = self._compute_msg_local(
            raw_t, 
            dst_last_update,
            dst,
            src,
            raw_msg,
            dst_memory,
            src_memory,
            self.msg_d_module,
        )

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        # pdb.set_trace()
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # aggr =  torch.cat([aggr[self._assoc[src_s], aggr[self._assoc[src_s]]])

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, n_id_mem)

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]

        return memory, last_update


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

    

    def _compute_msg_local(
            self,
            src_t, 
            src_last_update,
            src,
            dst,
            raw_msg,
            src_memory,
            dst_memory,
            msg_module,
            
    ):
        t_rel = src_t - src_last_update
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = msg_module(src_memory, dst_memory, raw_msg, t_enc)
        return msg, src_t, src, dst


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











