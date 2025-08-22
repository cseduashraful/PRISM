import sampler
import torch
import chunkio
class GRN_Stream:
    def __init__(self, sampler_data, max_chunk_per_node, k, num_nodes, chunk_size, outdir, cache_size = 100, device='cuda', apan = False, skip_cnt = 1):
        self.device = device
        self.chunk_size = chunk_size
        self.max_chunk_per_node = max_chunk_per_node
        self.k = k
        self.num_nodes = num_nodes
        self.assoc = torch.arange(self.num_nodes, device = device)
        self.apan = apan
        if apan:
            self.skip_cnt = skip_cnt
        else:
            self.skip_cnt = 1
        self.cache_size = cache_size
        self.cache = chunkio.TorchChunkCache(outdir, self.chunk_size, self.cache_size)
        # Preprocess and store TCI structure on CPU
        self.chunk_map = torch.tensor(sampler_data.chunk_map, dtype=torch.long, device=device)
        self.chunk_last_ts = torch.tensor(sampler_data.chunk_last_ts, dtype=torch.float64, device=device)

        self.sampler_data = sampler_data
        # self.ts_chunks_all_cpu = torch.tensor(sampler_data['ts_chunks'], dtype=torch.float64, pin_memory=True)
        # self.eid_chunks_all_cpu = torch.tensor(sampler_data['eid_chunks'], dtype=torch.int64, pin_memory=True)
        # self.other_node_chunks_all_cpu = torch.tensor(sampler_data['other_node_chunks'], dtype=torch.int64, pin_memory=True)

        # CUDA streams
        self.prefetch_stream = torch.cuda.Stream(device=device)  # For background prefetching
        self.compute_stream = torch.cuda.default_stream(device=device)  # Main compute stream

        # Double buffer: keep two prefetch slots
        self.prefetch_buffer_ts = [None, None]
        self.prefetch_buffer_eid = [None, None]
        self.prefetch_buffer_other_node = [None, None]
        self.current_prefetch_idx = 0

        # self.data = data

    def _select_and_prefetch(self, root_node, root_ts):
        """Internal: Select chunks and async prefetch pinned CPU -> GPU"""

        # Step 1: Find chunk ids needed
        # breakpoint()
        # print("root node: ", root_node)
        # print("root ts: ", root_ts)
#         chunk_map = chunk_map.contiguous()
# chunk_last_ts = chunk_last_ts.contiguous()
        chunk_ids, previous_chunk_ids = sampler.find_chunk_from_last_ts(
            root_node,
            root_ts,
            self.chunk_map,
            self.chunk_last_ts,
            self.max_chunk_per_node
        )
        # breakpoint()
        # print("after")
        # print("previous_chunk_ids: ", previous_chunk_ids)
        # print("chunk_ids: ", chunk_ids)
        
        # breakpoint()
        # breakpoint()

        all_chunk_ids = torch.cat([chunk_ids, previous_chunk_ids], dim=0)
        valid_mask = all_chunk_ids != -1
        all_chunk_ids = all_chunk_ids[valid_mask]
        all_needed_chunk_ids, _ = torch.sort(torch.unique(all_chunk_ids))

        # Step 2: Create CPU slices
        ts_chunks_selected_cpu, eid_chunks_selected_cpu, other_node_chunks_selected_cpu = self.cache.get_chunks_torch(all_needed_chunk_ids.cpu())
        # ts_chunks_selected_cpu = self.ts_chunks_all_cpu[all_needed_chunk_ids.cpu()]
        # eid_chunks_selected_cpu = self.eid_chunks_all_cpu[all_needed_chunk_ids.cpu()]
        # other_node_chunks_selected_cpu = self.other_node_chunks_all_cpu[all_needed_chunk_ids.cpu()]

        # Step 3: Allocate prefetch slot
        slot = self.current_prefetch_idx

        # Step 4: Async copy CPU -> GPU using pinned memory + prefetch stream
        with torch.cuda.stream(self.prefetch_stream):
            self.prefetch_buffer_ts[slot] = ts_chunks_selected_cpu.to(self.device, non_blocking=True)
            self.prefetch_buffer_eid[slot] = eid_chunks_selected_cpu.to(self.device, non_blocking=True)
            self.prefetch_buffer_other_node[slot] = other_node_chunks_selected_cpu.to(self.device, non_blocking=True)

        # Step 5: Record needed mapping
        max_chunk_id = torch.max(all_needed_chunk_ids).item() + 1
        global_to_local = torch.full((max_chunk_id,), -1, dtype=torch.long, device=self.device)
        global_to_local[all_needed_chunk_ids] = torch.arange(all_needed_chunk_ids.size(0), device=self.device)

        return chunk_ids, previous_chunk_ids, global_to_local

    def sample(self, root_node, root_ts, k=None, neg_cnt = 1):
        """Main sampling call: will use prefetched data if available"""

        if k is None:
            k = self.k

        # Step 1: Prefetch next batch
        chunk_ids, previous_chunk_ids, global_to_local = self._select_and_prefetch(root_node, root_ts)

        # Step 2: Wait for previous prefetch stream to finish
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)

        # Step 3: Now use prefetched data
        slot = self.current_prefetch_idx
        ts_chunks_selected = self.prefetch_buffer_ts[slot]
        eid_chunks_selected = self.prefetch_buffer_eid[slot]
        other_node_chunks_selected = self.prefetch_buffer_other_node[slot]

        chunk_ids_local = global_to_local[chunk_ids]
        # print("chubk ids: ",chunk_ids)
        previous_chunk_ids_local = torch.where(
            previous_chunk_ids != -1,
            global_to_local[previous_chunk_ids],
            torch.full_like(previous_chunk_ids, -1)
        )
        # breakpoint()
        # Step 4: Fused find + collect sampling
        collected_ts_indices = sampler.fused_find_and_collect(
            ts_chunks_selected,
            chunk_ids_local,
            previous_chunk_ids_local,
            root_ts,
            ts_chunks_selected.size(-1),
            k
        )
        # breakpoint()
        eid_chunks_flattened = eid_chunks_selected.flatten()
        other_node_chunks_flattened = other_node_chunks_selected.flatten()
        # print("collected_ts_indices: ", collected_ts_indices)
        # print("eid_chunks_flattened: ", eid_chunks_flattened)

        # breakpoint()
        sampled_eids = eid_chunks_flattened[collected_ts_indices]
        sampled_eids[collected_ts_indices == -1] = -1
        # print("sampled eids: ", sampled_eids)
        # breakpoint()

        sampled_other_nodes = other_node_chunks_flattened[collected_ts_indices]
        sampled_other_nodes[collected_ts_indices == -1] = -1

        # Step 5: Rotate prefetch buffer for next batch
        self.current_prefetch_idx = (self.current_prefetch_idx + 1) % 2
        # print("here:  ", self.apan)
        if self.apan:
            return self.transform_eids_for_apan(sampled_eids, sampled_other_nodes, root_node, neg_cnt = neg_cnt)
        else:
            return self.transform_eids(sampled_eids, sampled_other_nodes, root_node, neg_cnt = neg_cnt)
    

    def find_recent_occurrences(self, edge_index, pos_node_s, pos_node_d, batch_size, assoc):
        recent_indices = []

        for i in range(edge_index.size(1)):
            target_node = edge_index[0, i].item()
            max_idx = edge_index[1, i % batch_size].item()

            found_idx = -1
            for j in reversed(range(min(max_idx, pos_node_s.size(0)))):
                if pos_node_s[j].item() == target_node:
                    found_idx = j
                    break
                if pos_node_d[j].item() == target_node:
                    found_idx = j+batch_size
                    break
            if found_idx == -1:
                found_idx = assoc[target_node]
            recent_indices.append(found_idx)

        return torch.tensor(recent_indices, device=edge_index.device)

    def transform_eids(self, sampled_eids, sampled_other_nodes, root_node, neg_cnt = 1):
        # breakpoint()
        batch_size, k = sampled_eids.shape
        # print("Sampled eids: ", sampled_eids)
        # Step 1: Flatten and find valid entries
        eids_flat = sampled_eids.view(-1)
        other_nodes_flat = sampled_other_nodes.view(-1)
        # print(eids_flat)

        valid_mask = (eids_flat != -1)
        valid_eids = eids_flat[valid_mask].cpu()
        # print("valid eids: ", valid_eids)
        valid_other_nodes = other_nodes_flat[valid_mask]

        # Step 2: Always update assoc mapping (even if already present)
        # on = valid_other_nodes.unique()
        on = torch.cat([sampled_other_nodes.view(-1)[valid_mask],root_node] ).unique()
        self.assoc[on] = torch.arange(root_node.shape[0], root_node.shape[0] + on.shape[0], device=self.device)

        # Step 3: Map sampled other nodes
        mapped_other_nodes = self.assoc[other_nodes_flat]

        # breakpoint()

        bs = root_node.shape[0]//(2+neg_cnt) 
        pos_node_s = root_node[:bs]
        pos_node_d = root_node[bs:2*bs]


        # Step 4: Build edge indices
        edge_index_dst = torch.arange(root_node.shape[0], device=self.device).unsqueeze(1).expand(root_node.shape[0], k)
        edge_index_dst = edge_index_dst.reshape(-1)
        # breakpoint()
        edge_index = torch.stack([mapped_other_nodes[valid_mask], edge_index_dst[valid_mask]], dim=0)

        # Step 5: Concatenate n_ids (root_node + all sampled other nodes ONCE)

        n_ids = torch.cat([root_node, on])
        return n_ids, valid_eids, edge_index

    def coalesce(self, row_node_pairs):
        # row_node_pairs: [N, 3]  (col1, col2, col3), on CUDA
        keys = row_node_pairs[:, 1:3]                             # (col2, col3)
        keys_unique, inv = torch.unique(keys, dim=0, return_inverse=True)

        vals = row_node_pairs[:, 0]                               # col1
        num_groups = keys_unique.size(0)

        # Use scatter-reduce to get per-group mins (fast + GPU-friendly)
        group_mins = torch.empty(num_groups, device=vals.device, dtype=vals.dtype)
        group_mins.fill_(torch.iinfo(vals.dtype).max)
        group_mins.scatter_reduce_(0, inv, vals, reduce='amin', include_self=True)

        # Write back: replace col1 by the min for its (col2,col3) group
        row_node_pairs[:, 0] = group_mins[inv]
        return row_node_pairs

    def map_valid(self, x, offset):
        # breakpoint()
        mask = x != -1
        new_x = torch.full_like(x, -1)

        # Get valid values and their row indices
        flat_vals = x[mask]
        row_idx = torch.arange(x.size(0), device=x.device).repeat_interleave(x.size(1))[mask.view(-1)]

        # Build [row_idx, node_id] pairs
        row_node_pairs = torch.stack([row_idx, flat_vals], dim=1)  # shape: [num_valid, 2]
        


        # breakpoint()
        # Get unique pairs and inverse indices
        

        # breakpoint()
        if self.skip_cnt>1:
            clone = row_node_pairs.clone()
            row_node_pairs[:, 0] = row_node_pairs[:, 0] // self.skip_cnt

            clone = torch.cat([clone, row_node_pairs], dim=1)
            clone = clone[:, :3]
            clone = self.coalesce(clone)
            row_node_pairs = clone[:, :2]

        unique_pairs, inverse = torch.unique(row_node_pairs, dim=0, return_inverse=True)

        # u, i = torch.unique(row_node_pairs, dim=0, return_inverse=True)
        # breakpoint()


        inverse = inverse + offset
        # Fill new_x with ID
        new_x[mask] = inverse

        # Outputs:
        # new_x → mapped tensor with IDs
        # unique_pairs → reverse mapping: id_to_pair[i] = [row_index, node_id]
        id_to_pair = unique_pairs

        # print("Remapped tensor:\n", new_x)
        # print("Reverse mapping (id → [row, node_id]):\n", id_to_pair)
        return new_x, id_to_pair
    
    def transform_eids_for_apan(self, sampled_eids, sampled_other_nodes, root_node, neg_cnt = 1):
        # breakpoint()
        batch_size, k = sampled_eids.shape

        # Step 1: Flatten and find valid entries
        eids_flat = sampled_eids.view(-1)
        # other_nodes_flat = sampled_other_nodes.view(-1)
        # print(eids_flat)

        valid_mask = (eids_flat != -1)
        valid_eids = eids_flat[valid_mask].cpu()
        # valid_other_nodes = other_nodes_flat[valid_mask]

        mapped_other_nodes, id_to_pair = self.map_valid(sampled_other_nodes, root_node.shape[0])
        mapped_other_nodes = mapped_other_nodes.view(-1)
        n_ids = torch.cat([root_node, id_to_pair[:, 1]])
        self.id_to_pair = id_to_pair
        self.offset =  root_node.shape[0]
        # breakpoint()

        # Step 2: Always update assoc mapping (even if already present)
        # on = valid_other_nodes.unique()
        # on = torch.cat([sampled_other_nodes.view(-1)[valid_mask],root_node] ).unique()
        # self.assoc[on] = torch.arange(root_node.shape[0], root_node.shape[0] + on.shape[0], device=self.device)

        # # Step 3: Map sampled other nodes
        # mapped_other_nodes = self.assoc[other_nodes_flat]

        # breakpoint()

        # bs = root_node.shape[0]//(2+neg_cnt) 
        # pos_node_s = root_node[:bs]
        # pos_node_d = root_node[bs:2*bs]


        # Step 4: Build edge indices
        edge_index_dst = torch.arange(root_node.shape[0], device=self.device).unsqueeze(1).expand(root_node.shape[0], k)
        edge_index_dst = edge_index_dst.reshape(-1)
        # breakpoint()
        edge_index = torch.stack([mapped_other_nodes[valid_mask], edge_index_dst[valid_mask]], dim=0)

        # Step 5: Concatenate n_ids (root_node + all sampled other nodes ONCE)

        # n_ids = torch.cat([root_node, on])
        return n_ids, valid_eids, edge_index




