import sampler
import torch



    
class Recent_K_Sampler_old:
    def __init__(self, sampler_data,  max_chunk_per_node, k):

        self.chunk_map = torch.tensor(sampler_data['chunk_map'], dtype=torch.long, device='cuda')
        self.chunk_last_ts = torch.tensor(sampler_data['chunk_last_ts'], dtype=torch.float64, device='cuda')

        
        self.max_chunk_per_node = max_chunk_per_node
        self.k = k 

        self.ts_chunks_all_cpu = torch.tensor(sampler_data['ts_chunks'], dtype=torch.float64)
        self.eid_chunks_all_cpu = torch.tensor(sampler_data['eid_chunks'], dtype=torch.int64)  # keep all ts_chunks on CPU


        self.sampler_data = sampler_data

    def sample(self, root_node, root_ts, k = None):
        if k is None:
            k =  self.k
        chunk_ids, previous_chunk_ids = sampler.find_chunk_from_last_ts(
            root_node,
            root_ts,
            self.chunk_map,
            self.chunk_last_ts,
            self.max_chunk_per_node
        )
       
        all_chunk_ids = torch.cat([chunk_ids, previous_chunk_ids], dim=0)
        valid_mask = all_chunk_ids != -1
        all_chunk_ids = all_chunk_ids[valid_mask]
        all_needed_chunk_ids, _ = torch.sort(torch.unique(all_chunk_ids))

        ts_chunks_selected_cpu = self.ts_chunks_all_cpu[all_needed_chunk_ids.cpu()]
        ts_chunks_selected = ts_chunks_selected_cpu.cuda()
        max_chunk_id = torch.max(all_needed_chunk_ids).item() + 1
        global_to_local = torch.full((max_chunk_id,), -1, dtype=torch.long, device=all_needed_chunk_ids.device)
        global_to_local[all_needed_chunk_ids] = torch.arange(all_needed_chunk_ids.size(0), device=all_needed_chunk_ids.device)
        chunk_ids_local = global_to_local[chunk_ids]
        index_and_flag = sampler.find_index_in_chunk(ts_chunks_selected[chunk_ids_local],  root_ts, self.ts_chunks_all_cpu.shape[-1], k)
        index_in_chunk = index_and_flag[:, 0]  # First column = best index
        is_previous_chunk_needed = index_and_flag[:, 1].bool()  # Second column = True/False
        previous_chunk_ids_local = torch.where((previous_chunk_ids != -1) & (is_previous_chunk_needed),global_to_local[previous_chunk_ids],torch.full_like(previous_chunk_ids, -1))
        
        collected_ts_indices = sampler.collect_prev_k_ts(
            chunk_ids_local,
            previous_chunk_ids_local, 
            index_in_chunk,
            ts_chunks_selected,
            self.ts_chunks_all_cpu.shape[-1],
            k
        )
        
        eid_chunks_selected_cpu = self.eid_chunks_all_cpu[all_needed_chunk_ids.cpu()]
        eid_chunks_selected = eid_chunks_selected_cpu.cuda()
        eid_chunks_flattened = eid_chunks_selected.flatten()
        print(eid_chunks_flattened[collected_ts_indices])
        

        breakpoint()



class Recent_K_Sampler_no_cache:
    def __init__(self, sampler_data, max_chunk_per_node, k):
        self.chunk_map = torch.tensor(sampler_data['chunk_map'], dtype=torch.long, device='cuda')
        self.chunk_last_ts = torch.tensor(sampler_data['chunk_last_ts'], dtype=torch.float64, device='cuda')

        self.max_chunk_per_node = max_chunk_per_node
        self.k = k

        # ✅ Use pinned memory for faster async CPU → GPU copy
        self.ts_chunks_all_cpu = torch.tensor(sampler_data['ts_chunks'], dtype=torch.float64).pin_memory()
        self.eid_chunks_all_cpu = torch.tensor(sampler_data['eid_chunks'], dtype=torch.int64).pin_memory()

        # self.sampler_data = sampler_data

    def sample(self, root_node, root_ts, k=None):
        if k is None:
            k = self.k

        # Step 1: Find chunks (current and previous)
        chunk_ids, previous_chunk_ids = sampler.find_chunk_from_last_ts(
            root_node,
            root_ts,
            self.chunk_map,
            self.chunk_last_ts,
            self.max_chunk_per_node
        )

        # Step 2: Gather all needed chunk IDs
        all_chunk_ids = torch.cat([chunk_ids, previous_chunk_ids], dim=0)
        valid_mask = all_chunk_ids != -1
        all_chunk_ids = all_chunk_ids[valid_mask]
        all_needed_chunk_ids, _ = torch.sort(torch.unique(all_chunk_ids))

        # Step 3: Transfer needed ts_chunks and eid_chunks (non-blocking pinned transfer)
        ts_chunks_selected_cpu = self.ts_chunks_all_cpu[all_needed_chunk_ids.cpu()]
        eid_chunks_selected_cpu = self.eid_chunks_all_cpu[all_needed_chunk_ids.cpu()]

        ts_chunks_selected = ts_chunks_selected_cpu.cuda(non_blocking=True)
        eid_chunks_selected = eid_chunks_selected_cpu.cuda(non_blocking=True)

        max_chunk_id = torch.max(all_needed_chunk_ids).item() + 1
        global_to_local = torch.full((max_chunk_id,), -1, dtype=torch.long, device=all_needed_chunk_ids.device)
        global_to_local[all_needed_chunk_ids] = torch.arange(all_needed_chunk_ids.size(0), device=all_needed_chunk_ids.device)

        chunk_ids_local = global_to_local[chunk_ids]

        # Step 4: Find index within current chunk
        index_and_flag = sampler.find_index_in_chunk(
            ts_chunks_selected[chunk_ids_local],  # select only current chunks
            root_ts,
            self.ts_chunks_all_cpu.shape[-1],
            k
        )
        index_in_chunk = index_and_flag[:, 0]
        is_previous_chunk_needed = index_and_flag[:, 1].bool()

        previous_chunk_ids_local = torch.where(
            (previous_chunk_ids != -1) & (is_previous_chunk_needed),
            global_to_local[previous_chunk_ids],
            torch.full_like(previous_chunk_ids, -1)
        )

        # Step 5: Collect previous k timestamps
        collected_ts_indices = sampler.collect_prev_k_ts(
            chunk_ids_local,
            previous_chunk_ids_local,
            index_in_chunk,
            ts_chunks_selected,
            self.ts_chunks_all_cpu.shape[-1],
            k
        )

        # Step 6: Flatten eid_chunks and gather
        eid_chunks_flattened = eid_chunks_selected.flatten()
        collected_eid_values = eid_chunks_flattened[collected_ts_indices]
        print(eid_chunks_flattened[collected_ts_indices])

        return collected_eid_values




class Recent_K_Sampler_without_prefetching:
    def __init__(self, data, sampler_data, max_chunk_per_node, k, cache_size_limit=512):
        self.chunk_map = torch.tensor(sampler_data['chunk_map'], dtype=torch.long, device='cuda')
        self.chunk_last_ts = torch.tensor(sampler_data['chunk_last_ts'], dtype=torch.float64, device='cuda')

        self.max_chunk_per_node = max_chunk_per_node
        self.k = k

        # ✅ Use pinned CPU memory for async GPU transfer
        self.ts_chunks_all_cpu = torch.tensor(sampler_data['ts_chunks'], dtype=torch.float64).pin_memory()
        self.eid_chunks_all_cpu = torch.tensor(sampler_data['eid_chunks'], dtype=torch.int64).pin_memory()

        # ✅ Caching related
        self.chunk_cache = {}  # chunk_id -> ts_chunk (on GPU)
        self.eid_cache = {}    # chunk_id -> eid_chunk (on GPU)
        self.cache_size_limit = cache_size_limit  # limit how many chunks in cache

        self.sampler = data

    def _get_chunk(self, chunk_id):
        if chunk_id in self.chunk_cache:
            return self.chunk_cache[chunk_id], self.eid_cache[chunk_id]
        else:
            # Load and cache
            ts_chunk_cpu = self.ts_chunks_all_cpu[chunk_id]
            eid_chunk_cpu = self.eid_chunks_all_cpu[chunk_id]

            ts_chunk_gpu = ts_chunk_cpu.cuda(non_blocking=True)
            eid_chunk_gpu = eid_chunk_cpu.cuda(non_blocking=True)

            if len(self.chunk_cache) >= self.cache_size_limit:
                # Simple eviction: randomly pop one (can do LRU later if needed)
                self.chunk_cache.pop(next(iter(self.chunk_cache)))
                self.eid_cache.pop(next(iter(self.eid_cache)))

            self.chunk_cache[chunk_id] = ts_chunk_gpu
            self.eid_cache[chunk_id] = eid_chunk_gpu

            return ts_chunk_gpu, eid_chunk_gpu

    def sample(self, root_node, root_ts, k=None):
        if k is None:
            k = self.k

        chunk_ids, previous_chunk_ids = sampler.find_chunk_from_last_ts(
            root_node,
            root_ts,
            self.chunk_map,
            self.chunk_last_ts,
            self.max_chunk_per_node
        )

        all_chunk_ids = torch.cat([chunk_ids, previous_chunk_ids], dim=0)
        valid_mask = all_chunk_ids != -1
        all_chunk_ids = all_chunk_ids[valid_mask]
        all_needed_chunk_ids, _ = torch.sort(torch.unique(all_chunk_ids))

        # ✅ Gather the chunks you need (from cache or transfer)
        ts_chunks_list = []
        eid_chunks_list = []

        for chunk_id in all_needed_chunk_ids.tolist():
            ts_chunk_gpu, eid_chunk_gpu = self._get_chunk(chunk_id)
            ts_chunks_list.append(ts_chunk_gpu.unsqueeze(0))  # add batch dimension
            eid_chunks_list.append(eid_chunk_gpu.unsqueeze(0))

        # Stack into batch tensors
        ts_chunks_selected = torch.cat(ts_chunks_list, dim=0)  # [num_chunks, chunk_size]
        eid_chunks_selected = torch.cat(eid_chunks_list, dim=0)

        # max_chunk_id = torch.max(all_needed_chunk_ids).item() + 1
        # global_to_local = torch.full((max_chunk_id,), -1, dtype=torch.long, device=all_needed_chunk_ids.device)
        # global_to_local[all_needed_chunk_ids] = torch.arange(all_needed_chunk_ids.size(0), device=all_needed_chunk_ids.device)

        # chunk_ids_local = global_to_local[chunk_ids]

        # index_and_flag = sampler.find_index_in_chunk(
        #     ts_chunks_selected[chunk_ids_local],
        #     root_ts,
        #     self.ts_chunks_all_cpu.shape[-1],
        #     k
        # )
        # index_in_chunk = index_and_flag[:, 0]
        # is_previous_chunk_needed = index_and_flag[:, 1].bool()

        # previous_chunk_ids_local = torch.where(
        #     (previous_chunk_ids != -1) & (is_previous_chunk_needed),
        #     global_to_local[previous_chunk_ids],
        #     torch.full_like(previous_chunk_ids, -1)
        # )

        # collected_ts_indices = sampler.collect_prev_k_ts(
        #     chunk_ids_local,
        #     previous_chunk_ids_local,
        #     index_in_chunk,
        #     ts_chunks_selected,
        #     self.ts_chunks_all_cpu.shape[-1],
        #     k
        # )

        
        collected_ts_indices_2 = sampler.fused_find_and_collect(
            ts_chunks_selected,
            chunk_ids,
            previous_chunk_ids,
            root_ts,
            self.ts_chunks_all_cpu.shape[-1],
            k
        )
        # breakpbreakpointoint()
        eid_chunks_flattened = eid_chunks_selected.flatten()
        # collected_eid_values = eid_chunks_flattened[collected_ts_indices]
        collected_eid_values = eid_chunks_flattened[collected_ts_indices_2]
        # print(collected_eid_values == collected_eid_values_2)



        return collected_eid_values



class Recent_K_Sampler:
    def __init__(self, sampler_data, max_chunk_per_node, k, num_nodes, device='cuda'):
        self.device = device
        self.max_chunk_per_node = max_chunk_per_node
        self.k = k
        self.num_nodes = num_nodes
        self.assoc = torch.arange(self.num_nodes, device = device)

        # Preprocess and store TCI structure on CPU
        self.chunk_map = torch.tensor(sampler_data['chunk_map'], dtype=torch.long, device=device)
        self.chunk_last_ts = torch.tensor(sampler_data['chunk_last_ts'], dtype=torch.float64, device=device)

        self.ts_chunks_all_cpu = torch.tensor(sampler_data['ts_chunks'], dtype=torch.float64, pin_memory=True)
        self.eid_chunks_all_cpu = torch.tensor(sampler_data['eid_chunks'], dtype=torch.int64, pin_memory=True)
        self.other_node_chunks_all_cpu = torch.tensor(sampler_data['other_node_chunks'], dtype=torch.int64, pin_memory=True)

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
        ts_chunks_selected_cpu = self.ts_chunks_all_cpu[all_needed_chunk_ids.cpu()]
        eid_chunks_selected_cpu = self.eid_chunks_all_cpu[all_needed_chunk_ids.cpu()]
        other_node_chunks_selected_cpu = self.other_node_chunks_all_cpu[all_needed_chunk_ids.cpu()]

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
        previous_chunk_ids_local = torch.where(
            previous_chunk_ids != -1,
            global_to_local[previous_chunk_ids],
            torch.full_like(previous_chunk_ids, -1)
        )

        # Step 4: Fused find + collect sampling
        collected_ts_indices = sampler.fused_find_and_collect(
            ts_chunks_selected,
            chunk_ids_local,
            previous_chunk_ids_local,
            root_ts,
            ts_chunks_selected.size(-1),
            k
        )

        eid_chunks_flattened = eid_chunks_selected.flatten()
        other_node_chunks_flattened = other_node_chunks_selected.flatten()
        # print("collected_ts_indices: ", collected_ts_indices)
        # print("eid_chunks_flattened: ", eid_chunks_flattened)

        sampled_eids = eid_chunks_flattened[collected_ts_indices]
        sampled_eids[collected_ts_indices == -1] = -1

        sampled_other_nodes = other_node_chunks_flattened[collected_ts_indices]

        # Step 5: Rotate prefetch buffer for next batch
        self.current_prefetch_idx = (self.current_prefetch_idx + 1) % 2

        return self.transform_eids(sampled_eids, sampled_other_nodes, root_node, neg_cnt = neg_cnt)
    

    def transform_eids_old(self, sampled_eids, sampled_other_nodes, root_node):
        batch_size, k = sampled_eids.shape

        # Step 1: Flatten eids and filter valid
        eids = sampled_eids.view(-1)
        valid_mask = eids != -1
        valid_eids = eids[valid_mask].cpu()
        on = sampled_other_nodes.view(-1)[valid_mask].unique()
        self.assoc[on] = torch.arange(root_node.shape[0], root_node.shape[0]+on.shape[0], device = self.device)
        
        mapped_sampled_other_nodes  = self.assoc[sampled_other_nodes]
        edge_index_dst = torch.arange(root_node.shape[0], device=self.device).unsqueeze(1).expand(root_node.shape[0], k)

        edge_index = torch.stack([mapped_sampled_other_nodes.view(-1)[valid_mask],edge_index_dst.reshape(-1)[valid_mask]], dim=0)
        n_ids = torch.cat([root_node, on])

        return n_ids, valid_eids, edge_index

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

        # Step 1: Flatten and find valid entries
        eids_flat = sampled_eids.view(-1)
        other_nodes_flat = sampled_other_nodes.view(-1)
        # print(eids_flat)

        valid_mask = (eids_flat != -1)
        valid_eids = eids_flat[valid_mask].cpu()
        valid_other_nodes = other_nodes_flat[valid_mask]

        # Step 2: Always update assoc mapping (even if already present)
        on = valid_other_nodes.unique()
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

        # edge_index = torch.stack([other_nodes_flat[valid_mask], edge_index_dst[valid_mask]], dim=0)
        # recent = self.find_recent_occurrences(edge_index, pos_node_s, pos_node_d, bs, self.assoc)
        # edge_index = torch.stack([recent, edge_index_dst[valid_mask]], dim=0)
        
        # print(recent)
        # breakpoint()

        # Step 5: Concatenate n_ids (root_node + all sampled other nodes ONCE)

        n_ids = torch.cat([root_node, on])

        # x, y, z = self.transform_eids_old(sampled_eids, sampled_other_nodes, root_node)
        # if torch.all(x == n_ids) and torch.all(y == valid_eids) and torch.all(z == edge_index):
        #     print("OK")
        # else:
        #     breakpoint()
        # breakpoint()
        return n_ids, valid_eids, edge_index



