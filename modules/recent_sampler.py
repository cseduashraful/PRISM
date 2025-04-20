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




class Recent_K_Sampler:
    def __init__(self, sampler_data, max_chunk_per_node, k, cache_size_limit=512):
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

        # self.sampler_data = sampler_data

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

        max_chunk_id = torch.max(all_needed_chunk_ids).item() + 1
        global_to_local = torch.full((max_chunk_id,), -1, dtype=torch.long, device=all_needed_chunk_ids.device)
        global_to_local[all_needed_chunk_ids] = torch.arange(all_needed_chunk_ids.size(0), device=all_needed_chunk_ids.device)

        chunk_ids_local = global_to_local[chunk_ids]

        index_and_flag = sampler.find_index_in_chunk(
            ts_chunks_selected[chunk_ids_local],
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

        collected_ts_indices = sampler.collect_prev_k_ts(
            chunk_ids_local,
            previous_chunk_ids_local,
            index_in_chunk,
            ts_chunks_selected,
            self.ts_chunks_all_cpu.shape[-1],
            k
        )

        eid_chunks_flattened = eid_chunks_selected.flatten()
        collected_eid_values = eid_chunks_flattened[collected_ts_indices]

        return collected_eid_values
