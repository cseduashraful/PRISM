from modules.data_utils import read_data #, get_TCSR, get_TCSR_py, verify_tcsr
# from modules.neg_sampler import NegLinkSamplerDest
# from modules.emb_module import GraphAttentionEmbedding
# from modules.memory_module import DAATGNMemory
# from modules.early_stopping import EarlyStopMonitor
# from modules.msg_agg import LastAggregator, MeanAggregator
# from modules.msg_func import IdentityMessage, MLPMessage
# from modules.decoder import LinkPredictor
# from modules.train_utils import train, test
# from sampler_core import ParallelSampler


from tgb.utils.utils import get_args, set_random_seed, save_results
import numpy as np
import torch
import timeit
import os
import os.path as osp
from pathlib import Path

import preprocessor #openmp
import sampler #cuda
    

def main():


    src = torch.tensor([1, 2, 3, 1, 2, 3, 4, 0, 1, 2, 0], dtype=torch.long)
    dst = torch.tensor([0, 0, 1, 2, 3, 4, 1, 2, 3, 2, 2], dtype=torch.long)
    ts = torch.tensor([20.1, 50.9, 60.11, 67.45, 89.28, 98.32, 112.89, 123.67, 145.52, 187.12, 190.5], dtype=torch.float64)
    eid = torch.arange(len(src), dtype=torch.long)

    num_nodes = 5
    chunk_size = 3
    max_chunk_per_node = 4
    k = 3

    # Convert tensors to Python lists because our C++ expects std::vector
    src_list = src.tolist()
    dst_list = dst.tolist()
    ts_list = ts.tolist()
    eid_list = eid.tolist()

    # Call the C++ function
    output = preprocessor.preprocess(
        src_list,
        dst_list,
        ts_list,
        eid_list,
        num_nodes,
        chunk_size,
        max_chunk_per_node
    )



    ts_chunks = output['ts_chunks']
    eid_chunks = output['eid_chunks']
    chunk_map = output['chunk_map']
    chunk_last_ts = output['chunk_last_ts']

    # Initialize global_chunk_counter
    global_chunk_counter = len(ts_chunks)

    print("ts_chunks: ", ts_chunks)
    print("eid_chunks: ", eid_chunks)
    print("chunk_map: ", chunk_map)
    print("chunk_last_ts: ", chunk_last_ts)

    root_node = torch.tensor([0, 1, 2, 3, 4, 1, 1, 0], dtype=torch.long, device='cuda')
    root_ts = torch.tensor([200.0, 115.0, 130.0, 140.0, 10.0, 56.0, 150.0, 10.0], dtype=torch.float64, device='cuda')

    # root_node = torch.tensor([0], dtype=torch.long, device='cuda')
    # root_ts = torch.tensor([10.0], dtype=torch.float64, device='cuda')


     # max_chunk_per_node
    max_chunk_per_node = 3


    chunk_map_tensor = torch.tensor(chunk_map, dtype=torch.long, device='cuda')
    chunk_last_ts_tensor = torch.tensor(chunk_last_ts, dtype=torch.float64, device='cuda')

    # Run the CUDA kernel
    chunk_ids, previous_chunk_ids = sampler.find_chunk_from_last_ts(
        root_node,
        root_ts,
        chunk_map_tensor,
        chunk_last_ts_tensor,
        max_chunk_per_node
    )

    # print("Root nodes:", root_node.cpu().tolist())
    # print("Root timestamps:", root_ts.cpu().tolist())
    # print("Found chunk IDs:", chunk_ids.cpu().tolist())
    # print("previous chunk IDs:", previous_chunk_ids.cpu().tolist())


    # breakpoint()

    # --- Step 4: Optimized Gather of Required Chunks ---
    all_chunk_ids = torch.cat([chunk_ids, previous_chunk_ids], dim=0)
    valid_mask = all_chunk_ids != -1
    all_chunk_ids = all_chunk_ids[valid_mask]

    all_needed_chunk_ids, _ = torch.sort(torch.unique(all_chunk_ids))
    ts_chunks_all_cpu = torch.tensor(output['ts_chunks'], dtype=torch.float64)  # keep all ts_chunks on CPU
    eid_chunks_all_cpu = torch.tensor(output['eid_chunks'], dtype=torch.int64)  # keep all ts_chunks on CPU

    ts_chunks_selected_cpu = ts_chunks_all_cpu[all_needed_chunk_ids.cpu()]
    ts_chunks_selected = ts_chunks_selected_cpu.cuda()

    eid_chunks_selected_cpu = eid_chunks_all_cpu[all_needed_chunk_ids.cpu()]
    eid_chunks_selected = eid_chunks_selected_cpu.cuda()

    max_chunk_id = torch.max(all_needed_chunk_ids).item() + 1
    global_to_local = torch.full((max_chunk_id,), -1, dtype=torch.long, device=all_needed_chunk_ids.device)
    global_to_local[all_needed_chunk_ids] = torch.arange(all_needed_chunk_ids.size(0), device=all_needed_chunk_ids.device)

    chunk_ids_local = global_to_local[chunk_ids]
    # previous_chunk_ids_local = torch.where(
    #     previous_chunk_ids != -1,
    #     global_to_local[previous_chunk_ids],
    #     torch.full_like(previous_chunk_ids, -1)
    # )

    # previous_chunk_ids_local = global_to_local[previous_chunk_ids]

    # print("previous_chunk_ids_local: ", previous_chunk_ids_local)

    # index_and_flag = sampler.find_index_in_chunk(ts_chunks_selected[chunk_ids_local],  root_ts, self.ts_chunks_all_cpu.shape[-1], k)

    index_and_flag = sampler.find_index_in_chunk(
        ts_chunks_selected[chunk_ids_local],  # Select only current chunks
        root_ts,
        chunk_size,
        k
    )
    index_in_chunk = index_and_flag[:, 0]  # First column = best index
    print(index_in_chunk)
    print("ts_chunks_selected[chunk_ids_local]: ", ts_chunks_selected[chunk_ids_local])
    is_previous_chunk_needed = index_and_flag[:, 1].bool()  # Second column = True/False

    #previous_chunk_ids_local = torch.where((previous_chunk_ids != -1) & (is_previous_chunk_needed),global_to_local[previous_chunk_ids],torch.full_like(previous_chunk_ids, -1))
    previous_chunk_ids_local = torch.where(
        (previous_chunk_ids != -1) & (is_previous_chunk_needed),
        global_to_local[previous_chunk_ids],
        torch.full_like(previous_chunk_ids, -1)
    )

    # --- Step 6: Collect previous k timestamps ---
    #collected_ts_indices = sampler.collect_prev_k_ts(chunk_ids_local,previous_chunk_ids_local, index_in_chunk,ts_chunks_selected,self.ts_chunks_all_cpu.shape[-1],k)
    collected_ts_indices = sampler.collect_prev_k_ts(
        chunk_ids_local,
        previous_chunk_ids_local,
        index_in_chunk,
        ts_chunks_selected,
        chunk_size,
        k
    )

     # --- Step 7: Output Results ---
    print("Root nodes:", root_node.cpu().tolist())
    print("Root timestamps:", root_ts.cpu().tolist())
    print("Found chunk IDs:", chunk_ids.cpu().tolist())
    print("Previous chunk IDs:", previous_chunk_ids.cpu().tolist())
    print("Index inside chunks:", index_in_chunk.cpu().tolist())
    print("Collected ts indices (global):", collected_ts_indices.cpu().tolist())

    # breakpoint()
    # Optional: Gather actual timestamps
    ts_chunks_flattened = ts_chunks_selected.flatten()
    eid_chunks_flattened = eid_chunks_selected.flatten()
    
    collected_ts_values = []

    for indices in collected_ts_indices:
        values = []
        for idx in indices:
            if idx >= 0:
                values.append(ts_chunks_flattened[idx].item())
            else:
                values.append(None)
        collected_ts_values.append(values)

    print("Collected actual ts values:", collected_ts_values)

    collected_eid_values = []

    for indices in collected_ts_indices:
        values = []
        for idx in indices:
            if idx >= 0:
                values.append(eid_chunks_flattened[idx].item())
            else:
                values.append(None)
        collected_eid_values.append(values)

    print("Collected actual eid values:", collected_eid_values)



    breakpoint()


    


if __name__ == "__main__":
    main()