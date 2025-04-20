def main():
    import torch
    import preprocessor
    import sampler

    src = torch.tensor([1, 2, 3, 1, 2, 3, 4, 0, 1, 2, 0], dtype=torch.long)
    dst = torch.tensor([0, 0, 1, 2, 3, 4, 1, 2, 3, 2, 2], dtype=torch.long)
    ts = torch.tensor([20.1, 50.9, 60.11, 67.45, 89.28, 98.32, 112.89, 123.67, 145.52, 187.12, 190.5], dtype=torch.float64)
    eid = torch.arange(len(src), dtype=torch.long)

    num_nodes = 5
    chunk_size = 3
    max_chunk_per_node = 4
    k = 3

    src_list = src.tolist()
    dst_list = dst.tolist()
    ts_list = ts.tolist()
    eid_list = eid.tolist()

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

    print("ts_chunks:", ts_chunks)
    print("eid_chunks:", eid_chunks)
    print("chunk_map:", chunk_map)
    print("chunk_last_ts:", chunk_last_ts)

    root_node = torch.tensor([0, 1, 2, 3, 4, 1, 1, 0], dtype=torch.long, device='cuda')
    root_ts = torch.tensor([200.0, 115.0, 130.0, 140.0, 10.0, 56.0, 150.0, 10.0], dtype=torch.float64, device='cuda')

    max_chunk_per_node = 3

    chunk_map_tensor = torch.tensor(chunk_map, dtype=torch.long, device='cuda')
    chunk_last_ts_tensor = torch.tensor(chunk_last_ts, dtype=torch.float64, device='cuda')

    # Step 1: Find the latest chunk
    chunk_ids, previous_chunk_ids = sampler.find_chunk_from_last_ts(
        root_node,
        root_ts,
        chunk_map_tensor,
        chunk_last_ts_tensor,
        max_chunk_per_node
    )

    # Step 2: Load all CPU-side data
    ts_chunks_all_cpu = torch.tensor(ts_chunks, dtype=torch.float64)
    eid_chunks_all_cpu = torch.tensor(eid_chunks, dtype=torch.int64)

    # Step 3: Quickly get ts for chunk_ids
    ts_chunks_selected_cpu = ts_chunks_all_cpu[chunk_ids.cpu()]
    ts_chunks_selected = ts_chunks_selected_cpu.cuda(non_blocking=True)

    # Step 4: Find index in chunk
    index_and_flag = sampler.find_index_in_chunk(
        ts_chunks_selected,
        root_ts,
        chunk_size,
        k
    )
    index_in_chunk = index_and_flag[:, 0]
    is_previous_chunk_needed = index_and_flag[:, 1].bool()

    # Step 5: Only load previous chunks if needed
    needs_previous_mask = is_previous_chunk_needed
    valid_prev_chunk_ids = previous_chunk_ids[needs_previous_mask]
    valid_prev_chunk_ids = valid_prev_chunk_ids[valid_prev_chunk_ids != -1]
    valid_prev_chunk_ids = torch.unique(valid_prev_chunk_ids)

    # Step 6: Prepare list of all needed chunks
    all_chunk_ids = torch.cat([chunk_ids, valid_prev_chunk_ids], dim=0)
    valid_mask = all_chunk_ids != -1
    all_chunk_ids = all_chunk_ids[valid_mask]
    all_needed_chunk_ids, _ = torch.sort(torch.unique(all_chunk_ids))

    # Step 7: Ultra-fast gather directly to CUDA
    ts_chunks_selected = torch.index_select(
        ts_chunks_all_cpu,
        dim=0,
        index=all_needed_chunk_ids.cpu()
    ).to('cuda', non_blocking=True)

    eid_chunks_selected = torch.index_select(
        eid_chunks_all_cpu,
        dim=0,
        index=all_needed_chunk_ids.cpu()
    ).to('cuda', non_blocking=True)

    # Step 8: Build global-to-local chunk ID mapping
    max_chunk_id = torch.max(all_needed_chunk_ids).item() + 1
    global_to_local = torch.full((max_chunk_id,), -1, dtype=torch.long, device=all_needed_chunk_ids.device)
    global_to_local[all_needed_chunk_ids] = torch.arange(all_needed_chunk_ids.size(0), device=all_needed_chunk_ids.device)

    chunk_ids_local = global_to_local[chunk_ids]
    previous_chunk_ids_local = torch.where(
        (previous_chunk_ids != -1) & (is_previous_chunk_needed),
        global_to_local[previous_chunk_ids],
        torch.full_like(previous_chunk_ids, -1)
    )

    # Step 9: Collect k previous timestamps
    collected_ts_indices = sampler.collect_prev_k_ts(
        chunk_ids_local,
        previous_chunk_ids_local,
        index_in_chunk,
        ts_chunks_selected,
        chunk_size,
        k
    )

    # Step 10: Output
    print("Root nodes:", root_node.cpu().tolist())
    print("Root timestamps:", root_ts.cpu().tolist())
    print("Found chunk IDs:", chunk_ids.cpu().tolist())
    print("Previous chunk IDs:", previous_chunk_ids.cpu().tolist())
    print("Index inside chunks:", index_in_chunk.cpu().tolist())
    print("Is previous chunk needed:", is_previous_chunk_needed.cpu().tolist())
    print("Collected ts indices (global):", collected_ts_indices.cpu().tolist())

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

if __name__ == "__main__":
    main()
