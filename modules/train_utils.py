import torch
import numpy as np
from tqdm import tqdm
# from apan_mem_wrapper import call_cuda_kernel as apan_g
import mem_update_graph

from modules.training_runtime import TrainRuntime


def _cached_or_host_lookup(cache_tensor, host_tensor, idx, device):
    idx_long = idx.long()
    if cache_tensor is not None:
        if idx_long.device != cache_tensor.device:
            idx_long = idx_long.to(cache_tensor.device)
        return cache_tensor[idx_long]
    return host_tensor[idx_long.cpu()].to(device)


def _cached_src_dirs(cache_src, host_src, eid_idx, n_id, store_quad, device):
    store_eid = eid_idx.long()
    if cache_src is not None:
        if store_eid.device != cache_src.device:
            store_eid = store_eid.to(cache_src.device)
        lhs = cache_src[store_eid]
        rhs = n_id[store_quad[0]]
        if rhs.device != lhs.device:
            rhs = rhs.to(lhs.device)
        return lhs == rhs
    store_eid_cpu = store_eid.cpu()
    return host_src[store_eid_cpu] == n_id[store_quad[0]].cpu()


def train_neg_sampler(min_dst_idx, max_dst_idx, pos_dst, device, neg_sampler, known_dsts = None):
    bs = pos_dst.shape[0]
    if known_dsts is not None and neg_sampler is None:
        # breakpoint()
        neg_dst = torch.randint(
                0,
                known_dsts.size(0),
                (bs,),
                dtype=torch.long,
        )
        neg_dst = known_dsts[neg_dst].to(device)
        # print(neg_dst)
    elif neg_sampler is None:
        neg_dst = torch.randint(
                min_dst_idx,
                max_dst_idx + 1,
                (bs,),
                dtype=torch.long,
                device=device,
            )
    else:
        neg_dst = neg_sampler.sample(pos_dst.cpu()).to(device)
    return neg_dst

def get_fixed_neighbors(merged, edge_index, n_id):
    edge_src, edge_tgt = edge_index  # shape [2, E]
    neighbor_dict = {}

    for node_id, idx in merged.items():
        # Mask where edge_index[1] == idx
        mask = (edge_tgt == idx)
        matched_src = edge_src[mask]

        # Resolve real node IDs
        real_neighbors = n_id[matched_src].tolist()

        # # Pad or truncate to length k
        # if len(real_neighbors) < k:
        #     real_neighbors += [-1] * (k - len(real_neighbors))
        # else:
        #     real_neighbors = real_neighbors[:k]

        neighbor_dict[node_id] = real_neighbors

    return neighbor_dict

def latest_index_per_node(src: torch.Tensor):
    # Step 1: Flip to find the last occurrence
    flipped_src = src.flip(0)  # reverse the tensor
    unique_nodes, inverse_idx = flipped_src.unique(return_inverse=True)
    latest_indices_in_flipped = inverse_idx.flip(0)

    # Step 2: Convert to original indices
    latest_idx_dict = {}
    for i, node in enumerate(unique_nodes):
        # Get the position of the latest occurrence
        original_idx = len(src) - 1 - (flipped_src == node).nonzero(as_tuple=False)[0].item()
        latest_idx_dict[node.item()] = original_idx

    return latest_idx_dict

def get_latest_neighbors_per_node(src, pos_dst, t, edge_index, n_id):
    # Step 1: Combine and find latest occurrence index per node
    node_list = torch.cat([src, pos_dst])  # e.g. [0,1,1,2,...]
    ts = torch.cat([t,t])
    all_indices = torch.arange(len(node_list), device=node_list.device)
    src_i = latest_index_per_node(src)
    dst_i = latest_index_per_node(pos_dst)
    # merged = {**src_i, **{k: max(v, src_i[k]) if k in src_i else v for k, v in dst_i.items()}}
    merged = {
        k: (
            max(v + src.shape[0], src_i[k]) if k in src_i and v > src_i[k] else
            src_i[k] if k in src_i else v + src.shape[0]
        )
        for k, v in dst_i.items()
    }
    merged.update({k: v for k, v in src_i.items() if k not in merged})
    x = torch.cat([pos_dst, src])
    result = {k: x[v].item() for k, v in merged.items()}
    neigh = get_fixed_neighbors(merged, edge_index, n_id)
    for k, v in result.items():
        neigh[k].append(v)
        neigh[k] = list(set(neigh[k]))   
    return neigh


# def apan_cuda_launcher(od_updated, bs, max_seen_eid, match_dict, device):
#     match_ptr = [0]
#     match_indices = []
#     max_key = max(match_dict.keys())
#     for k in range(max_key + 1):
#         match_list = match_dict.get(k, [])
#         match_indices.extend(match_list)
#         match_ptr.append(len(match_indices))
#     breakpoint()
#     match_ptr = torch.tensor(match_ptr, device=device, dtype=torch.long)
#     match_indices = torch.tensor(match_indices, device=device, dtype=torch.long)

#     threads_per_block = 256
#     blocks = (od_updated.size(0) + threads_per_block - 1) // threads_per_block

#     max_edges = od_updated.size(0) * 10
#     mem_graph_out = torch.zeros((4, max_edges), dtype=torch.long, device=device)
#     store_quad_out = torch.zeros((4, od_updated.size(0)), dtype=torch.long, device=device)
#     edge_counter = torch.zeros(1, dtype=torch.int32, device=device)
#     msg_counter = torch.zeros(1, dtype=torch.int32, device=device)


#     mem_update_graph.apan_mem_graph(
#         od_updated.contiguous(),  # arg0: torch.Tensor [N, 5]
#         match_ptr,                # arg1
#         match_indices,            # arg2
#         bs,                       # arg3: int
#         max_seen_eid,             # arg4: int
#         mem_graph_out,            # arg5: torch.Tensor [4, max_edges]
#         store_quad_out,           # arg6: torch.Tensor [4, max_msgs]
#         edge_counter,             # arg7: torch.int32
#         msg_counter               # arg8: torch.int32
#     )
#      # Step 4: Slice output using counters
#     e_cnt = edge_counter.item()
#     m_cnt = msg_counter.item()
#     mem_graph_final = mem_graph_out[:, :e_cnt]
#     store_quad_final = store_quad_out[:, :m_cnt]

#     return mem_graph_final, store_quad_final





def getMatchDict(od_updated):
    od = od_updated[:,:2]
    valid_vals = od[od[:, 1] != -1, 1]  # Filter out -1s from column 1

            # Get the indices for each valid value in col 1 that appear in col 0
            # matches = [(val.item(), (od_updated[:, 2] == val).nonzero(as_tuple=True)[0]) for val in valid_vals]
    match_dict = {val.item(): (od_updated[:, 2] == val).nonzero(as_tuple=True)[0] for val in valid_vals}
    return match_dict

# import torch

def vectorized_getMem_graph(od_updated, bs, max_seen_eid):
    
    # breakpoint()
    N = od_updated.shape[0]
    conds = od_updated[:, 0]
    keys = od_updated[:, 1]
    match_vals = od_updated[:, 2]
    bidxs = od_updated[:, 3]
    # breakpoint()
    
    valid_mask = keys != -1
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze()

    valid_keys = keys[valid_mask]        # shape: [M]
    valid_bidxs = bidxs[valid_mask]      # shape: [M]
    match_vals_all = match_vals.unsqueeze(0)  # shape: [1, N]
    key_vals_all = valid_keys.unsqueeze(1)    # shape: [M, 1]



    # kept_i_valid, kept_i_match = mem_update_graph.find_matches(
    #     valid_keys, valid_bidxs, valid_indices, match_vals, bidxs
    # )


    # kept_i_valid, kept_i_match = mem_update_graph.find_matches(valid_keys, valid_bidxs, match_vals, bidxs)
    # breakpoint()
    # Compare every valid key to all match_vals
    # breakpoint()
    chunk_size = 8192#int(280000000/match_vals_all.shape[1])#32768#16384
    # print("loops {}".format(match_vals_all.shape[1]/chunk_size))
    # print(chunk_size)
    # breakpoint()
    start = 0
    kv_list = []
    km_list = []
    while start < key_vals_all.shape[0]:
        end = min(start+chunk_size, key_vals_all.shape[0])
        partial_key = key_vals_all[start:end,]
        partial_match_mask = (match_vals_all == partial_key)
        # breakpoint()
        partial_dla_indices_per_row = partial_match_mask.nonzero(as_tuple=False)
        partial_i_valid = partial_dla_indices_per_row[:, 0] + start  # which row in valid_keys
        partial_i_match = partial_dla_indices_per_row[:, 1]  # which match index
        partial_ref_bidxs = valid_bidxs[partial_i_valid]
        partial_match_bidxs = bidxs[partial_i_match]
        partial_keep_mask = partial_match_bidxs > partial_ref_bidxs
        partial_kept_i_valid = partial_i_valid[partial_keep_mask]
        partial_kept_i_match = partial_i_match[partial_keep_mask]
        kv_list.append(partial_kept_i_valid)
        km_list.append(partial_kept_i_match)
        start = end

    # match_mask = (match_vals_all == key_vals_all)  # shape: [M, N]

    # # breakpoint()
    # # Get dla indices per valid row
    # dla_indices_per_row = match_mask.nonzero(as_tuple=False)  # shape: [*, 2], [i_valid, match_idx]
    # i_valid = dla_indices_per_row[:, 0]  # which row in valid_keys
    # i_match = dla_indices_per_row[:, 1]  # which match index

    # # key_vals_all_f = key_vals_all.flatten()
    # # match_vals_all_f = match_vals_all.flatten()

    # # i_valid_new, match_idx = mem_update_graph.match_indices(key_vals_all_f, match_vals_all_f, 100000)
    
    # # breakpoint()

    # # Get the bidxs to compare against
    # ref_bidxs = valid_bidxs[i_valid]     # shape: [*,]

    # # Now get bidxs for the matched rows
    # match_bidxs = bidxs[i_match]         # shape: [*,]

    # # Keep only where match_bidx > ref_bidx
    # keep_mask = match_bidxs > ref_bidxs


    # kept_i_valid_2 = i_valid[keep_mask]
    # kept_i_match_2 = i_match[keep_mask]

    kept_i_valid = torch.cat(kv_list, dim=0)
    kept_i_match = torch.cat(km_list, dim=0)
    
    # if(torch.equal(kept_i_match_2, kept_i_match)) and torch.equal(kept_i_valid, kept_i_valid_2):
    #     print("Same")
    # else:
    #     print("different")
    
 


    # breakpoint()
    M = valid_keys.size(0)
    has_valid_dla = torch.zeros(M, dtype=torch.bool, device=od_updated.device)
    has_valid_dla.index_fill_(0, kept_i_valid, True)

    no_dla_mask = ~has_valid_dla
    no_dla_indices = valid_indices[no_dla_mask]  # original od_updated row indices

    store_cond = conds[no_dla_indices]
    store_key = keys[no_dla_indices]
    store_bidx = bidxs[no_dla_indices]

    store_dst = torch.where(store_cond < bs, store_cond + bs, store_cond % bs)
    store_eid = store_bidx + max_seen_eid + 1

    store_quad = torch.stack([store_cond, store_dst, store_key, store_eid], dim=0)



    # You can now form e_src, e_dst, e_dla, etc. using od_updated[kept_i_match]
    # Example:
    e_src = conds[valid_indices[kept_i_valid]]
    cond = e_src#conds[valid_indices[kept_i_valid]]
    # e_dst_bs = e_src+bs
    e_dst = torch.where(cond < bs, cond + bs, cond % bs) #e_dst_bs % (2*bs) #keys[valid_indices[kept_i_valid]] #cond + bs if cond < bs else cond % bs
    # breakpoint()

    e_dla = kept_i_match
    # e_eid = valid_indices[kept_i_valid]
    e_eid = bidxs[valid_indices[kept_i_valid]] + (1+max_seen_eid)
    # breakpoint()
    mem_graph_quad = torch.stack([e_src, e_dst, e_dla, e_eid])


    return mem_graph_quad, store_quad


# def getMem_graph_apan_store(od_updated, bs, max_seen_eid, device):

#     match_dict = getMatchDict(od_updated)
#     msg_store_src, msg_store_dst, msg_store_nid, msg_store_eid = [], [], [], []

#     keys = od_updated[:, 1]
#     conds = od_updated[:, 0]
#     bidxs = od_updated[:, 3]
#     for i in range(od_updated.shape[0]):
#         key = keys[i].item()
#         if key == -1:
#             continue
#         cond = conds[i].item()
#         batch_edge_index = bidxs[i].item()

#         # msk = od_updated[:,2]==key
#         # dla = info[msk]
#         dla = match_dict[key]

#         valid_dla_mask = od_updated[dla, 3] > batch_edge_index
#         valid_dla = dla[valid_dla_mask]
#         # If no valid dla, append to msg_store
#         if valid_dla.shape[0] == 0:
#             msg_store_nid.append(key)
#             msg_store_eid.append(batch_edge_index + max_seen_eid + 1)
#             msg_store_src.append(cond)
#             msg_store_dst.append(cond + bs if cond < bs else cond % bs)
#             continue

#     store_quad = torch.tensor([msg_store_src, msg_store_dst, msg_store_nid, msg_store_eid], device=device, dtype=torch.long)
    
#     return store_quad




# def getMem_graph_apan_v2(od_updated, bs, max_seen_eid, device):
#     # info = torch.arange(od_updated.size(0), device = od_updated.device)
#     # breakpoint()
    
#     match_dict = getMatchDict(od_updated)
#     # breakpoint()
#     e_src, e_dst, e_dla, e_eid = [], [], [], []
#     msg_store_src, msg_store_dst, msg_store_nid, msg_store_eid = [], [], [], []

#     keys = od_updated[:, 1]
#     conds = od_updated[:, 0]
#     bidxs = od_updated[:, 3]
#     for i in range(od_updated.shape[0]):
#         key = keys[i].item()
#         if key == -1:
#             continue


#         cond = conds[i].item()
#         batch_edge_index = bidxs[i].item()

#         # msk = od_updated[:,2]==key
#         # dla = info[msk]
#         dla = match_dict[key]

#         valid_dla_mask = od_updated[dla, 3] > batch_edge_index
#         valid_dla = dla[valid_dla_mask]
        

#         # If no valid dla, append to msg_store
#         if valid_dla.shape[0] == 0:
#             msg_store_nid.append(key)
#             msg_store_eid.append(batch_edge_index + max_seen_eid + 1)
#             msg_store_src.append(cond)
#             msg_store_dst.append(cond + bs if cond < bs else cond % bs)
#             continue

#         # Efficient vectorized creation
#         n = valid_dla.shape[0]
#         dev = valid_dla.device
#         eid_val = batch_edge_index + max_seen_eid + 1
#         dst_val = cond + bs if cond < bs else cond % bs

#         e_src.append(torch.full((n,), cond, device=dev))
#         e_dst.append(torch.full((n,), dst_val, device=dev))
#         e_dla.append(valid_dla)
#         e_eid.append(torch.full((n,), eid_val, device=dev))

#     # Final concatenation
#     e_src = torch.cat(e_src) if e_src else torch.tensor([], device=od_updated.device)
#     e_dst = torch.cat(e_dst) if e_dst else torch.tensor([], device=od_updated.device)
#     e_dla = torch.cat(e_dla) if e_dla else torch.tensor([], device=od_updated.device)
#     e_eid = torch.cat(e_eid) if e_eid else torch.tensor([], device=od_updated.device)
#     # if torch.equal(e_src, e_src2) and torch.equal(e_dst, e_dst2) and torch.equal(e_dla, e_dla2) and torch.equal(e_eid, e_eid2):
#     #     print("okay")
#     # else:
#     #     breakpoint()

    
#     mem_graph_quad = torch.stack([e_src, e_dst, e_dla, e_eid])
#     store_quad = torch.tensor([msg_store_src, msg_store_dst, msg_store_nid, msg_store_eid], device=device, dtype=torch.long)
    
#     # if torch.equal(s_quad, store_quad):
#     #     print("store okay")
#     # else:
#     #     breakpoint()

#     # if torch.equal(m_quad, mem_graph_quad):
#     #     print("mem okay")
#     # else:
#     #     breakpoint()
#     # # breakpoint()
#     # x, y = apan_cuda_launcher(od_updated, bs, max_seen_eid, match_dict, device)
#     # breakpoint()
#     return mem_graph_quad, store_quad





def getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, src, pos_dst, device):
    bmsk = e_id>max_seen_eid
    bdst = torch.arange(bs*2)
    bsrc = torch.cat([torch.arange(bs, bs*2), torch.arange(bs)])
    bedge = torch.stack([bsrc, bdst]).to(device)

    bedge_all = model['memory'].mem_graph(n_id[torch.cat([bedge[0,:], bedge[1,:]])],torch.cat([bedge[1,:], bedge[1,:]]) , src, pos_dst)
    

  
    
    fall_back = neighbor_loader.assoc[n_id[torch.cat([bedge[0,:], bedge[1,:]])]]

    updated_ball = torch.where(bedge_all != -1, bedge_all, fall_back)
    mem_graph_triplet = torch.stack([updated_ball[:bedge.shape[1]], updated_ball[bedge.shape[1]:], bedge[1]])

    b_edge_index = edge_index[:,bmsk]
    new_eids = torch.arange(max_seen_eid+1, max_seen_eid+1+bs)

    mem_eid = torch.cat([new_eids, new_eids, e_id[bmsk]])
    del_addr = torch.cat([mem_graph_triplet[2, :], b_edge_index[1]])
    relative_mem_id = mem_eid - (max_seen_eid + 1)
    # breakpoint()
    mem_graph_quad_tmp =  torch.vstack([mem_graph_triplet[:2,relative_mem_id], del_addr, mem_eid.to(device)])
    direction = del_addr<bs
    src_part = mem_graph_quad_tmp[:, direction]
    dst_part = mem_graph_quad_tmp[:, ~direction]
    dst_part[[0, 1]] = dst_part[[1, 0]]
    mem_graph_quad = torch.cat([src_part, dst_part], dim=1)

    return mem_graph_quad


def filter_mem_graph(mem_graph_quad_uf, remap):
    used = remap.unique()
    mask = torch.isin(mem_graph_quad_uf[2], used)
    return  mem_graph_quad_uf[:, mask]
    
def append_unique_rows(od, new_rows):
    # Ensure both inputs are 2D with shape [N, 2]
    assert od.shape[1] == 2 and new_rows.shape[1] == 2

    # Compute scalar keys for comparison
    max_b = torch.cat([od[:, 1], new_rows[:, 1]]).max() + 1
    od_keys = od[:, 0] * max_b + od[:, 1]
    new_keys = new_rows[:, 0] * max_b + new_rows[:, 1]

    # Find new entries not already in od
    mask = ~torch.isin(new_keys, od_keys)

    return torch.cat([od, new_rows[mask]], dim=0)


def train(runtime: TrainRuntime, max_seen_id):
    model_bundle = runtime.model_bundle
    model = model_bundle.model
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    model['memory'].reset_state()

    optimizer = model_bundle.optimizer
    criterion = model_bundle.criterion

    dataset_runtime = runtime.dataset_runtime
    dataset = dataset_runtime.dataset
    data = dataset_runtime.data
    data_cache = dataset_runtime.data_cache or {}
    cached_t = data_cache.get('t')
    cached_msg = data_cache.get('msg')
    cached_src = data_cache.get('src')
    train_loader = dataset['train_dataloader']
    device = runtime.device
    min_dst_idx = dataset_runtime.min_dst_idx
    max_dst_idx = dataset_runtime.max_dst_idx
    neighbor_loader = runtime.sampler_runtime.sampler

    neg_sampler = runtime.neg_sampler
    known_dsts = dataset_runtime.known_dsts
    deliver_to = runtime.deliver_to
    decoder = runtime.decoder
    embedding = runtime.embedding


    total_loss = 0
    max_seen_eid = max_seen_id

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        bs  = src.shape[0]

        neg_dst = train_neg_sampler(min_dst_idx, max_dst_idx, pos_dst, device, neg_sampler, known_dsts = known_dsts)
        # neg_dst = torch.randint(
        #     min_dst_idx,
        #     max_dst_idx + 1,
        #     (src.size(0),),
        #     dtype=torch.long,
        #     device=device,
        # )
        root_ts = torch.cat([t, t, t], dim = 0).double()
        root_nodes = torch.cat([src, pos_dst, neg_dst], dim = 0)
        n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
        # breakpoint()
        if decoder == 'NCN':
            nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
            nid_ts[:root_nodes.shape[0]] = root_ts
            n_id, e_id, edge_index = neighbor_loader.sample(n_id, nid_ts)
        # breakpoint()
        if deliver_to == "neighbor":
            # breakpoint()
            # neighbors = get_latest_neighbors_per_node(src, pos_dst, t, edgrooe_index, n_id)
            bsrc = torch.stack([torch.arange(src.shape[0]).to(device), pos_dst])
            bdst = torch.stack([torch.arange(src.shape[0], 2*src.shape[0]).to(device), src])
            bndst = torch.stack([torch.arange(2*src.shape[0], 3*src.shape[0]).to(device), torch.full((src.shape[0],), -1).to(device)])
            
            od = torch.cat([ bsrc.T, bdst.T, bndst.T, neighbor_loader.id_to_pair], dim=0)
            od_updated  = torch.cat([od, n_id.unsqueeze(1)], dim=1)
            tmp = od_updated[:, 0] % bs
            od_updated = torch.cat([od_updated, tmp.unsqueeze(1)], dim=1)
            # breakpoint()
            # mem_graph_quad_v2, store_quad_v2 = getMem_graph_apan_v2(od_updated,bs, max_seen_eid, device)
            mem_graph_quad, store_quad = vectorized_getMem_graph(od_updated, bs, max_seen_eid)
            # mem_graph_quad, store_quad = model['memory'].mem_graph(od_updated,bs, max_seen_eid)
            
            # if torch.equal(mem_graph_quad, mem_graph_quad_v2) and torch.equal(store_quad, store_quad_v2):
            #     print("Same")
            # else:
            #     breakpoint()
            # print("Memgraph Constructed")
            # breakpoint()
            b_eid = mem_graph_quad[3]#e_id[bmsk]
            b_eid_cpu = b_eid.cpu().long()
            # print(b_eid_cpu)
            # breakpoint()
            b_t = _cached_or_host_lookup(cached_t, data.t, b_eid_cpu, device)
            b_raw_msg = _cached_or_host_lookup(cached_msg, data.msg, b_eid_cpu, device)
            # z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg, data = dataset['data'])




            # Get unique (src, dst, eid) triples
            triplet_keys = mem_graph_quad[[0, 1, 3]].T  # shape: [N, 3]
            unique_keys, inverse_indices = torch.unique(triplet_keys, dim=0, return_inverse=True)

            # Use unique EIDs to extract just the raw messages and timestamps once
            unique_eids = unique_keys[:, 2]  # this is just eid column
            b_t_unique = _cached_or_host_lookup(cached_t, data.t, unique_eids, device)
            b_raw_msg_unique = _cached_or_host_lookup(cached_msg, data.msg, unique_eids, device)
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=data)
                







            # breakpoint()
            # print("Updated memory Computed")
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                _cached_or_host_lookup(cached_t, data.t, e_id, device),
                _cached_or_host_lookup(cached_msg, data.msg, e_id, device),
            )
            # breakpoint()
            # print("Embedding generated")
            pos_out = model['link_pred'](z[0:bs], z[bs:2*bs])
            neg_out = model['link_pred'](z[0:bs], z[2*bs:3*bs])


            # breakpoint()
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))
            # print("batch loss computed")
            loss.backward()
            optimizer.step()

            # z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg)
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=data)
            
            store_eid = store_quad[3]
            # breakpoint()
            dirs = _cached_src_dirs(cached_src, data.src, store_eid, n_id, store_quad, device)
            
            model['memory'].update_state(n_id, z, last_update, store_quad, dirs.to(device))
            # print("State Updated")
            model['memory'].detach()
            total_loss += float(loss) * batch.num_events
            


        else:
            mem_graph_quad_uf  = getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, src, pos_dst, device)
            mem_graph_quad = mem_graph_quad_uf
            # breakpoint()

            remap_partial = model['memory'].mem_graph(n_id[:3*bs],torch.arange(3*bs).to(device) , src, pos_dst)
            remap_fall_back = neighbor_loader.assoc[n_id[:3*bs]]
            remap = torch.where(remap_partial != -1, remap_partial, remap_fall_back)

            # filtered =  False
            # if filtered:
            #     mem_graph_quad = filter_mem_graph(mem_graph_quad_uf, remap)
            # else:
            #     mem_graph_quad = mem_graph_quad_uf
            
            # from exclude.longdp import max_depth
            # max_depth(mem_graph_quad, remap)
            # breakpoint()
            

            # breakpoint()
            b_eid = mem_graph_quad[3]#e_id[bmsk]
            b_eid_cpu = b_eid.cpu()
            b_t = _cached_or_host_lookup(cached_t, data.t, b_eid_cpu, device)
            b_raw_msg = _cached_or_host_lookup(cached_msg, data.msg, b_eid_cpu, device)
            if cached_src is not None:
                b_eid_idx = b_eid.long()
                if b_eid_idx.device != cached_src.device:
                    b_eid_idx = b_eid_idx.to(cached_src.device)
                lhs_src = cached_src[b_eid_idx]
                rhs_src = n_id[mem_graph_quad[1]]
                if rhs_src.device != lhs_src.device:
                    rhs_src = rhs_src.to(lhs_src.device)
                b_isrc = rhs_src == lhs_src
            else:
                b_isrc = n_id[mem_graph_quad[1]].cpu() == data.src[b_eid_cpu]

            z, last_update = model['memory'](n_id, mem_graph_quad[0:2,:], b_t, b_raw_msg, b_isrc, delivery_addr = mem_graph_quad[2])
            z = torch.cat([z[remap], z[3*bs:]])
            last_update = torch.cat([last_update[remap], last_update[3*bs:]])
            
            # breakpoint()
            ei_src_all = model['memory'].mem_graph(n_id[edge_index[0,:]],edge_index[1,:] , src, pos_dst)

            updated_src = torch.where(ei_src_all != -1, ei_src_all, edge_index[0, :])
            edge_index = torch.stack([updated_src, edge_index[1,:]])
            # breakpoint()
            # if edge_index.max()>=z.size(0):
            #     breakpoint()
            if embedding == "time_emb":
                nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
                nid_ts[:root_nodes.shape[0]] = root_ts
                z = model['gnn'](
                    z,
                    last_update,
                    nid_ts
                )
            else:
                z = model['gnn'](
                    z,
                    last_update,
                    edge_index,
                    _cached_or_host_lookup(cached_t, data.t, e_id, device),
                    _cached_or_host_lookup(cached_msg, data.msg, e_id, device),
                )
            # breakpoint()
            if decoder == "NCN":
                # pos_out = model['link_pred'](z[0:bs], z[bs:2*bs])
                # neg_out = model['link_pred'](z[0:bs], z[2*bs:3*bs])

                time_info = (last_update, t)
                src_re = torch.arange(bs)
                pos_re = torch.arange(bs, 2*bs)
                neg_re = torch.arange(2*bs, 3*bs)
                pos_out = model['link_pred'](z, edge_index, torch.stack([src_re,pos_re]), 2, cn_time_decay=False, time_info=time_info)
                neg_out = model['link_pred'](z, edge_index, torch.stack([src_re,neg_re]), 2, cn_time_decay=False, time_info=time_info)

            else:
                pos_out = model['link_pred'](z[0:bs], z[bs:2*bs])
                neg_out = model['link_pred'](z[0:bs], z[2*bs:3*bs])


            # breakpoint()
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            loss.backward()
            optimizer.step()
            # model['memory'].update_state(src, pos_dst, t, msg)
            z_m, last_update = model['memory'](n_id, mem_graph_quad[0:2,:], b_t, b_raw_msg, b_isrc, delivery_addr = mem_graph_quad[2])
            #model['memory'](n_id, b_edge_index, b_t, b_raw_msg, b_isrc)
            z_m = torch.cat([z_m[remap], z_m[3*bs:]])
            last_update = torch.cat([last_update[remap], last_update[3*bs:]])

            model['memory'].update_state_v2(
                mem_graph_quad_uf[2], remap, bs, 
                src, pos_dst, t, msg, 
                n_id, last_update, z_m)
            model['memory'].detach()

            total_loss += float(loss) * batch.num_events
        
        
        
        max_seen_eid += batch.num_events
        # break
    return total_loss/dataset['train_length'], max_seen_eid



@torch.no_grad()
def test_new(runtime: TrainRuntime, max_seen_id, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    model_bundle = runtime.model_bundle
    model = model_bundle.model
    neighbor_loader = runtime.sampler_runtime.sampler
    dataset_runtime = runtime.dataset_runtime
    dataset = dataset_runtime.dataset
    data = dataset_runtime.data
    data_cache = dataset_runtime.data_cache or {}
    cached_t = data_cache.get('t')
    cached_msg = data_cache.get('msg')
    cached_src = data_cache.get('src')
    deliver_to = runtime.deliver_to
    if split_mode == 'val':
        loader = dataset['val_dataloader']
    else:
        loader = dataset['test_dataloader']
    device = runtime.device

    metric = dataset['metric']
    evaluator = dataset['evaluator']
    neg_sampler = dataset['neg_sampler']
    decoder = runtime.decoder
    embedding = runtime.embedding
    val_neg = runtime.val_neg
    min_dst_idx = dataset_runtime.min_dst_idx
    max_dst_idx = dataset_runtime.max_dst_idx



    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []

    max_seen_eid = max_seen_id

    # for pos_batch in loader:
    # breakpoint()
    for pos_batch in loader:#tqdm(loader, desc=split_mode):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        if neg_sampler is not None:
            neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
            min_len = min(len(inner) for inner in neg_batch_list)
            neg_batch_tensor = torch.tensor([inner[:min_len] for inner in neg_batch_list])
            neg_batch_tensor_T = neg_batch_tensor.T
            if val_neg > -1:
                idx = torch.randperm(neg_batch_tensor_T.size(0))[:val_neg]
                neg_batch_tensor_T = neg_batch_tensor_T[idx]
        else:
            # Default fallback when offline negatives are unavailable.
            num_rand_neg = val_neg if val_neg > 0 else 100
            neg_batch_tensor_T = torch.randint(
                min_dst_idx,
                max_dst_idx + 1,
                (num_rand_neg, pos_src.shape[0]),
                dtype=torch.long,
            )
        num_neg = neg_batch_tensor_T.shape[0]
        bs = pos_src.shape[0]
        preds = []

        for i in range(num_neg):
            # print(i)
            # breakpoint()
            neg_dst = neg_batch_tensor_T[i]
            root_ts = torch.cat([pos_t, pos_t, pos_t], dim = 0).double().to(device)
            root_nodes = torch.cat([pos_src, pos_dst, neg_dst], dim = 0).to(device)
            # breakpoint()
            n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
            if decoder == 'NCN':
                nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
                nid_ts[:root_nodes.shape[0]] = root_ts
                n_id, e_id, edge_index = neighbor_loader.sample(n_id, nid_ts)

            # bmsk = e_id>max_seen_eid
            if deliver_to == "neighbor":
                # neighbors = get_latest_neighbors_per_node(src, pos_dst, t, edge_index, n_id)
                # breakpoint()
                bsrc = torch.stack([torch.arange(pos_src.shape[0]).to(device), pos_dst.to(device)])
                bdst = torch.stack([torch.arange(pos_src.shape[0], 2*pos_src.shape[0]).to(device), pos_src.to(device)])
                bndst = torch.stack([torch.arange(2*pos_src.shape[0], 3*pos_src.shape[0]).to(device), torch.full((pos_src.shape[0],), -1).to(device)])
            
                od = torch.cat([ bsrc.T, bdst.T, bndst.T, neighbor_loader.id_to_pair], dim=0)
                od_updated  = torch.cat([od, n_id.unsqueeze(1)], dim=1)
                tmp = od_updated[:, 0] % bs
                od_updated = torch.cat([od_updated, tmp.unsqueeze(1)], dim=1)

                mem_graph_quad, store_quad = vectorized_getMem_graph(od_updated,bs, max_seen_eid)
                # print("Memgraph Constructed")
                # breakpoint()
                # b_eid = mem_graph_quad[3]#e_id[bmsk]
                # b_eid_cpu = b_eid.cpu()
                # # breakpoint()
                # b_t = dataset['data'].t[b_eid_cpu].to(device)
                # b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
                # z_m, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg, data=dataset['data'])
                # breakpoint()


                # Get unique (src, dst, eid) triples
                triplet_keys = mem_graph_quad[[0, 1, 3]].T  # shape: [N, 3]
                unique_keys, inverse_indices = torch.unique(triplet_keys, dim=0, return_inverse=True)

                # Use unique EIDs to extract just the raw messages and timestamps once
                unique_eids = unique_keys[:, 2]  # this is just eid column
                b_t_unique = _cached_or_host_lookup(cached_t, data.t, unique_eids, device)
                b_raw_msg_unique = _cached_or_host_lookup(cached_msg, data.msg, unique_eids, device)
                z_m, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=data)
                # _apply_intra_batch_info(mem_graph_quad, n_id, b_t_unique, b_raw_msg_unique, old_mem, unique_keys, inverse_indices)





                
                
            else:


                mem_graph_quad = getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, pos_src, pos_dst, device)
                b_eid = mem_graph_quad[3]#e_id[bmsk]
                b_eid_cpu = b_eid.cpu()
                b_t = _cached_or_host_lookup(cached_t, data.t, b_eid_cpu, device)
                b_raw_msg = _cached_or_host_lookup(cached_msg, data.msg, b_eid_cpu, device)
                if cached_src is not None:
                    b_eid_idx = b_eid.long()
                    if b_eid_idx.device != cached_src.device:
                        b_eid_idx = b_eid_idx.to(cached_src.device)
                    lhs_src = cached_src[b_eid_idx]
                    rhs_src = n_id[mem_graph_quad[1]]
                    if rhs_src.device != lhs_src.device:
                        rhs_src = rhs_src.to(lhs_src.device)
                    b_isrc = rhs_src == lhs_src
                else:
                    b_isrc = n_id[mem_graph_quad[1]].cpu() == data.src[b_eid_cpu]

                z_m, last_update = model['memory'](n_id, mem_graph_quad[0:2,:], b_t, b_raw_msg, b_isrc, delivery_addr = mem_graph_quad[2])


                remap_partial = model['memory'].mem_graph(n_id[:3*bs],torch.arange(3*bs).to(device) , pos_src, pos_dst)
                remap_fall_back = neighbor_loader.assoc[n_id[:3*bs]]
                remap = torch.where(remap_partial != -1, remap_partial, remap_fall_back)

                z_m = torch.cat([z_m[remap], z_m[3*bs:]])
                last_update = torch.cat([last_update[remap], last_update[3*bs:]])

                ei_src_all = model['memory'].mem_graph(n_id[edge_index[0,:]],edge_index[1,:] , pos_src, pos_dst)
                updated_src = torch.where(ei_src_all != -1, ei_src_all, edge_index[0, :])
                edge_index = torch.stack([updated_src, edge_index[1,:]])

            # z = model['gnn'](
            #     z_m,
            #     last_update,
            #     edge_index,
            #     dataset['data'].t[e_id].to(device),
            #     dataset['data'].msg[e_id].to(device),
            # )
            # breakpoint()
            if embedding == "time_emb":
                nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
                nid_ts[:root_nodes.shape[0]] = root_ts
                z = model['gnn'](
                    z_m,
                    last_update,
                    nid_ts
                )
            else:
                z = model['gnn'](
                    z_m,
                    last_update,
                    edge_index,
                    _cached_or_host_lookup(cached_t, data.t, e_id, device),
                    _cached_or_host_lookup(cached_msg, data.msg, e_id, device),
                )

            if decoder == "NCN":
                # pos_out = model['link_pred'](z[0:bs], z[bs:2*bs])
                # neg_out = model['link_pred'](z[0:bs], z[2*bs:3*bs])

                time_info = (last_update, pos_t)
                src_re = torch.arange(bs)
                pos_re = torch.arange(bs, 2*bs)
                neg_re = torch.arange(2*bs, 3*bs)
                if i == 0:
                    pos_out = model['link_pred'](z, edge_index, torch.stack([src_re,pos_re]), 2, cn_time_decay=False, time_info=time_info)
                    preds.append(pos_out)
                neg_out = model['link_pred'](z, edge_index, torch.stack([src_re,neg_re]), 2, cn_time_decay=False, time_info=time_info)
                preds.append(neg_out)
            else:
                if i == 0:
                    pos_out = model['link_pred'](z[0:bs], z[bs:2*bs])
                    preds.append(pos_out)
                neg_out = model['link_pred'](z[0:bs], z[2*bs:3*bs])
                preds.append(neg_out)
            
            # torch.cuda.empty_cache()
            # Free all temporary variables from the current negative sample loop
            # del neg_dst
            # del root_ts, root_nodes
            # del n_id, e_id, edge_index
            # del mem_graph_quad
            # del store_quad
            # del b_eid, b_eid_cpu
            # del b_t, b_raw_msg
            # del z_m, last_update
            # del z
            # # del pos_out, neg_out  # even if not always present, safe to delete
            # torch.cuda.empty_cache()

        all_y_preds = torch.cat(preds, dim=1)
        del preds
        # breakpoint()
        for i in range(all_y_preds.size(0)):
            y_pred = all_y_preds[i]  # shape [1000]
            if evaluator is not None:
                input_dict = {
                    "y_pred_pos": np.array([y_pred[0].item()]),
                    "y_pred_neg": np.array(y_pred[1:].cpu()),
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])
            else:
                pos_score = y_pred[0]
                neg_scores = y_pred[1:]
                rank = 1 + (neg_scores >= pos_score).sum().item()
                perf_list.append(1.0 / float(rank))
        if deliver_to == "neighbor":
            root_ts = torch.cat([pos_t, pos_t], dim = 0).double().to(device)
            root_nodes = torch.cat([pos_src, pos_dst], dim = 0).to(device)
            n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)

            bsrc = torch.stack([torch.arange(pos_src.shape[0]).to(device), pos_dst.to(device)])
            bdst = torch.stack([torch.arange(pos_src.shape[0], 2*pos_src.shape[0]).to(device), pos_src.to(device)])
            # bndst = torch.stack([torch.arange(2*pos_src.shape[0], 3*pos_src.shape[0]).to(device), torch.full((pos_src.shape[0],), -1).to(device)])
            
            od = torch.cat([ bsrc.T, bdst.T, neighbor_loader.id_to_pair], dim=0)
            od_updated  = torch.cat([od, n_id.unsqueeze(1)], dim=1)
            tmp = od_updated[:, 0] % bs
            od_updated = torch.cat([od_updated, tmp.unsqueeze(1)], dim=1)
            mem_graph_quad, store_quad = vectorized_getMem_graph(od_updated,bs, max_seen_eid)

            # b_eid = mem_graph_quad[3]#e_id[bmsk]
            # b_eid_cpu = b_eid.cpu()
            #     # breakpoint()
            # b_t = dataset['data'].t[b_eid_cpu].to(device)
            # b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)

            # # breakpoint()
            # # n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
            # z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg, data = dataset['data'])


            triplet_keys = mem_graph_quad[[0, 1, 3]].T  # shape: [N, 3]
            unique_keys, inverse_indices = torch.unique(triplet_keys, dim=0, return_inverse=True)

                # Use unique EIDs to extract just the raw messages and timestamps once
            unique_eids = unique_keys[:, 2]  # this is just eid column
            b_t_unique = _cached_or_host_lookup(cached_t, data.t, unique_eids, device)
            b_raw_msg_unique = _cached_or_host_lookup(cached_msg, data.msg, unique_eids, device)
            z_m, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=data)
                

            store_eid = store_quad[3]
            dirs = _cached_src_dirs(cached_src, data.src, store_eid, n_id, store_quad, device)
            model['memory'].update_state(n_id, z_m, last_update, store_quad, dirs.to(device))
            # update_state(n_id, z, last_update, store_quad, dataset['data'].t[store_eid].to(device), dataset['data'].msg[store_eid])
        else:
            model['memory'].update_state_v2(
                mem_graph_quad[2], remap, bs, 
                pos_src.to(device), pos_dst.to(device), pos_t.to(device), pos_msg.to(device), 
                n_id, last_update, z_m)        

        max_seen_eid += bs
    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics, max_seen_eid




def train_with_custom_neg_sampler(runtime: TrainRuntime):
    model_bundle = runtime.model_bundle
    model = model_bundle.model
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    model['memory'].reset_state()

    optimizer = model_bundle.optimizer
    criterion = model_bundle.criterion

    dataset_runtime = runtime.dataset_runtime
    dataset = dataset_runtime.dataset
    train_loader = dataset['train_dataloader']
    device = runtime.device
    min_dst_idx = dataset_runtime.min_dst_idx
    max_dst_idx = dataset_runtime.max_dst_idx
    neighbor_loader = runtime.sampler_runtime.sampler
    neg_sampler = runtime.neg_sampler

    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        bs  = src.shape[0]
        # neg_dst = torch.randint(
        #     min_dst_idx,
        #     max_dst_idx + 1,
        #     (src.size(0),),
        #     dtype=torch.long,
        #     device=device,
        # )
        neg_dst = neg_sampler.sample(pos_dst.cpu()).to(device)

        root_ts = torch.cat([t, t, t], dim = 0).double()
        root_nodes = torch.cat([src, pos_dst, neg_dst], dim = 0)
        n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
        # breakpoint()


        bmsk = dataset['data'].t[e_id]>=batch.t[0].cpu()
        b_edge_index = edge_index[:,bmsk]

        # breakpoint()

        b_eid = e_id[bmsk]
        b_t = dataset['data'].t[b_eid].to(device)
        b_raw_msg = dataset['data'].msg[b_eid].to(device)
        b_isrc = n_id[b_edge_index[1]].cpu() == dataset['data'].src[b_eid]


        z, last_update = model['memory'](n_id, b_edge_index, b_t, b_raw_msg, b_isrc)
        # breakpoint()
        # # if z.size(0) < 2 * bs:
        # #     breakpoint()
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            dataset['data'].t[e_id].to(device),
            dataset['data'].msg[e_id].to(device),
        )
        # breakpoint()
        
        pos_out = model['link_pred'](z[0:bs], z[bs:2*bs])
        neg_out = model['link_pred'](z[0:bs], z[2*bs:3*bs])


        # breakpoint()
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        loss.backward()
        optimizer.step()
        model['memory'].update_state(src, pos_dst, t, msg)
        model['memory'].detach()

        total_loss += float(loss) * batch.num_events
        print("step loss: ", float(loss) * batch.num_events)
        # break

    # breakpoint()
    return total_loss/dataset['train_length']








def train_sample_only(runtime: TrainRuntime, max_seen_id):
    model_bundle = runtime.model_bundle
    model = model_bundle.model
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    model['memory'].reset_state()

    optimizer = model_bundle.optimizer
    criterion = model_bundle.criterion

    dataset_runtime = runtime.dataset_runtime
    dataset = dataset_runtime.dataset
    train_loader = dataset['train_dataloader']
    device = runtime.device
    min_dst_idx = dataset_runtime.min_dst_idx
    max_dst_idx = dataset_runtime.max_dst_idx
    neighbor_loader = runtime.sampler_runtime.sampler

    neg_sampler = runtime.neg_sampler
    known_dsts = dataset_runtime.known_dsts
    deliver_to = runtime.deliver_to
    decoder = runtime.decoder
    embedding = runtime.embedding


    total_loss = 0
    max_seen_eid = max_seen_id

    for batch in train_loader:
        batch = batch.to(device)
        # optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        bs  = src.shape[0]

        neg_dst = train_neg_sampler(min_dst_idx, max_dst_idx, pos_dst, device, neg_sampler, known_dsts = known_dsts)
        # neg_dst = torch.randint(
        #     min_dst_idx,
        #     max_dst_idx + 1,
        #     (src.size(0),),
        #     dtype=torch.long,
        #     device=device,
        # )
        root_ts = torch.cat([t, t, t], dim = 0).double()
        root_nodes = torch.cat([src, pos_dst, neg_dst], dim = 0)
        n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
        # breakpoint()
        if decoder == 'NCN':
            nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
            nid_ts[:root_nodes.shape[0]] = root_ts
            n_id, e_id, edge_index = neighbor_loader.sample(n_id, nid_ts)
        # breakpoint()
                # break
    return None, None#total_loss/dataset['train_length'], max_seen_eid
