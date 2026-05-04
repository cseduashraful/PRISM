import torch
import numpy as np
from tqdm import tqdm
# from apan_mem_wrapper import call_cuda_kernel as apan_g
import mem_update_graph


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

def SYNC(tag):
    torch.cuda.synchronize()
    print(f"[SYNC OK] {tag}", flush=True)



def getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, src, pos_dst, device):
    # SYNC("get mmgraph start")
    bmsk = e_id>max_seen_eid
    bdst = torch.arange(bs*2)
    bsrc = torch.cat([torch.arange(bs, bs*2), torch.arange(bs)])
    bedge = torch.stack([bsrc, bdst]).to(device)
    # SYNC("bedge construction")
    # breakpoint()
    bedge_all = model['memory'].mem_graph(n_id[torch.cat([bedge[0,:], bedge[1,:]])],torch.cat([bedge[1,:], bedge[1,:]]) , src, pos_dst)
    

    # SYNC("bedge all construction")
    # torch.cuda.synchronize()
    





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
import timeit


def sum_list(values):
    total = 0
    for v in values:
        total += v
    return total


def train(targs, max_seen_id):
    model = targs['model']
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    model['memory'].reset_state()

    optimizer = targs['optimizer']
    criterion = targs['criterion']

    dataset = targs['dataset']
    train_loader = dataset['train_dataloader']
    device = targs['device']
    min_dst_idx = targs['min_dst_idx']
    max_dst_idx = targs['max_dst_idx']
    neighbor_loader = targs['sampler']

    neg_sampler = targs['neg_sampler']
    known_dsts = targs['known_dsts']
    deliver_to = targs['deliver_to']
    decoder = targs['decoder']
    embedding = targs['embedding']


    total_loss = 0
    max_seen_eid = max_seen_id
    mcg_t = 0
    depth_cal = False
    freq_tensor = torch.zeros(8000, dtype=torch.long)

    s_times = []
    mcg_times = []
    mem_module_times = []
    emb_module_times = []
    decoder_times = []
    mem_store_times = []
    b_pass_times = []
    data_movement_times = []
    e_times = []


    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        bs  = src.shape[0]

        time_start = timeit.default_timer()
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
        s_times.append(timeit.default_timer() - time_start)
        time_start = timeit.default_timer()
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
            mcg_ts = timeit.default_timer()
            mem_graph_quad, store_quad = vectorized_getMem_graph(od_updated, bs, max_seen_eid)

            mcg_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()

            ##Disable later
            # depths = analyze_mcg_depth_apan(mem_graph_quad)

            # mcg_t = mcg_t + (-mcg_ts + timeit.default_timer())
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
            b_t = dataset['data'].t[b_eid_cpu].to(device)
            b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
            data_movement_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()
            # z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg, data = dataset['data'])




            # Get unique (src, dst, eid) triples
            triplet_keys = mem_graph_quad[[0, 1, 3]].T  # shape: [N, 3]
            unique_keys, inverse_indices = torch.unique(triplet_keys, dim=0, return_inverse=True)

            # Use unique EIDs to extract just the raw messages and timestamps once
            unique_eids = unique_keys[:, 2]  # this is just eid column
            b_t_unique = dataset['data'].t[unique_eids.cpu()].to(device)
            b_raw_msg_unique = dataset['data'].msg[unique_eids.cpu()].to(device)
            # time_start = timeit.default_timer()
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=dataset['data'])
                
            mem_module_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()








            # breakpoint()
            # print("Updated memory Computed")
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                dataset['data'].t[e_id].to(device),
                dataset['data'].msg[e_id].to(device),
            )

            emb_module_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer() 

            # breakpoint()
            # print("Embedding generated")
            pos_out = model['link_pred'](z[0:bs], z[bs:2*bs])
            neg_out = model['link_pred'](z[0:bs], z[2*bs:3*bs])

            decoder_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()


            # breakpoint()
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))
            # print("batch loss computed")
            loss.backward()
            optimizer.step()

            b_pass_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()

            # z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg)
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=dataset['data'])
            
            store_eid = store_quad[3].cpu()
            # breakpoint()
            dirs = dataset['data'].src[store_eid] == n_id[store_quad[0]].cpu()#store_quad[0].cpu()
            
            model['memory'].update_state(n_id, z, last_update, store_quad, dirs.to(device))
            # print("State Updated")
            model['memory'].detach()
            mem_store_times.append(timeit.default_timer() - time_start) 
            total_loss += float(loss) * batch.num_events
            


        else:
            # mcg_ts = timeit.default_timer()
            
            mem_graph_quad_uf  = getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, src, pos_dst, device)
            mem_graph_quad = mem_graph_quad_uf
            
            mem_graph = n_id[mem_graph_quad[:2]]


            ###Disable later
            # if depth_cal:
            #     analyze_mcg_depth_tgn(mem_graph_quad, freq_tensor)
                # depth_tensor =  torch.cat([depth_tensor, depths])

            # breakpoint()

            remap_partial = model['memory'].mem_graph(n_id[:3*bs],torch.arange(3*bs).to(device) , src, pos_dst)
            remap_fall_back = neighbor_loader.assoc[n_id[:3*bs]]
            remap = torch.where(remap_partial != -1, remap_partial, remap_fall_back)
            # mcg_t = mcg_t + (-mcg_ts + timeit.default_timer())
            mcg_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()

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
            b_t = dataset['data'].t[b_eid_cpu].to(device)
            b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
            b_isrc = n_id[mem_graph_quad[1]].cpu() == dataset['data'].src[b_eid_cpu]

            z, last_update = model['memory'](n_id, mem_graph_quad[0:2,:], b_t, b_raw_msg, b_isrc, delivery_addr = mem_graph_quad[2])
            z = torch.cat([z[remap], z[3*bs:]])
            last_update = torch.cat([last_update[remap], last_update[3*bs:]])
            

            # breakpoint()
            ei_src_all = model['memory'].mem_graph(n_id[edge_index[0,:]],edge_index[1,:] , src, pos_dst)

            updated_src = torch.where(ei_src_all != -1, ei_src_all, edge_index[0, :])
            edge_index = torch.stack([updated_src, edge_index[1,:]])

            mem_module_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()
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
                    dataset['data'].t[e_id].to(device),
                    dataset['data'].msg[e_id].to(device),
                )
            # breakpoint()
            emb_module_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()

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

            decoder_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer() 
            # breakpoint()
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            loss.backward()
            optimizer.step()

            b_pass_times.append(timeit.default_timer() - time_start)
            time_start = timeit.default_timer()
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

            mem_store_times.append(timeit.default_timer() - time_start)
            total_loss += float(loss) * batch.num_events
        
        
        
        max_seen_eid += batch.num_events
        # break
    # if depth_cal:
    #     print(f"MCG construction time: {mcg_t}")
    #     idx = torch.where(freq_tensor != 0)[0]
    #     print_depth_stats(freq_tensor, idx)
    #     breakpoint()
    
    print(f"'s_times' : {sum_list(s_times)},")
    print(f"'memory_module_times' : {sum_list(mcg_times)+sum_list(mem_module_times)},")
    print(f"'emb_module_times' : {sum_list(emb_module_times)},")
    print(f"'decoder_times' : {sum_list(decoder_times)},")
    print(f"'memory_store_times' : {sum_list(mem_store_times)-sum_list(mem_module_times)},")



    # print(f"mcg_times = {mcg_times}")
    # print(f"mem_module_times = {mem_module_times}")
    # print(f"emb_module_times = {emb_module_times}")
    # print(f"decoder_times = {decoder_times}")
    # print(f"b_pass_times = {b_pass_times}")
    # print(f"mem_store_times = {mem_store_times}")

    # breakpoint()

    return total_loss/dataset['train_length'], max_seen_eid




def print_depth_stats(freq_tensor, idx, ks=(1,2,3,4,5)):
    """
    freq_tensor : 1D tensor of counts indexed by depth
    idx         : tensor of depth values that have nonzero frequency
    ks          : tuple of pass thresholds to report cumulative percentages
    """

    depths = idx.long()
    freqs = freq_tensor[depths].long()

    N = freqs.sum().item()

    # Mean
    mean = (depths * freqs).sum().float() / N

    # Cumulative distribution
    cum = torch.cumsum(freqs, dim=0)

    # Median
    median_depth = depths[torch.searchsorted(cum, torch.tensor(N // 2))].item()

    # 95th percentile
    p95_depth = depths[torch.searchsorted(cum, torch.tensor(int(0.95 * N)))].item()

    # 99th percentile
    p99_depth = depths[torch.searchsorted(cum, torch.tensor(int(0.99 * N)))].item()

    # Max depth
    max_depth = depths[-1].item()

    print("===== Dependency Depth Statistics =====")
    print(f"Total samples      : {N}")
    print(f"Mean depth         : {mean:.4f}")
    print(f"Median depth       : {median_depth}")
    print(f"95th percentile    : {p95_depth}")
    print(f"99th percentile    : {p99_depth}")
    print(f"Maximum depth      : {max_depth}")

    # Cumulative percentages for small pass counts
    for k in ks:
        if k >= depths[-1]:
            pct = 100.0
        else:
            mask = depths <= k
            pct = freqs[mask].sum().item() / N * 100
        print(f"% resolved ≤ {k} passes : {pct:.2f}%")

    print("========================================")

@torch.no_grad()
def test_new(targs, max_seen_id, split_mode):
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
    model = targs['model']
    neighbor_loader = targs['sampler']
    dataset = targs['dataset']
    deliver_to = targs['deliver_to']
    if split_mode == 'val':
        loader = dataset['val_dataloader']
    else:
        loader = dataset['test_dataloader']
    device = targs['device']

    metric = dataset['metric']
    evaluator = dataset['evaluator']
    neg_sampler = dataset['neg_sampler']
    decoder = targs['decoder']
    embedding = targs['embedding']
    val_neg = targs['val_neg']



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

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
        min_len = min(len(inner) for inner in neg_batch_list)
        neg_batch_tensor = torch.tensor([inner[:min_len] for inner in neg_batch_list])
        neg_batch_tensor_T = neg_batch_tensor.T
        # print()
        # breakpoint()
        if val_neg > -1:
            idx = torch.randperm(neg_batch_tensor_T.size(0))[:val_neg]
            neg_batch_tensor_T = neg_batch_tensor_T[idx]
        num_neg = neg_batch_tensor_T.shape[0]
        bs = pos_src.shape[0]
        preds = []

        for i in range(num_neg):
            # print(i)
            # breakpoint()
            # SYNC("enter test_new loop")
            neg_dst = neg_batch_tensor_T[i]
            root_ts = torch.cat([pos_t, pos_t, pos_t], dim = 0).double().to(device)
            root_nodes = torch.cat([pos_src, pos_dst, neg_dst], dim = 0).to(device)
            # breakpoint()
            n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
            # SYNC("after sampling")
            if decoder == 'NCN':
                nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
                nid_ts[:root_nodes.shape[0]] = root_ts
                n_id, e_id, edge_index = neighbor_loader.sample(n_id, nid_ts)
            # SYNC("after building ncn decode")
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
                b_t_unique = dataset['data'].t[unique_eids.cpu()].to(device)
                b_raw_msg_unique = dataset['data'].msg[unique_eids.cpu()].to(device)
                z_m, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=dataset['data'])
                # _apply_intra_batch_info(mem_graph_quad, n_id, b_t_unique, b_raw_msg_unique, old_mem, unique_keys, inverse_indices)

                
            else:

                # SYNC("enter tgn else block")
                mem_graph_quad = getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, pos_src, pos_dst, device)
                b_eid = mem_graph_quad[3]#e_id[bmsk]
                b_eid_cpu = b_eid.cpu()
                b_t = dataset['data'].t[b_eid_cpu].to(device)
                b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
                b_isrc = n_id[mem_graph_quad[1]].cpu() == dataset['data'].src[b_eid_cpu]

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
                    dataset['data'].t[e_id].to(device),
                    dataset['data'].msg[e_id].to(device),
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
            
            input_dict = {
                "y_pred_pos": np.array([y_pred[0].item()]),  # scalar wrapped in array
                "y_pred_neg": np.array(y_pred[1:].cpu()),    # shape [999]
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
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
            b_t_unique = dataset['data'].t[unique_eids.cpu()].to(device)
            b_raw_msg_unique = dataset['data'].msg[unique_eids.cpu()].to(device)
            z_m, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=dataset['data'])
                

            store_eid = store_quad[3].cpu()
            dirs = dataset['data'].src[store_eid] == n_id[store_quad[0]].cpu()#store_quad[0].cpu()
            model['memory'].update_state(n_id, z_m, last_update, store_quad, dirs.to(device))
            # update_state(n_id, z, last_update, store_quad, dataset['data'].t[store_eid].to(device), dataset['data'].msg[store_eid])
        else:
            model['memory'].update_state_v2(
                mem_graph_quad[2], remap, bs, 
                pos_src.to(device), pos_dst.to(device), pos_t.to(device), pos_msg.to(device), 
                n_id, last_update, z_m)        

        max_seen_eid += bs
        # print(max_seen_eid)
    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics, max_seen_eid















@torch.no_grad()
def online_test(targs, max_seen_id, split_mode):
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
    model = targs['model']
    neighbor_loader = targs['sampler']
    dataset = targs['dataset']
    data = dataset['data']
    deliver_to = targs['deliver_to']
    if split_mode == 'val':
        loader = dataset['val_dataloader']
    else:
        loader = dataset['test_dataloader']
    device = targs['device']

    metric = dataset['metric']
    evaluator = dataset['evaluator']
    neg_sampler = dataset['neg_sampler']
    decoder = targs['decoder']
    embedding = targs['embedding']
    val_neg = targs['val_neg']



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
        bs = pos_src.shape[0]



        #update neighborloader
        
        #todo: update sampler (neighbor_loader) with pos_src, pos_dst, pos_t,
        new_start = max_seen_eid+1
        new_end = new_start+bs
        
        neighbor_loader.extend_tci( data.src[new_start:new_end].tolist(), data.dst[new_start:new_end].tolist(), 
                                   data.t[new_start:new_end].double().tolist(), torch.arange(new_start, new_end).tolist(),
                                   duplicate_undirected=True)

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
        min_len = min(len(inner) for inner in neg_batch_list)
        neg_batch_tensor = torch.tensor([inner[:min_len] for inner in neg_batch_list])
        neg_batch_tensor_T = neg_batch_tensor.T
        # print()
        # breakpoint()
        if val_neg > -1:
            idx = torch.randperm(neg_batch_tensor_T.size(0))[:val_neg]
            neg_batch_tensor_T = neg_batch_tensor_T[idx]
        num_neg = neg_batch_tensor_T.shape[0]
        
        preds = []

        for i in range(num_neg):
            # print(i)
            # breakpoint()
            # SYNC("enter test_new loop")
            neg_dst = neg_batch_tensor_T[i]
            root_ts = torch.cat([pos_t, pos_t, pos_t], dim = 0).double().to(device)
            root_nodes = torch.cat([pos_src, pos_dst, neg_dst], dim = 0).to(device)
            # breakpoint()
            n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
            # SYNC("after sampling")
            if decoder == 'NCN':
                nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
                nid_ts[:root_nodes.shape[0]] = root_ts
                n_id, e_id, edge_index = neighbor_loader.sample(n_id, nid_ts)
            # SYNC("after building ncn decode")
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
                b_t_unique = dataset['data'].t[unique_eids.cpu()].to(device)
                b_raw_msg_unique = dataset['data'].msg[unique_eids.cpu()].to(device)
                z_m, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=dataset['data'])
                # _apply_intra_batch_info(mem_graph_quad, n_id, b_t_unique, b_raw_msg_unique, old_mem, unique_keys, inverse_indices)

                
            else:

                # SYNC("enter tgn else block")
                mem_graph_quad = getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, pos_src, pos_dst, device)
                b_eid = mem_graph_quad[3]#e_id[bmsk]
                b_eid_cpu = b_eid.cpu()
                b_t = dataset['data'].t[b_eid_cpu].to(device)
                b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
                b_isrc = n_id[mem_graph_quad[1]].cpu() == dataset['data'].src[b_eid_cpu]

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
                    dataset['data'].t[e_id].to(device),
                    dataset['data'].msg[e_id].to(device),
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
            
            input_dict = {
                "y_pred_pos": np.array([y_pred[0].item()]),  # scalar wrapped in array
                "y_pred_neg": np.array(y_pred[1:].cpu()),    # shape [999]
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
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
            b_t_unique = dataset['data'].t[unique_eids.cpu()].to(device)
            b_raw_msg_unique = dataset['data'].msg[unique_eids.cpu()].to(device)
            z_m, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=dataset['data'])
                

            store_eid = store_quad[3].cpu()
            dirs = dataset['data'].src[store_eid] == n_id[store_quad[0]].cpu()#store_quad[0].cpu()
            model['memory'].update_state(n_id, z_m, last_update, store_quad, dirs.to(device))
            # update_state(n_id, z, last_update, store_quad, dataset['data'].t[store_eid].to(device), dataset['data'].msg[store_eid])
        else:
            model['memory'].update_state_v2(
                mem_graph_quad[2], remap, bs, 
                pos_src.to(device), pos_dst.to(device), pos_t.to(device), pos_msg.to(device), 
                n_id, last_update, z_m)        

        max_seen_eid += bs
        # print(max_seen_eid)
    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics, max_seen_eid







































def train_with_custom_neg_sampler(targs):
    model = targs['model']
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    model['memory'].reset_state()

    optimizer = targs['optimizer']
    criterion = targs['criterion']

    dataset = targs['dataset']
    train_loader = dataset['train_dataloader']
    device = targs['device']
    min_dst_idx = targs['min_dst_idx']
    max_dst_idx = targs['max_dst_idx']
    neighbor_loader = targs['sampler']
    neg_sampler = targs['neg_sampler']

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








def train_sample_only(targs, max_seen_id):
    model = targs['model']
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    model['memory'].reset_state()

    optimizer = targs['optimizer']
    criterion = targs['criterion']

    dataset = targs['dataset']
    train_loader = dataset['train_dataloader']
    device = targs['device']
    min_dst_idx = targs['min_dst_idx']
    max_dst_idx = targs['max_dst_idx']
    neighbor_loader = targs['sampler']

    neg_sampler = targs['neg_sampler']
    known_dsts = targs['known_dsts']
    deliver_to = targs['deliver_to']
    decoder = targs['decoder']
    embedding = targs['embedding']


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


















def train_prof(targs, max_seen_id):
    model = targs['model']
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    model['memory'].reset_state()

    optimizer = targs['optimizer']
    criterion = targs['criterion']

    dataset = targs['dataset']
    train_loader = dataset['train_dataloader']
    device = targs['device']
    min_dst_idx = targs['min_dst_idx']
    max_dst_idx = targs['max_dst_idx']
    neighbor_loader = targs['sampler']

    neg_sampler = targs['neg_sampler']
    known_dsts = targs['known_dsts']
    deliver_to = targs['deliver_to']
    decoder = targs['decoder']
    embedding = targs['embedding']


    total_loss = 0
    max_seen_eid = max_seen_id
    mcg_t = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        bs  = src.shape[0]

        with torch.profiler.record_function("negative sampler"):
            neg_dst = train_neg_sampler(min_dst_idx, max_dst_idx, pos_dst, device, neg_sampler, known_dsts = known_dsts)
   
        root_ts = torch.cat([t, t, t], dim = 0).double()
        root_nodes = torch.cat([src, pos_dst, neg_dst], dim = 0)
        with torch.profiler.record_function("neighborhood sampler"):
            n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
        # breakpoint()
        if decoder == 'NCN':
            nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
            nid_ts[:root_nodes.shape[0]] = root_ts
            n_id, e_id, edge_index = neighbor_loader.sample(n_id, nid_ts)
        # breakpoint()
        if deliver_to == "neighbor": #APAN
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
            b_eid = mem_graph_quad[3]#e_id[bmsk]
            b_eid_cpu = b_eid.cpu().long()
            # print(b_eid_cpu)
            # breakpoint()
            b_t = dataset['data'].t[b_eid_cpu].to(device)
            b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
            # z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg, data = dataset['data'])
            # Get unique (src, dst, eid) triples
            triplet_keys = mem_graph_quad[[0, 1, 3]].T  # shape: [N, 3]
            unique_keys, inverse_indices = torch.unique(triplet_keys, dim=0, return_inverse=True)

            # Use unique EIDs to extract just the raw messages and timestamps once
            unique_eids = unique_keys[:, 2]  # this is just eid column
            b_t_unique = dataset['data'].t[unique_eids.cpu()].to(device)
            b_raw_msg_unique = dataset['data'].msg[unique_eids.cpu()].to(device)
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=dataset['data'])
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                dataset['data'].t[e_id].to(device),
                dataset['data'].msg[e_id].to(device),
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
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t_unique, b_raw_msg_unique, unique_keys, inverse_indices, data=dataset['data'])
            
            store_eid = store_quad[3].cpu()
            # breakpoint()
            dirs = dataset['data'].src[store_eid] == n_id[store_quad[0]].cpu()#store_quad[0].cpu()
            
            model['memory'].update_state(n_id, z, last_update, store_quad, dirs.to(device))
            # print("State Updated")
            model['memory'].detach()
            total_loss += float(loss) * batch.num_events

        else:
            with torch.profiler.record_function("MCG construction"):
                mem_graph_quad_uf  = getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, src, pos_dst, device)
                mem_graph_quad = mem_graph_quad_uf
                remap_partial = model['memory'].mem_graph(n_id[:3*bs],torch.arange(3*bs).to(device) , src, pos_dst)
                remap_fall_back = neighbor_loader.assoc[n_id[:3*bs]]
                remap = torch.where(remap_partial != -1, remap_partial, remap_fall_back)


            b_eid = mem_graph_quad[3]#e_id[bmsk]
            b_eid_cpu = b_eid.cpu()
            b_t = dataset['data'].t[b_eid_cpu].to(device)
            b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
            b_isrc = n_id[mem_graph_quad[1]].cpu() == dataset['data'].src[b_eid_cpu]
            with torch.profiler.record_function("Memory Module"):
                z, last_update = model['memory'](n_id, mem_graph_quad[0:2,:], b_t, b_raw_msg, b_isrc, delivery_addr = mem_graph_quad[2])
            z = torch.cat([z[remap], z[3*bs:]])
            last_update = torch.cat([last_update[remap], last_update[3*bs:]])
            
            # breakpoint()
            ei_src_all = model['memory'].mem_graph(n_id[edge_index[0,:]],edge_index[1,:] , src, pos_dst)

            updated_src = torch.where(ei_src_all != -1, ei_src_all, edge_index[0, :])
            edge_index = torch.stack([updated_src, edge_index[1,:]])
            if embedding == "time_emb":
                nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
                nid_ts[:root_nodes.shape[0]] = root_ts
                z = model['gnn'](
                    z,
                    last_update,
                    nid_ts
                )
            else:
                with torch.profiler.record_function("Embedding GAT"):
                    z = model['gnn'](
                        z,
                        last_update,
                        edge_index,
                        dataset['data'].t[e_id].to(device),
                        dataset['data'].msg[e_id].to(device),
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

            with torch.profiler.record_function("backword pass"):
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












def analyze_mcg_depth_tgn_old(mem_graph_quad):
    """
    Computes dependency depth for PRISM-style TGN MCG.

    mem_graph_quad: Tensor [4, E]
        row 2 = parent index (delivery address)
    """

    parents = mem_graph_quad[2].long()
    num_nodes = parents.numel()

    # Initialize depth array
    depth = torch.zeros(num_nodes, dtype=torch.long, device=parents.device)

    # Since dependencies only go forward in batch order,
    # a simple DP in order works.
    for child in range(num_nodes):
        parent = parents[child].item()
        if parent >= 0 and parent < child:
            depth[child] = depth[parent] + 1

    depth_cpu = depth.cpu()

    print("===== TGN-PRISM MCG Depth Stats =====")
    print("Max depth:", depth_cpu.max().item())
    print("Mean depth:", depth_cpu.float().mean().item())
    print("Median depth:", depth_cpu.median().item())
    print("95th percentile:", torch.quantile(depth_cpu.float(), 0.95).item())
    print("======================================")

    # Distribution for histogram
    unique_depths, counts = torch.unique(depth_cpu, return_counts=True)
    print("\nDepth Distribution:")
    for d, c in zip(unique_depths.tolist(), counts.tolist()):
        print(f"Depth {d}: {c}")

    return depth_cpu



def analyze_mcg_depth_apan(mem_graph_quad):
    """
    Robust MCG depth analyzer.
    Works for APAN and TGN.
    """

    parents_raw = mem_graph_quad[2].long()
    num_edges = parents_raw.numel()

    # Child nodes are implicit: 0..num_edges-1
    children_raw = torch.arange(num_edges, device=parents_raw.device)

    # Collect all node ids involved
    valid_mask = parents_raw >= 0
    all_nodes = torch.cat([
        children_raw,
        parents_raw[valid_mask]
    ])

    # Create compact id mapping
    unique_nodes = torch.unique(all_nodes)
    id_map = {node.item(): i for i, node in enumerate(unique_nodes)}

    num_nodes = len(unique_nodes)

    # Build adjacency
    depth = torch.zeros(num_nodes, dtype=torch.long)
    adj = [[] for _ in range(num_nodes)]

    for child_raw, parent_raw in zip(children_raw.tolist(), parents_raw.tolist()):
        if parent_raw >= 0:
            u = id_map[parent_raw]
            v = id_map[child_raw]
            adj[u].append(v)

    # Topological DP (nodes already time-ordered in practice)
    for u in range(num_nodes):
        for v in adj[u]:
            depth[v] = max(depth[v], depth[u] + 1)

    depth_cpu = depth.cpu()

    print("===== MCG Depth Stats =====")
    print("Max depth:", depth_cpu.max().item())
    print("Mean depth:", depth_cpu.float().mean().item())
    print("Median depth:", depth_cpu.median().item())
    print("95th percentile:", torch.quantile(depth_cpu.float(), 0.95).item())
    print("===========================")

    return depth_cpu





# import torch
from collections import defaultdict

def analyze_mcg_depth_tgn(mem_graph_quad, freq_tesnosr):
    dst = mem_graph_quad[0].tolist()
    src = mem_graph_quad[1].tolist()
    del_addr = mem_graph_quad[2].tolist()

    node_depth = {}

    for i in range(len(dst)):
        delivered = del_addr[i]

        parents = [src[i], dst[i]]

        parent_depths = [node_depth.get(p, 0) for p in parents]

        node_depth[delivered] = max(
            node_depth.get(delivered, 0),
            1 + max(parent_depths)
        )

    depths = list(node_depth.values())
    depth_tensor = torch.tensor(depths)

    print("Max node depth:", depth_tensor.max().item())
    print("Mean:", depth_tensor.float().mean().item())
    print("Median:", depth_tensor.median().item())
    print("95th percentile:",
          torch.quantile(depth_tensor.float(), 0.95).item())

    unique, counts = torch.unique(depth_tensor, return_counts=True)
    freq_tesnosr[unique] += counts

    _zf = mem_graph_quad[:3].max().item()-depth_tensor.size(0)

    freq_tesnosr[0] +=  _zf
    # breakpoint()

    # return freq_tesnosr