import torch
import numpy as np
from tqdm import tqdm




def train_neg_sampler(min_dst_idx, max_dst_idx, pos_dst, device, neg_sampler):
    bs = pos_dst.shape[0]
    if neg_sampler is None:
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

def getMem_graph_apan(od_updated, bs, max_seen_eid, device):
    # breakpoint()
    tmp = od_updated[:,0]%bs
    od_updated = torch.cat([od_updated, tmp.unsqueeze(1)], dim=1)
    od = od_updated[:,:2]
    valid_vals = od[od[:, 1] != -1, 1]  # Filter out -1s from column 1

            # Get the indices for each valid value in col 1 that appear in col 0
            # matches = [(val.item(), (od_updated[:, 2] == val).nonzero(as_tuple=True)[0]) for val in valid_vals]
    match_dict = {val.item(): (od_updated[:, 2] == val).nonzero(as_tuple=True)[0] for val in valid_vals}
    # breakpoint()
    ei_src = []
    ei_dst = []
    ei_dla = []
    ei_eid = []
    msg_store_nid = []
    msg_store_eid = []
    msg_store_src = []
    msg_store_dst = []
    for i in range(od_updated.shape[0]):
        batch_edge_index = od_updated[i, 3].item()
        cond = od_updated[i, 0].item()
        key = od_updated[i, 1].item() 
        if key == -1:
            continue
        dla = match_dict[key]
        valid_dla_mask = od_updated[dla, 3]>batch_edge_index
        valid_dla = dla[valid_dla_mask]
        if valid_dla.shape[0] == 0:
            msg_store_nid.append(key)
            msg_store_eid.append(batch_edge_index+max_seen_eid+1)
            msg_store_src.append(cond)
            if cond < bs:
                msg_store_dst.append(cond+bs)
            else:
                msg_store_dst.append(cond%bs)
            # msg_store_rev.append(cond>bs)
        for j in range(valid_dla.shape[0]):
            if cond < bs:
                ei_src.append(cond)
                ei_dst.append(cond+bs)
            else:
                ei_src.append(cond)
                ei_dst.append(cond%bs)
            ei_dla.append(valid_dla[j].item())
            ei_eid.append(batch_edge_index+max_seen_eid+1)
                # breakpoint()
            # breakpoint()
    mem_graph_quad = torch.tensor([ei_src, ei_dst, ei_dla, ei_eid]).long().to(device)
    # breakpoint()
    store_quad = torch.tensor([msg_store_src, msg_store_dst, msg_store_nid, msg_store_eid]).long().to(device)
    # breakpoint()
    return mem_graph_quad,  store_quad

def getMem_graph_apan_v2(od_updated, bs, max_seen_eid, device):
    tmp = od_updated[:, 0] % bs
    od_updated = torch.cat([od_updated, tmp.unsqueeze(1)], dim=1)
    od = od_updated[:, :2]
    
    # Extract unique keys
    valid_vals = od[od[:, 1] != -1, 1].unique()
    match_dict = {val.item(): (od_updated[:, 2] == val).nonzero(as_tuple=True)[0] for val in valid_vals}

 
    e_src, e_dst, e_dla, e_eid = [], [], [], []
    msg_store_src, msg_store_dst, msg_store_nid, msg_store_eid = [], [], [], []

    keys = od_updated[:, 1]
    conds = od_updated[:, 0]
    bidxs = od_updated[:, 3]
    # breakpoint()

    for i in range(od_updated.shape[0]):
        key = keys[i].item()
        cond = conds[i].item()
        batch_edge_index = bidxs[i].item()
        if key == -1:
            continue

        dla = match_dict[key]
        valid_dla_mask = od_updated[dla, 3] > batch_edge_index
        valid_dla = dla[valid_dla_mask]

        if valid_dla.shape[0] == 0:
            msg_store_nid.append(key)
            msg_store_eid.append(batch_edge_index + max_seen_eid + 1)
            msg_store_src.append(cond)
            msg_store_dst.append(cond + bs if cond < bs else cond % bs)
        n = valid_dla.shape[0]
        ei_src_l = torch.full((n,), cond, device=valid_dla.device)
        ei_dst_l = torch.full((n,), cond + bs if cond < bs else cond % bs, device=valid_dla.device)
        ei_dla_l = valid_dla
        ei_eid_l = torch.full((n,), batch_edge_index + max_seen_eid + 1, device=valid_dla.device)
        e_src.append(ei_src_l)
        e_dst.append(ei_dst_l)
        e_dla.append(ei_dla_l)
        e_eid.append(ei_eid_l)

    # breakpoint()
    e_src = torch.cat(e_src) 
    e_dst = torch.cat(e_dst)
    e_dla = torch.cat(e_dla)
    e_eid = torch.cat(e_eid)
    # breakpoint()
    mem_graph_quad = torch.stack([e_src, e_dst, e_dla, e_eid])
    # mem_graph_quad = torch.tensor([ei_src, ei_dst, ei_dla, ei_eid], device=device, dtype=torch.long)
    # if torch.equal(tensor1, tensor2)
    # breakpoint()

    store_quad = torch.tensor([msg_store_src, msg_store_dst, msg_store_nid, msg_store_eid], device=device, dtype=torch.long)
    
    return mem_graph_quad, store_quad

# import torch
import mem_update_graph  

def getMem_graph_apan(od_updated: torch.Tensor, bs: int, max_seen_eid: int, device: torch.device):
    N = od_updated.shape[0]
    
    # Append bidx = cond % bs as the 4th column
    tmp = od_updated[:, 0] % bs
    od_updated = torch.cat([od_updated, tmp.unsqueeze(1)], dim=1)  # shape: (N, 4)
    
    keys = od_updated[:, 1].long()
    conds = od_updated[:, 0].long()
    bidxs = od_updated[:, 3].long()

    # ----------------------------
    # Build match_dict lookup arrays
    # ----------------------------
    valid_keys = keys[keys != -1].unique()
    match_indices_list = []
    match_starts = torch.full((keys.max().item() + 2,), -1, dtype=torch.long, device=device)
    match_ends = torch.full_like(match_starts, -1)

    offset = 0
    for key in valid_keys:
        idxs = (od_updated[:, 2] == key).nonzero(as_tuple=True)[0]
        n = idxs.numel()
        if n > 0:
            match_starts[key] = offset
            match_ends[key] = offset + n
            match_indices_list.append(idxs)
            offset += n

    if match_indices_list:
        match_indices = torch.cat(match_indices_list)
    else:
        match_indices = torch.empty(0, dtype=torch.long, device=device)

    # ----------------------------
    # Allocate output memory (overallocate)
    # ----------------------------
    max_out = N * 5
    ei_src = torch.empty(max_out, dtype=torch.long, device=device)
    ei_dst = torch.empty_like(ei_src)
    ei_dla = torch.empty_like(ei_src)
    ei_eid = torch.empty_like(ei_src)

    msg_store_src = torch.empty(max_out, dtype=torch.long, device=device)
    msg_store_dst = torch.empty_like(msg_store_src)
    msg_store_nid = torch.empty_like(msg_store_src)
    msg_store_eid = torch.empty_like(msg_store_src)

    counter_edge = torch.zeros(1, dtype=torch.long, device=device)
    counter_store = torch.zeros(1, dtype=torch.long, device=device)

    # ----------------------------
    # Call the CUDA kernel
    # ----------------------------
    mem_update_graph.build_mem_graph(
        conds,
        keys,
        bidxs,
        od_updated,
        match_starts,
        match_ends,
        match_indices,
        int(max_seen_eid),
        int(bs),
        ei_src,
        ei_dst,
        ei_dla,
        ei_eid,
        msg_store_src,
        msg_store_dst,
        msg_store_nid,
        msg_store_eid,
        counter_edge,
        counter_store
    )

    # ----------------------------
    # Slice to actual sizes and return
    # ----------------------------
    ei_len = counter_edge.item()
    store_len = counter_store.item()

    mem_graph_quad = torch.stack([
        ei_src[:ei_len], ei_dst[:ei_len], ei_dla[:ei_len], ei_eid[:ei_len]
    ], dim=0)

    store_quad = torch.stack([
        msg_store_src[:store_len], msg_store_dst[:store_len],
        msg_store_nid[:store_len], msg_store_eid[:store_len]
    ], dim=0)

    return mem_graph_quad, store_quad




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
    deliver_to = targs['deliver_to']
    decoder = targs['decoder']
    embedding = targs['embedding']


    total_loss = 0
    max_seen_eid = max_seen_id

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        bs  = src.shape[0]

        neg_dst = train_neg_sampler(min_dst_idx, max_dst_idx, pos_dst, device, neg_sampler)
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
        if decoder == 'NCN':
            nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
            nid_ts[:root_nodes.shape[0]] = root_ts
            n_id, e_id, edge_index = neighbor_loader.sample(n_id, nid_ts)
        # breakpoint()
        if deliver_to == "neighbor":
            # neighbors = get_latest_neighbors_per_node(src, pos_dst, t, edge_index, n_id)
            bsrc = torch.stack([torch.arange(src.shape[0]).to(device), pos_dst])
            bdst = torch.stack([torch.arange(src.shape[0], 2*src.shape[0]).to(device), src])
            bndst = torch.stack([torch.arange(2*src.shape[0], 3*src.shape[0]).to(device), torch.full((src.shape[0],), -1).to(device)])
            
            od = torch.cat([ bsrc.T, bdst.T, bndst.T, neighbor_loader.id_to_pair], dim=0)
            od_updated  = torch.cat([od, n_id.unsqueeze(1)], dim=1)

            # mem_graph_quad, store_quad = getMem_graph_apan(od_updated,bs, max_seen_eid, device)

            mem_graph_quad, store_quad = getMem_graph_apan_v2(od_updated,bs, max_seen_eid, device)
            # tmp = model['memory'].mem_graph(od_updated,bs, max_seen_eid)
            # import mem_update_graph  # from the compiled module

            # Call the forward function
            # mem_graph_out, store_out, mem_counter, store_counter = model['memory'].mem_graph(od_updated,bs, max_seen_eid)

            # Trim output to actual size
            # mem_graph_quad = mem_graph_out[:, :mem_counter.item()]  # shape: [4, num_edges]
            # store_quad = store_out[:, :store_counter.item()]   # shape: [4, num_messages]

            # breakpoint()
            # mem_graph_quad, store_quad =None, None # model['memory'].mem_graph(od_updated,bs, max_seen_eid)

            # if torch.equal(mem_graph_quad, mem_graph_quad_v2) and torch.equal(store_quad, store_quad_v2):
            #     print("Same")
            # else:
            #     breakpoint()
            # print("Memgraph Constructed")
            # breakpoint()
            b_eid = mem_graph_quad[3]#e_id[bmsk]
            b_eid_cpu = b_eid.cpu()
            # breakpoint()
            b_t = dataset['data'].t[b_eid_cpu].to(device)
            b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg, data = dataset['data'])
            # breakpoint()
            # print("Updated memory Computed")
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
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg)
            store_eid = store_quad[3].cpu()
            # breakpoint()
            dirs = dataset['data'].src[store_eid] == n_id[store_quad[0]].cpu()#store_quad[0].cpu()
            
            model['memory'].update_state(n_id, z, last_update, store_quad, dirs.to(device))
            # print("State Updated")
            model['memory'].detach()
            total_loss += float(loss) * batch.num_events
            


        else:
            mem_graph_quad_uf  = getMem_graph(model, neighbor_loader, edge_index, n_id, e_id, bs, max_seen_eid, src, pos_dst, device)
            mem_graph_quad = mem_graph_quad_uf

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
            b_t = dataset['data'].t[b_eid_cpu].to(device)
            b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
            b_isrc = n_id[mem_graph_quad[1]].cpu() == dataset['data'].src[b_eid_cpu]

            z, last_update = model['memory'](n_id, mem_graph_quad[0:2,:], b_t, b_raw_msg, b_isrc, delivery_addr = mem_graph_quad[2])
            z = torch.cat([z[remap], z[3*bs:]])
            last_update = torch.cat([last_update[remap], last_update[3*bs:]])
            
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



    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []

    max_seen_eid = max_seen_id

    # for pos_batch in loader:
    for pos_batch in tqdm(loader, desc=split_mode):
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
        num_neg = neg_batch_tensor_T.shape[0]
        bs = pos_src.shape[0]
        preds = []

        for i in range(num_neg):
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
                mem_graph_quad, store_quad = getMem_graph_apan(od_updated,bs, max_seen_eid, device)
                # print("Memgraph Constructed")
                # breakpoint()
                b_eid = mem_graph_quad[3]#e_id[bmsk]
                b_eid_cpu = b_eid.cpu()
                # breakpoint()
                b_t = dataset['data'].t[b_eid_cpu].to(device)
                b_raw_msg = dataset['data'].msg[b_eid_cpu].to(device)
                z_m, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg)
                
            else:


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

        all_y_preds = torch.cat(preds, dim=1)
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
            z, last_update = model['memory'](n_id, mem_graph_quad, b_t, b_raw_msg)
            store_eid = store_quad[3].cpu()
            model['memory'].update_state(n_id, z, last_update, store_quad, dataset['data'].t[store_eid].to(device), dataset['data'].msg[store_eid])
        else:
            model['memory'].update_state_v2(
                mem_graph_quad[2], remap, bs, 
                pos_src.to(device), pos_dst.to(device), pos_t.to(device), pos_msg.to(device), 
                n_id, last_update, z_m)        

        max_seen_eid += bs
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
