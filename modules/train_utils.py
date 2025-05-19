import torch
import numpy as np
from tqdm import tqdm


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

    total_loss = 0
    max_seen_eid = max_seen_id

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        bs  = src.shape[0]
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )
        root_ts = torch.cat([t, t, t], dim = 0).double()
        root_nodes = torch.cat([src, pos_dst, neg_dst], dim = 0)
        n_id, e_id, edge_index = neighbor_loader.sample(root_nodes, root_ts)
        # breakpoint()


        # bmsk = dataset['data'].t[e_id]>=batch.t[0].cpu()

        bmsk = e_id>max_seen_eid  #dataset['data'].t[e_id]>=batch.t[0].cpu()
        # b_edge_index = edge_index[:,bmsk]

        # breakpoint()
        ei_src_all = model['memory'].mem_graph(n_id[edge_index[0,:]],edge_index[1,:] , src, pos_dst)
        updated_src = torch.where(ei_src_all != -1, ei_src_all, edge_index[0, :])
        edge_index = torch.stack([updated_src, edge_index[1,:]])
        b_edge_index = edge_index[:,bmsk]


        b_eid = e_id[bmsk]
        b_t = dataset['data'].t[b_eid].to(device)
        b_raw_msg = dataset['data'].msg[b_eid].to(device)
        b_isrc = n_id[b_edge_index[1]].cpu() == dataset['data'].src[b_eid]

        z, last_update = model['memory'](n_id, b_edge_index, b_t, b_raw_msg, b_isrc)

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
        # model['memory'].update_state(src, pos_dst, t, msg)
        z_m, last_update = model['memory'](n_id, b_edge_index, b_t, b_raw_msg, b_isrc)
        model['memory'].update_state_v2(
            b_edge_index, b_edge_index[0:,], bs, 
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
    if split_mode == 'val':
        loader = dataset['val_dataloader']
    else:
        loader = dataset['test_dataloader']
    device = targs['device']

    metric = dataset['metric']
    evaluator = dataset['evaluator']
    neg_sampler = dataset['neg_sampler']



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
            bmsk = e_id>max_seen_eid
            
            ei_src_all = model['memory'].mem_graph(n_id[edge_index[0,:]],edge_index[1,:] , pos_src, pos_dst)
            updated_src = torch.where(ei_src_all != -1, ei_src_all, edge_index[0, :])
            edge_index = torch.stack([updated_src, edge_index[1,:]])
            b_edge_index = edge_index[:,bmsk]
            b_eid = e_id[bmsk]
            b_t = dataset['data'].t[b_eid].to(device)
            b_raw_msg = dataset['data'].msg[b_eid].to(device)
            b_isrc = n_id[b_edge_index[1]].cpu() == dataset['data'].src[b_eid]
            z_m, last_update = model['memory'](n_id, b_edge_index, b_t, b_raw_msg, b_isrc)

            z = model['gnn'](
                z_m,
                last_update,
                edge_index,
                dataset['data'].t[e_id].to(device),
                dataset['data'].msg[e_id].to(device),
            )
            # breakpoint()
            if i == 0:
                pos_out = model['link_pred'](z[0:bs], z[bs:2*bs])
                preds.append(pos_out)
            neg_out = model['link_pred'](z[0:bs], z[2*bs:3*bs])
            preds.append(neg_out)

        all_y_preds = torch.cat(preds, dim=1)
        for i in range(all_y_preds.size(0)):
            y_pred = all_y_preds[i]  # shape [1000]
            
            input_dict = {
                "y_pred_pos": np.array([y_pred[0].item()]),  # scalar wrapped in array
                "y_pred_neg": np.array(y_pred[1:].cpu()),    # shape [999]
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
        
        model['memory'].update_state_v2(
            b_edge_index, b_edge_index[0:,], bs, 
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
