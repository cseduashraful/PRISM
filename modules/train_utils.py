import torch
# import numpy as np
# from tqdm import tqdm


def train(targs):

    dataset = targs['dataset']
    train_loader = dataset['train_dataloader']
    device = targs['device']
    min_dst_idx = targs['min_dst_idx']
    max_dst_idx = targs['max_dst_idx']
    neighbor_loader = targs['sampler']


    for batch in train_loader:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )
        root_ts = torch.cat([t, t, t], dim = 0).double()
        root_nodes = torch.cat([src, pos_dst, neg_dst], dim = 0)
        n_ids, e_ids, edge_index = neighbor_loader.sample(root_nodes, root_ts)
        # breakpoint()
    return 0
        

