import torch
import os
import yaml
# import dgl
import time
import pandas as pd
import numpy as np


from torch.utils.data import DataLoader

# tgb imports
from modules.evaluate import Evaluator
from modules.dataset_pyg import PyGLinkPropPredDataset

import pdb


# from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding, TimeEmbedding, IdentityEmbedding
from modules.msg_func import IdentityMessage, MLPMessage
from modules.msg_agg import LastAggregator, MeanAggregator


# internal imports
# from modules.temporal_dataset import TemporalGraphDataset
# from dependencyGraph import dependecyAwareBatch as dab, dependecyAwareBatch_neutron as dab_neutron, dependecyAwareBatch_v2 as dab2
from modules.neighbor_loader import LastNeighborLoader

# def getDataWithDependecyBlock(args, NUM_NEIGHBORS, csv=False, load_neg_sampler=True, device = torch.device("cpu"), algo='da_tgnn'):
#     DATA = args.data
#     if csv:
#         print("Not implemeneted yet")
#     else:
#         # dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
#         dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
#         print("data loading done")
#         # pdb.set_trace()
#         train_mask = dataset.train_mask
#         val_mask = dataset.val_mask
#         test_mask = dataset.test_mask
#     # data = dataset.get_TemporalData()
#         data = dataset.get_TemporalData()
#         # data = data.to(device)

#     metric = dataset.eval_metric
#     train_data = data[train_mask]
#     train_length = train_data.num_events
#     val_data = data[val_mask]
#     test_data = data[test_mask]


#     observed_nodes = torch.cat([train_data.src, train_data.dst]).unique()
#     transductive_mask = {
#         "val": torch.isin(val_data.src, observed_nodes).cpu().numpy(),
#         "test": torch.isin(test_data.src, observed_nodes).cpu().numpy(),
#     }
#     inductive_mask = {
#         "val": ~ transductive_mask["val"],
#         "test": ~ transductive_mask["test"],
#     }






#     neg_sampler = None
#     evaluator = None
#     if load_neg_sampler:
#         dataset.load_val_ns()
#         dataset.load_test_ns()
#         neg_sampler = dataset.negative_sampler
#         evaluator = Evaluator(name=DATA)

#     train_dataset = TemporalGraphDataset(train_data.src, train_data.dst, train_data.t, train_data.msg)
#     val_dataset = TemporalGraphDataset(val_data.src, val_data.dst, val_data.t, val_data.msg)
#     test_dataset = TemporalGraphDataset(test_data.src, test_data.dst, test_data.t, test_data.msg)
#     train_dataloader = DataLoader(train_dataset, batch_size= args.bs, shuffle=False)
#     val_dataloader = DataLoader(val_dataset, batch_size= args.bs, shuffle=False)
#     test_dataloader = DataLoader(test_dataset, batch_size= args.bs, shuffle=False)

#     if algo == "neutron_stream":
#         # raise NotImplementedError("Not yet implemented")
#         train_blocks = dab_neutron(train_dataloader)
#         val_blocks = dab_neutron(val_dataloader)
#         test_blocks = dab_neutron(test_dataloader)
#     else:
#         sampler = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS)
#         train_blocks = dab2(train_dataloader, sampler)
#         val_blocks = dab2(val_dataloader, sampler)
#         test_blocks = dab2(test_dataloader, sampler)

#     train_dataset = TemporalGraphDataset(train_data.src, train_data.dst, train_data.t, train_data.msg, batch=train_blocks)
#     val_dataset = TemporalGraphDataset(val_data.src, val_data.dst, val_data.t, val_data.msg, batch=val_blocks)
#     test_dataset = TemporalGraphDataset(test_data.src, test_data.dst, test_data.t, test_data.msg, batch=test_blocks)

#     train_dataloader = DataLoader(train_dataset, batch_size= args.bs, shuffle=False)
#     val_dataloader = DataLoader(val_dataset, batch_size= args.bs, shuffle=False)
#     test_dataloader = DataLoader(test_dataset, batch_size= args.bs, shuffle=False)

#     ret = {
#         "data": data,
#         "train_dataloader": train_dataloader,
#         "val_dataloader": val_dataloader,
#         "test_dataloader": test_dataloader,
#         "neg_sampler": neg_sampler,
#         "evaluator": evaluator,
#         "metric": metric,
#         "train_length": train_length,
#         "transductive_mask": transductive_mask,
#         "inductive_mask": inductive_mask,
#     }
    
#     return ret#data, train_dataloader, val_dataloader, test_dataloader, neg_sampler, evaluator, metric, train_length


def get_msg_func(msg_func, data, MEM_DIM, TIME_DIM):
    if msg_func == 'identity':
        return IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM)
    elif msg_func == 'mlp':
        return MLPMessage(data.msg.size(-1), MEM_DIM, TIME_DIM)
    else:
        raise ValueError('Invalid message function! Function must be "identity" or "mlp".')

def get_agg_module(agg_func):
    if agg_func == 'last':
        return LastAggregator()
    elif agg_func == 'mean':
        return MeanAggregator()
    else:
        raise ValueError('Invalid aggregator function! Function must be "last" or "mean".')
    
def get_emb_module(emb_func, data, MEM_DIM, EMB_DIM, memory, device):
    if emb_func == 'GraphAttention':
        return GraphAttentionEmbedding(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)
    elif emb_func == 'Time':
        return TimeEmbedding(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)
    elif emb_func == 'Identity':
        return IdentityEmbedding(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)
    else:
        raise ValueError('Invalid embedding function! Function must be "GraphAttentionEmbedding", "TimeEmbedding", or "IdentityEmbedding".')
