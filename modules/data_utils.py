from modules.dataset_pyg import PyGLinkPropPredDataset
from modules.evaluate import Evaluator

import torch
import numpy as np
from tqdm import tqdm
import itertools

from torch_geometric.loader import TemporalDataLoader

def read_data(DATA, BATCH_SIZE, load_neg_sampler = True):
    dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()
    metric = dataset.eval_metric

    train_data = data[train_mask]
    train_length = train_data.num_events
    val_data = data[val_mask]
    test_data = data[test_mask]


    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)


    observed_nodes = torch.cat([train_data.src, train_data.dst]).unique()
    transductive_mask = {
        "val": torch.isin(val_data.src, observed_nodes).cpu().numpy(),
        "test": torch.isin(test_data.src, observed_nodes).cpu().numpy(),
    }
    inductive_mask = {
        "val": ~ transductive_mask["val"],
        "test": ~ transductive_mask["test"],
    }
    neg_sampler = None
    evaluator = None
    if load_neg_sampler:
        dataset.load_val_ns()
        dataset.load_test_ns()
        neg_sampler = dataset.negative_sampler
        evaluator = Evaluator(name=DATA)
    # breakpoint()
    return {
        "data": data,
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "test_dataloader": test_loader,
        "neg_sampler": neg_sampler,
        "evaluator": evaluator,
        "metric": metric,
        "train_length": train_length,
        "transductive_mask": transductive_mask,
        "inductive_mask": inductive_mask,
    }
