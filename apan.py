
import os
import timeit
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import TemporalDataLoader

from tgb.utils.utils import get_args, set_random_seed
from modules.dataset_pyg import PyGLinkPropPredDataset
from modules.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import MeanAggregator as Agg
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import APANMemory
from modules.train_utils import get_latest_neighbors_per_node
debug = True

# from apan_train_test_pipeline import train_apan, test_apan
def train_apan(model, data, train_loader, neighbor_loader, optimizer, criterion, device, assoc, min_dst_idx, max_dst_idx):
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()
    neighbor_loader.reset_state()

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        neighbors = get_latest_neighbors_per_node(src, pos_dst, t, edge_index, n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        # breakpoint()
        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        model['memory'].update_state(src, pos_dst, t, msg, neighbors, n_id)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / data.num_events



@torch.no_grad()
def test(loader, neg_sampler, split_mode='val'):
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
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []

    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        # model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        neighbors = get_latest_neighbors_per_node(pos_src, pos_dst, pos_t, edge_index, n_id)
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg, neighbors, n_id)


        neighbor_loader.insert(pos_src, pos_dst)


    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics





# Set parameters
args, _ = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA = args.data
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
LR = args.lr
BATCH_SIZE = args.bs
NUM_EPOCH = args.num_epoch
SEED = args.seed
NUM_NEIGHBORS = 10
MAX_TR_TIME = 12 * 60 * 60

# Load dataset
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
data = dataset.get_TemporalData().to(device)
train_data = data[dataset.train_mask]
val_data = data[dataset.val_mask]
test_data = data[dataset.test_mask]
metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler

evaluator = Evaluator(name=DATA)

# Loaders
train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Build model
memory = APANMemory(
    num_nodes=data.num_nodes,
    raw_msg_dim=data.msg.size(-1),
    memory_dim=MEM_DIM,
    time_dim=TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=Agg(),
    mailbox_size=10,
    # num_heads=4,
).to(device)
memory._init_message_store()

# breakpoint()
gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

model = {'memory': memory, 'gnn': gnn, 'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Support setup
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# Train
losses, times = [], []
total_train_time = 0
dataset.load_val_ns()
for epoch in range(1, NUM_EPOCH + 1):
    if total_train_time > MAX_TR_TIME:
        break
    start = timeit.default_timer()
    loss = train_apan(model, data, train_loader, neighbor_loader, optimizer, criterion, device, assoc, min_dst_idx, max_dst_idx)
    duration = timeit.default_timer() - start
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Time: {duration:.2f}s")
    losses.append(loss)
    times.append(duration)
    total_train_time += duration
            # validation
        # start_val = timeit.default_timer()
    # def test_apan(model, data, loader, neg_sampler, split_mode, neighbor_loader, device, assoc, metric):
    if not debug:
        perf_metric_val = test(val_loader, neg_sampler, split_mode='val')#test_apan(model, data, val_loader, neg_sampler, "val", neighbor_loader, device, assoc, metric)
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        # print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        # val_perf_list.append(perf_metric_val)

# # Evaluate
# dataset.load_test_ns()
# test_metric = test_apan(model, data, test_loader, neg_sampler, "test", neighbor_loader, device, assoc, metric)
# print(f"Final Test {metric}: {test_metric:.4f}")