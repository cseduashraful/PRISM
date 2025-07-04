
import os
import os.path as osp
import timeit
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import TemporalDataLoader

from tgb.utils.utils import get_args, set_random_seed, save_results
# from tgb.utils.utils import get_args, set_random_seed, 

from modules.dataset_pyg import PyGLinkPropPredDataset
from modules.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import MeanAggregator as Agg
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import APANMemory
from modules.early_stopping import EarlyStopMonitor
from modules.train_utils import latest_index_per_node,get_fixed_neighbors
debug = False

import torch

def get_in_neighbors_padded(edges: torch.Tensor, n_id: torch.Tensor, max_neighbors: int = 10) -> torch.Tensor:
    """
    For each node in n_id, find up to max_neighbors in-neighbors from edges[0] where edges[1] == node.
    Pads with -1 if fewer than max_neighbors.
    
    Args:
        edges (Tensor): shape (2, E), where edges[0] = src, edges[1] = dst
        n_id (Tensor): 1D tensor of node ids
        max_neighbors (int): number of neighbors to retrieve per node

    Returns:
        Tensor of shape (len(n_id), max_neighbors) with in-neighbors (padded with -1)
    """
    device = edges.device
    num_nodes = n_id.size(0)

    # Map node IDs in n_id to row indices
    nid_max = int(n_id.max().item()) + 1
    nid_map = torch.full((nid_max,), -1, dtype=torch.long, device=device)
    nid_map[n_id] = torch.arange(num_nodes, device=device)

    src, dst = edges[0], edges[1]
    valid = nid_map[dst] >= 0

    src_valid = src[valid]
    dst_valid = dst[valid]
    dst_idx = nid_map[dst_valid]  # maps dst node id to row index in output

    # Sort by dst_idx
    sorted_idx = torch.argsort(dst_idx)
    src_sorted = src_valid[sorted_idx]
    dst_sorted = dst_idx[sorted_idx]

    # Count neighbors per node
    counts = torch.bincount(dst_sorted, minlength=num_nodes)
    max_nbrs = counts.clamp(max=max_neighbors)
    cumsum = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(counts, dim=0)])

    neighbors_padded = torch.full((num_nodes, max_neighbors), -1, dtype=torch.long, device=device)

    # Generate flat indices and row assignments
    offsets = cumsum[:-1]
    mask_matrix = torch.arange(max_neighbors, device=device).expand(num_nodes, max_neighbors) < max_nbrs.unsqueeze(1)

    flat_indices = (offsets.unsqueeze(1) + torch.arange(max_neighbors, device=device)).reshape(-1)
    flat_indices = flat_indices[mask_matrix.reshape(-1)]
    rows = torch.arange(num_nodes, device=device).repeat_interleave(max_neighbors)
    rows = rows[mask_matrix.reshape(-1)]
    cols = torch.arange(max_neighbors, device=device).repeat(num_nodes)
    cols = cols[mask_matrix.reshape(-1)]

    # Assign to padded output
    neighbors_padded[rows, cols] = src_sorted[flat_indices]

    return neighbors_padded

def get_latest_neighbors_per_node(src, pos_dst, t, edge_index, n_id):
    neighbors_padded = get_in_neighbors_padded(n_id[edge_index], n_id)
    # breakpoint()
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
    return neigh, neighbors_padded


# from apan_train_test_pipeline import train_apan, test_apan
def train_apan(model, data, train_loader, neighbor_loader, optimizer, criterion, device, assoc, min_dst_idx, max_dst_idx):
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()
    neighbor_loader.reset_state()
    max_seen_eid = -1
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
        neighbors, npadded = get_latest_neighbors_per_node(src, pos_dst, t, edge_index, n_id)
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

        model['memory'].update_state(src, pos_dst, t, msg, neighbors, n_id, npadded = npadded, eid_start = max_seen_eid+1, assoc = assoc)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events
        max_seen_eid += batch.num_events

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
start_overall = timeit.default_timer()
args, _ = get_args()
print("INFO: Arguments:", args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA = args.data
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
LR = args.lr
BATCH_SIZE = args.bs
NUM_EPOCH = args.num_epoch
SEED = args.seed
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
MAX_TR_TIME = 48 * 60 * 60
MAX_EXEC_TIME = 72 * 60 * 60
MODEL_NAME = "APAN"



print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

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


# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'


for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)
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


    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    # Train
    start_train_val = timeit.default_timer()
    losses, times, mrrs = [], [], []
    total_train_time = 0
    total_execution_time = 0
    dataset.load_val_ns()
    for epoch in range(1, NUM_EPOCH + 1):
        start_epoch_train = timeit.default_timer()
        loss = train_apan(model, data, train_loader, neighbor_loader, optimizer, criterion, device, assoc, min_dst_idx, max_dst_idx)
        duration = timeit.default_timer() - start_epoch_train
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Time: {duration:.2f}s")
        losses.append(loss)
        times.append(duration)
        total_train_time += duration
                # validation
            # start_val = timeit.default_timer()
        # def test_apan(model, data, loader, neg_sampler, split_mode, neighbor_loader, device, assoc, metric):
        if not debug:
            # perf_metric_val = test(val_loader, neg_sampler, split_mode='val')#test_apan(model, data, val_loader, neg_sampler, "val", neighbor_loader, device, assoc, metric)
            # print(f"\tValidation {metric}: {perf_metric_val: .4f}")
            # # print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
            # # val_perf_list.append(perf_metric_val)
            # mrrs.append(perf_metric_val)
            
            # # validation
            start_val = timeit.default_timer()
            perf_metric_val = test(val_loader, neg_sampler, split_mode="val")
            print(f"\tValidation {metric}: {perf_metric_val: .4f}")
            mrrs.append(perf_metric_val)
            print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        
            # check for early stopping
            
            if early_stopper.step_check(perf_metric_val, model):
                break
        if total_train_time > MAX_TR_TIME:
            break
        total_execution_time += timeit.default_timer() - start_epoch_train
        if total_execution_time > MAX_EXEC_TIME:
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")
    print("'loss' : ",losses,",")
    print("'time' : ",times,",")
    print("'mrr' : ",mrrs,",")



    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)

#     # loading the test negative samples
    dataset.load_test_ns()

#     # final testing
    start_test = timeit.default_timer()
    perf_metric_test = test(test_loader, neg_sampler, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  f'val {metric}': mrrs,
                  f'test {metric}': perf_metric_test,
                  'test_time': test_time,
                  'tot_train_val_time': train_val_time
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
