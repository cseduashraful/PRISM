from modules.data_utils import read_data #, get_TCSR, get_TCSR_py, verify_tcsr
from modules.recent_sampler import Recent_K_Sampler
from modules.train_utils import train as actrain, test_new as test, train_with_custom_neg_sampler, train_sample_only
from modules.memory_module import DAATGNMemory, DAAAPANMemory, DA_APANMemory

from modules.neg_sampler import NegLinkSamplerDest
from modules.emb_module import GraphAttentionEmbedding, TimeEmbedding
from modules.early_stopping import EarlyStopMonitor
from modules.msg_agg import LastAggregator, MeanAggregator as Agg, AttentionAggregator, TransformerAggregator
from modules.msg_func import IdentityMessage, MLPMessage
from modules.decoder import LinkPredictor
from modules.NCNDecoder.NCNPred import NCNPredictor

# from torch.optim.lr_scheduler import StepLR

from tgb.utils.utils import get_args, set_random_seed, save_results
import numpy as np
import torch

import timeit
import os
import sys
import os.path as osp
from pathlib import Path
import argparse

import preprocessor #openmp

def deep_size_info(obj):
    import sys
    import torch
    
    cpu_bytes, gpu_bytes = 0, 0
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            c, g = deep_size_info(v)
            cpu_bytes += c
            gpu_bytes += g
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            c, g = deep_size_info(v)
            cpu_bytes += c
            gpu_bytes += g
    elif torch.is_tensor(obj):
        if obj.device.type == 'cpu':
            cpu_bytes += obj.element_size() * obj.nelement()
        else:
            gpu_bytes += obj.element_size() * obj.nelement()
    else:
        cpu_bytes += sys.getsizeof(obj)
    
    return cpu_bytes, gpu_bytes

import sys
import torch

def object_size_info(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0, 0
    seen.add(obj_id)

    cpu_bytes, gpu_bytes = 0, 0

    if torch.is_tensor(obj):
        # Tensors
        if obj.device.type == "cpu":
            cpu_bytes += obj.element_size() * obj.nelement()
        else:
            gpu_bytes += obj.element_size() * obj.nelement()

    elif isinstance(obj, dict):
        for k, v in obj.items():
            c, g = object_size_info(k, seen)
            cpu_bytes += c; gpu_bytes += g
            c, g = object_size_info(v, seen)
            cpu_bytes += c; gpu_bytes += g

    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            c, g = object_size_info(v, seen)
            cpu_bytes += c; gpu_bytes += g

    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            c, g = object_size_info(v, seen)
            cpu_bytes += c; gpu_bytes += g

    else:
        # Regular Python object
        cpu_bytes += sys.getsizeof(obj)

    return cpu_bytes, gpu_bytes


def main():
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument('--mxtt', type=int, default=48)
    custom_parser.add_argument('--mxet', type=int, default=72)
    # custom_parser.add_argument('--debug', type=bool, default=False)
    custom_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    
    custom_parser.add_argument('--custom_neg', type=bool, default=False)
    custom_parser.add_argument('--deliver_to', type=str, default='self')
    custom_parser.add_argument('--decoder', type=str, default='fc')
    custom_parser.add_argument('--embedding', type=str, default='gat')
    custom_parser.add_argument('--val_neg', type=int, default=-1)
    # custom_parser.add_argument('--ns', type=bool, default=True)
    custom_parser.add_argument('--no-ns', dest='ns', action='store_false', help='Disable negative sampling')
    custom_parser.add_argument('--chunk_size', type=int, default=256)
    custom_parser.add_argument('--skip_cnt', type=int, default=16)
    custom_parser.add_argument('--m_pass', type=int, default=3)

    custom_args, remaining_argv = custom_parser.parse_known_args()

    # Step 2: Replace sys.argv with only recognized args for get_args
    sys.argv = [sys.argv[0]] + remaining_argv
    args, _ = get_args()
    args.mxtt = custom_args.mxtt
    args.mxet = custom_args.mxet
    args.debug = custom_args.debug
    args.custom_neg = custom_args.custom_neg
    args.deliver_to = custom_args.deliver_to
    args.decoder = custom_args.decoder
    args.embedding = custom_args.embedding
    args.val_neg = custom_args.val_neg
    args.load_ns = custom_args.ns
    args.chunk_size = custom_args.chunk_size
    args.skip_cnt = custom_args.skip_cnt
    args.m_pass = custom_args.m_pass 

    # args.num_epoch =  1000
    args.num_run = 1
    args.patience = args.num_epoch

    print("INFO: Arguments:", args)

    DATA = args.data
    LR =args.lr# max(args.lr, (args.lr*args.bs)/200)
    BATCH_SIZE = args.bs
    K_VALUE = args.k_value  
    NUM_EPOCH = args.num_epoch
    SEED = args.seed
    MEM_DIM = args.mem_dim
    TIME_DIM = args.time_dim
    EMB_DIM = args.emb_dim
    TOLERANCE = args.tolerance
    PATIENCE = args.patience
    NUM_RUNS = args.num_run
    NUM_NEIGHBORS = K_VALUE
    MODEL_NAME = 'SDA-TGN'
    MAX_TR_TIME = args.mxtt*60*60#12*60*60
    MAX_EXEC_TIME = args.mxet*60*60
    debug = args.debug
    APAN = args.deliver_to == 'neighbor'

    if args.decoder == "NCN":
        HOP_NUM = 2
        NCN_MODE = 2
    # ==========
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # breakpoint()
    dataset = read_data(DATA, BATCH_SIZE, load_neg_sampler = args.load_ns)
    data = dataset['data']
    unique_destination_nodes =  torch.unique(data.dst)
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    if args.custom_neg:
        neg_dest_sampler = NegLinkSamplerDest(unique_destination_nodes)
    else:
        neg_dest_sampler = None


    chunk_size = args.chunk_size
    items = torch.cat([data.src, data.dst])
    # breakpoint()
    unique_elements, counts = torch.unique(items, return_counts=True)
    max_freq = counts.max().item()
    max_chunk_per_node = 1+max_freq//chunk_size
    # breakpoint()
    print("Converting data to tci data.")
    start_epoch_train = timeit.default_timer()
    tci_data = preprocessor.preprocess(
        data.src.tolist(),
        data.dst.tolist(),#dst_list,
        data.t.double().tolist(),#ts_list,
        torch.arange(data.src.shape[0]).tolist(),#eid_list,
        data.num_nodes,
        chunk_size,
        max_chunk_per_node
    )
    cpu_size, gpu_size = deep_size_info(tci_data)
    print(f"CPU size: {cpu_size/1024/1024:.2f} MB")
    print(f"GPU size: {gpu_size/1024/1024:.2f} MB")

    # breakpoint()
    print(f"Done. Conversion  Time (s): {timeit.default_timer() - start_epoch_train: .4f}")

    # breakpoint()
    sampler = Recent_K_Sampler(tci_data, max_chunk_per_node, K_VALUE, data.num_nodes, apan = APAN, skip_cnt = args.skip_cnt)
    cpu_size, gpu_size = object_size_info(sampler)
    print(f"CPU size: {cpu_size/1024/1024:.2f} MB")
    print(f"GPU size: {gpu_size/1024/1024:.2f} MB")
    breakpoint()
       # for saving the results...
    results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
    if not osp.exists(results_path):
        os.mkdir(results_path)
        print('INFO: Create directory {}'.format(results_path))
    Path(results_path).mkdir(parents=True, exist_ok=True)
    results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_{BATCH_SIZE}_results.json'

    for run_idx in range(NUM_RUNS):
        print('-------------------------------------------------------------------------------')
        print(f"INFO: >>>>> Run: {run_idx} <<<<<")
        start_run = timeit.default_timer()

        # set the seed for deterministic results...
        torch.manual_seed(run_idx + SEED)
        set_random_seed(run_idx + SEED)
        if args.deliver_to == "self":
            memory = DAATGNMemory(
                data.num_nodes,
                data.msg.size(-1),
                MEM_DIM,
                TIME_DIM,
                message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
                aggregator_module=Agg(emb_dim=data.msg.size(-1) + 2 * MEM_DIM + TIME_DIM),
                layer = args.m_pass,
            ).to(device)
        else:
            memory = DA_APANMemory(
                data.num_nodes,
                data.msg.size(-1),
                MEM_DIM,
                TIME_DIM,
                message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
                aggregator_module=Agg(emb_dim=data.msg.size(-1) + 2 * MEM_DIM + TIME_DIM),
                layer =args.m_pass,
            ).to(device)
        if args.embedding == "time_emb":
            gnn = TimeEmbedding(
                in_channels=MEM_DIM,
                out_channels=EMB_DIM,
            ).to(device)
        else:
            gnn = GraphAttentionEmbedding(
                in_channels=MEM_DIM,
                out_channels=EMB_DIM,
                msg_dim=data.msg.size(-1),
                time_enc=memory.time_enc,
            ).to(device)
        if args.decoder == "NCN":
            hidden_channels = 256
            link_pred = NCNPredictor(in_channels=EMB_DIM, hidden_channels=hidden_channels,
                         out_channels=1, NCN_mode=NCN_MODE).to(device)
        else:
            link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

        model = {'memory': memory,
                'gnn': gnn,
                'link_pred': link_pred}

        optimizer = torch.optim.Adam(
            set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
            lr=LR,
        )
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # decay LR by 0.1 every 10 epochs


        criterion = torch.nn.BCEWithLogitsLoss()

        # Helper vector to map global node indices to local ones.
        # assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

        # define an early stopper
        save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
        save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                        tolerance=TOLERANCE, patience=PATIENCE)
        if args.data == "superuser":
            known_dsts = unique_destination_nodes
        else:
            known_dsts = None
        targs = {
            'model': model,
            'optimizer':optimizer,
            'criterion':criterion,
            'dataset': dataset,
            # 'assoc': assoc,
            'min_dst_idx': min_dst_idx,
            'max_dst_idx': max_dst_idx,
            'device': device,
            'sampler': sampler,
            'neg_sampler': neg_dest_sampler,
            'deliver_to': args.deliver_to,
            'decoder': args.decoder,
            'embedding': args.embedding,
            'val_neg': args.val_neg,
            'known_dsts': known_dsts,
        }
        val_perf_list = []
        start_train_val = timeit.default_timer()
        losses = []
        tims = []
        t_tims = 0
        e_tims = 0
        mrrs = []
        for epoch in range(1, NUM_EPOCH + 1):
            # training
            start_epoch_train = timeit.default_timer()
            loss, max_seen_eid = train_sample_only(targs, -1)
            tim = timeit.default_timer() - start_epoch_train
            print(f"Train Sampling Time (s): {tim: .4f}")
            # print(
            #     f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
            # )

            # tims.append(tim)
            # losses.append(loss)
            t_tims += tim

        s_time = t_tims/NUM_EPOCH
        print(f"Average Train Sampling Time (s): {s_time: .4f}")

        #     if not debug:
            
        #         perf_metric_val, max_seen_id = test(targs, max_seen_eid, split_mode="val")
        #         print(f"\tValidation {dataset['metric']}: {perf_metric_val: .4f}")
        #         # # print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        #         val_perf_list.append(perf_metric_val)
        #         # check for early stopping
                
        #         if early_stopper.step_check(perf_metric_val, model):
        #             break
        #         if t_tims > MAX_TR_TIME:
        #             break

        #         e_tims += timeit.default_timer() - start_epoch_train
        #         if e_tims > MAX_EXEC_TIME:
        #             break
            
        # # train_val_time = timeit.default_timer() - start_train_val
        # print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")
        # print("'mrr' : ", val_perf_list, ",")
        # print("'loss' : ",losses,",")
        # print("'time' : ",tims, ",")
        # # ==================================================== Test
        # if not debug:
        #     # first, load the best model
        #     early_stopper.load_checkpoint(model)
        #     # final testing
        #     start_test = timeit.default_timer()
        #     perf_metric_test, max_seen_eid = test(targs, max_seen_id, split_mode="test")

        #     print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        #     print(f"\tTest: {dataset['metric']}: {perf_metric_test: .4f}")


if __name__ == "__main__":
    main()