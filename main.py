from modules.data_utils import read_data #, get_TCSR, get_TCSR_py, verify_tcsr
from modules.recent_sampler import Recent_K_Sampler
from modules.train_utils import train as actrain, test, train_with_custom_neg_sampler
from modules.memory_module import DAATGNMemory

from modules.neg_sampler import NegLinkSamplerDest
from modules.emb_module import GraphAttentionEmbedding
from modules.early_stopping import EarlyStopMonitor
from modules.msg_agg import LastAggregator, MeanAggregator, AttentionAggregator as Agg, TransformerAggregator
from modules.msg_func import IdentityMessage, MLPMessage
from modules.decoder import LinkPredictor

# from torch.optim.lr_scheduler import StepLR

from tgb.utils.utils import get_args, set_random_seed, save_results
import numpy as np
import torch

import timeit
import os
import os.path as osp
from pathlib import Path

import preprocessor #openmp


    

def main():
    args, _ = get_args()
    DATA = args.data
    print("INFO: Arguments:", args)

    LR = max(args.lr, (args.lr*args.bs)/200)
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
    MODEL_NAME = 'TGN'
    # ==========
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = read_data(DATA, BATCH_SIZE, load_neg_sampler = True)
    data = dataset['data']
    unique_destination_nodes =  torch.unique(data.dst)
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    neg_dest_sampler = NegLinkSamplerDest(unique_destination_nodes)


    chunk_size = 256
    items = torch.cat([data.src, data.dst])
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
    print(f"Done. Conversion  Time (s): {timeit.default_timer() - start_epoch_train: .4f}")


    # output = tci_data
    # chunk_map = torch.tensor(output['chunk_map'], dtype=torch.long)
    # # Count valid chunks per node (non -1 values)
    # valid_chunk_counts = (chunk_map != -1).sum(dim=1)

    # # Get min/max values and corresponding node indices
    # min_chunks = valid_chunk_counts.min().item()
    # max_chunks = valid_chunk_counts.max().item()

    # min_nodes = (valid_chunk_counts == min_chunks).nonzero(as_tuple=True)[0].tolist()
    # max_nodes = (valid_chunk_counts == max_chunks).nonzero(as_tuple=True)[0].tolist()

    # print(f"[Info] Valid chunk counts per node:\n{valid_chunk_counts.tolist()}")
    # print(f"[Info] Minimum valid chunks: {min_chunks}, Nodes: {min_nodes}")
    # print(f"[Info] Maximum valid chunks: {max_chunks}, Nodes: {max_nodes}")

    # breakpoint()



    # breakpoint()
    sampler = Recent_K_Sampler(tci_data, max_chunk_per_node, K_VALUE, data.num_nodes)
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
        memory = DAATGNMemory(
            data.num_nodes,
            data.msg.size(-1),
            MEM_DIM,
            TIME_DIM,
            message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
            aggregator_module=Agg(emb_dim=data.msg.size(-1) + 2 * MEM_DIM + TIME_DIM),
        ).to(device)

        gnn = GraphAttentionEmbedding(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)

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
        }
        val_perf_list = []
        start_train_val = timeit.default_timer()
        for epoch in range(1, NUM_EPOCH + 1):
            # training
            start_epoch_train = timeit.default_timer()
            loss = actrain(targs)
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
            )
            perf_metric_val = test(targs, split_mode="val")
            print(f"\tValidation {dataset['metric']}: {perf_metric_val: .4f}")
            # print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
            val_perf_list.append(perf_metric_val)
            # check for early stopping
            if early_stopper.step_check(perf_metric_val, model):
                break
            
        train_val_time = timeit.default_timer() - start_train_val
        print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

        # ==================================================== Test
        # first, load the best model
        early_stopper.load_checkpoint(model)
         # final testing
        start_test = timeit.default_timer()
        perf_metric_test = test(split_mode="test")

        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        print(f"\tTest: {dataset['metric']}: {perf_metric_test: .4f}")
        test_time = timeit.default_timer() - start_test
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

        save_results({'model': MODEL_NAME,
                    'data': DATA,
                    'run': run_idx,
                    'seed': SEED,
                    f'val {dataset["metric"]}': val_perf_list,
                    f'test {dataset["metric"]}': perf_metric_test,
                    'test_time': test_time,
                    'tot_train_val_time': train_val_time
                    }, 
        results_filename)

        print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
        print('-------------------------------------------------------------------------------')








    # breakpoint()


    


if __name__ == "__main__":
    main()