import json
import os.path as osp
from types import SimpleNamespace

import torch

from modules.data_utils import read_data, save_tci_data, load_tci_data
from modules.memory_module import DAATGNMemory, DA_APANMemory
from modules.msg_func import IdentityMessage
from modules.msg_agg import MeanAggregator as Agg
from modules.emb_module import GraphAttentionEmbedding, TimeEmbedding
from modules.decoder import LinkPredictor
from modules.NCNDecoder.NCNPred import NCNPredictor

# --------------------------------------------------
# Load args
# --------------------------------------------------
def load_args_json(args_path: str) -> SimpleNamespace:
    with open(args_path, "r") as f:
        return SimpleNamespace(**json.load(f))


# --------------------------------------------------
# Build model (must match training exactly)
# --------------------------------------------------
def build_model(args: SimpleNamespace, device: torch.device) -> dict:
    # data = dataset["data"]

    # Memory module
    if args.deliver_to == "self":
        memory = DAATGNMemory(
            args.num_nodes,
            args.msg_size,
            args.mem_dim,
            args.time_dim,
            message_module=IdentityMessage(
                args.msg_size, args.mem_dim, args.time_dim
            ),
            aggregator_module=Agg(
                emb_dim=args.msg_size + 2 * args.mem_dim + args.time_dim
            ),
            layer=args.m_pass,
        ).to(device)
    else:
        memory = DA_APANMemory(
            args.num_nodes,
            args.msg_size,
            args.mem_dim,
            args.time_dim,
            message_module=IdentityMessage(
                args.msg_size, args.mem_dim, args.time_dim
            ),
            aggregator_module=Agg(
                emb_dim=args.msg_size + 2 * args.mem_dim + args.time_dim
            ),
            layer=args.m_pass,
        ).to(device)

    # Embedding / GNN
    if args.embedding == "time_emb":
        gnn = TimeEmbedding(
            in_channels=args.mem_dim,
            out_channels=args.emb_dim,
        ).to(device)
    else:
        gnn = GraphAttentionEmbedding(
            in_channels=args.mem_dim,
            out_channels=args.emb_dim,
            msg_dim=args.msg_size,
            time_enc=memory.time_enc,
        ).to(device)

    # Decoder
    if args.decoder == "NCN":
        hidden_channels = 256
        NCN_MODE = 2
        link_pred = NCNPredictor(
            in_channels=args.emb_dim,
            hidden_channels=hidden_channels,
            out_channels=1,
            NCN_mode=NCN_MODE,
        ).to(device)
    else:
        link_pred = LinkPredictor(in_channels=args.emb_dim).to(device)

    return {
        "memory": memory,
        "gnn": gnn,
        "link_pred": link_pred,
    }


# --------------------------------------------------
# Load checkpoint (saved by EarlyStopMonitor)
# --------------------------------------------------
def load_checkpoint(model: dict, ckpt_path: str, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device)

    model["memory"].load_state_dict(checkpoint["memory"], strict=True)
    model["gnn"].load_state_dict(checkpoint["gnn"], strict=True)
    model["link_pred"].load_state_dict(checkpoint["link_pred"], strict=True)

    for m in model.values():
        m.eval()


# --------------------------------------------------
# End-to-end: load everything for inference
# --------------------------------------------------
def load_for_inference(save_model_dir: str, save_model_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args_path = osp.join(save_model_dir, f"{save_model_id}.args.json")
    ckpt_path = osp.join(save_model_dir, f"{save_model_id}.pth")

    args = load_args_json(args_path)

    
    # dataset = read_data(args.data, args.bs, load_neg_sampler=load_ns)

    model = build_model(args, device)
    load_checkpoint(model, ckpt_path, device)

    return args, model, device



def get_test_args(model, dataset, sampler):
    if args.data == "superuser":
        known_dsts = unique_destination_nodes
    else:
        known_dsts = None
    targs = {
        'model': model,
            # 'optimizer':optimizer,
            # 'criterion':criterion,
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
    return targs

# --------------------------------------------------
# Example usage
# --------------------------------------------------
import argparse
if __name__ == "__main__":
    # save_model_dir = "/path/to/saved_models"
    # save_model_id = "SDA-TGN_wikipedia_1_0"
    id = "2dd9a193"
    custom_parser = argparse.ArgumentParser(add_help=False)
    # custom_parser.add_argument('--mxtt', type=int, default=48)
    # custom_parser.add_argument('--mxet', type=int, default=72)
    # # custom_parser.add_argument('--debug', type=bool, default=False)
    # custom_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    
    # custom_parser.add_argument('--custom_neg', type=bool, default=False)
    # custom_parser.add_argument('--deliver_to', type=str, default='self')
    # custom_parser.add_argument('--decoder', type=str, default='fc')
    custom_parser.add_argument('--dir', type=str, default=f'saved_models/cache_{id}')
    custom_parser.add_argument('--id', type=str, default='SDA-TGN_tgbl-wiki_1_0')
    custom_args, remaining_argv = custom_parser.parse_known_args()
    
    save_model_dir = custom_args.dir
    save_model_id = custom_args.id

    # breakpoint()

    args, model, device = load_for_inference(
        save_model_dir, save_model_id
    )

    print("Model loaded for inference")
    print("Device:", device)
    print("Dataset:", args.data)
  
    dataset = read_data(args.data, args.bs, load_neg_sampler = args.load_ns)
    data = dataset['data']
    unique_destination_nodes =  torch.unique(data.dst)
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    from modules.neg_sampler import NegLinkSamplerDest
    if args.custom_neg:
        neg_dest_sampler = NegLinkSamplerDest(unique_destination_nodes)
    else:
        neg_dest_sampler = None


    # chunk_size = args.chunk_size
    # # breakpoint()

    max_seen_eid = 133852#int(data.src.size(0)*.86)
    trvl = max_seen_eid+1
    # trvl = int(data.src.size(0))


    # items = torch.cat([data.src[:trvl], data.dst[:trvl]])
    # # breakpoint()
    # unique_elements, counts = torch.unique(items, return_counts=True)
    # max_freq = counts.max().item()
    # max_chunk_per_node = 1+max_freq//chunk_size
    # # breakpoint()
    # print("Converting data to tci data.")

    import timeit
    # # import preprocessor
    import chunkio

    # start_epoch_train = timeit.default_timer()
    # from uuid import uuid4
    # folder_name = f"cache_{uuid4().hex[:8]}"
    # outdir = "inference_preproc_out/"+folder_name


    # tci_data_old = chunkio.preprocess_streaming(
    #     data.src[:trvl].tolist(),
    #     data.dst[:trvl].tolist(),#dst_list,
    #     data.t[:trvl].double().tolist(),#ts_list,
    #     torch.arange(data.src[:trvl].shape[0]).tolist(),#eid_list,
    #     data.num_nodes,
    #     chunk_size=chunk_size,
    #     max_chunk_per_node=max_chunk_per_node,
    #     duplicate_undirected=True,
    #     out_dir=outdir,#"preproc_out",
    #     num_shards=256,
    # )

    # print(f"Done. Conversion  Time (s): {timeit.default_timer() - start_epoch_train: .4f}")

    # save_tci_data(tci_data_old, "inference_preproc_out/cache_2dd9a193tci.pkl")
    # breakpoint()
    
    tci_data = load_tci_data(f"inference_preproc_out/cache_{id}tci.pkl")
    outdir = f"inference_preproc_out/cache_{id}"
    # breakpoint()

    # later edges (later timestamps)
    tci_data = chunkio.extend_streaming_latestk_ordered_reuse(
        tci_data,
        data.src[trvl:].tolist(),
        data.dst[trvl:].tolist(),
        data.t[trvl:].double().tolist(),
        torch.arange(data.src[trvl:].shape[0]).tolist(),
        duplicate_undirected=True,
    )

    # breakpoint()




    max_chunk_per_node = len(tci_data.chunk_map[0])
    from modules.train_utils import test_new as test
    from modules.grnstream import GRN_Stream
    sampler = GRN_Stream(tci_data, max_chunk_per_node, args.k_value, args.num_nodes, args.chunk_size, outdir, cache_size = args.bs, apan =  args.deliver_to == 'neighbor', skip_cnt = args.skip_cnt)
    # # sampler = Recent_K_Sampler(tci_data, max_chunk_per_node, args.k_value, data.num_nodes, apan =  args.deliver_to == 'neighbor', skip_cnt = args.skip_cnt)
    start_test = timeit.default_timer()
    targs =  get_test_args(model, dataset, sampler)
    perf_metric_test, max_seen_eid = test(targs, max_seen_eid, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: MRR: {perf_metric_test: .4f}")




