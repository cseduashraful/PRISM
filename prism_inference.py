import json
import os.path as osp
from types import SimpleNamespace

import torch

from modules.data_utils import read_data
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
def build_model(args: SimpleNamespace, dataset: dict, device: torch.device) -> dict:
    data = dataset["data"]

    # Memory module
    if args.deliver_to == "self":
        memory = DAATGNMemory(
            data.num_nodes,
            data.msg.size(-1),
            args.mem_dim,
            args.time_dim,
            message_module=IdentityMessage(
                data.msg.size(-1), args.mem_dim, args.time_dim
            ),
            aggregator_module=Agg(
                emb_dim=data.msg.size(-1) + 2 * args.mem_dim + args.time_dim
            ),
            layer=args.m_pass,
        ).to(device)
    else:
        memory = DA_APANMemory(
            data.num_nodes,
            data.msg.size(-1),
            args.mem_dim,
            args.time_dim,
            message_module=IdentityMessage(
                data.msg.size(-1), args.mem_dim, args.time_dim
            ),
            aggregator_module=Agg(
                emb_dim=data.msg.size(-1) + 2 * args.mem_dim + args.time_dim
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
            msg_dim=data.msg.size(-1),
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

    load_ns = getattr(args, "load_ns", True)
    dataset = read_data(args.data, args.bs, load_neg_sampler=load_ns)

    model = build_model(args, dataset, device)
    load_checkpoint(model, ckpt_path, device)

    return args, model, dataset, device


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    save_model_dir = "/path/to/saved_models"
    save_model_id = "SDA-TGN_wikipedia_1_0"

    args, model, dataset, device = load_for_inference(
        save_model_dir, save_model_id
    )

    print("Model loaded for inference")
    print("Device:", device)
    print("Dataset:", args.data)
