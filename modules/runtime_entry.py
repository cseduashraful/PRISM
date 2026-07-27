import argparse
import hashlib
import os
import sys
import timeit
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from tgb.utils.utils import get_args

from modules.data_utils import read_data
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding, TimeEmbedding
from modules.memory_module import DAATGNMemory, DA_APANMemory
from modules.msg_agg import MeanAggregator as Agg
from modules.msg_func import IdentityMessage
from modules.neg_sampler import NegLinkSamplerDest
from modules.NCNDecoder.NCNPred import NCNPredictor
from modules.sampler_builder import SamplerBuildSpec, build_sampler_backend


@dataclass
class PrismRuntimeConfig:
    args: Any
    data: str
    lr: float
    batch_size: int
    k_value: int
    num_epoch: int
    seed: int
    mem_dim: int
    time_dim: int
    emb_dim: int
    tolerance: float
    patience: int
    num_runs: int
    model_name: str
    max_train_seconds: int
    max_exec_seconds: int
    debug: bool
    apan: bool


@dataclass
class DatasetBundle:
    dataset: Dict[str, Any]
    data: Any
    data_cache: Dict[str, Optional[torch.Tensor]]
    unique_destination_nodes: torch.Tensor
    min_dst_idx: int
    max_dst_idx: int
    neg_dest_sampler: Optional[NegLinkSamplerDest]
    known_dsts: Optional[torch.Tensor]
    max_chunk_per_node: int


@dataclass
class SamplerBundle:
    sampler: Any
    backend: str
    build_time_seconds: float


@dataclass
class ModelBundle:
    memory: torch.nn.Module
    gnn: torch.nn.Module
    link_pred: torch.nn.Module
    model: Dict[str, torch.nn.Module]
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module


def parse_runtime_config() -> PrismRuntimeConfig:
    custom_parser = argparse.ArgumentParser(add_help=False)
    custom_parser.add_argument("--mxtt", type=int, default=48)
    custom_parser.add_argument("--mxet", type=int, default=72)
    custom_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    custom_parser.add_argument("--custom_neg", type=bool, default=False)
    custom_parser.add_argument("--deliver_to", type=str, default="self")
    custom_parser.add_argument("--decoder", type=str, default="fc")
    custom_parser.add_argument("--embedding", type=str, default="gat")
    custom_parser.add_argument("--val_neg", type=int, default=-1)
    custom_parser.add_argument("--no-ns", dest="ns", action="store_false", help="Disable negative sampling")
    custom_parser.add_argument("--chunk_size", type=int, default=256)
    custom_parser.add_argument("--skip_cnt", type=int, default=16)
    custom_parser.add_argument("--m_pass", type=int, default=3)
    custom_parser.add_argument(
        "--offload-mode",
        choices=["auto", "off", "on"],
        default="auto",
        help=(
            "Sampler storage mode: 'off' keeps preprocessing in-memory, "
            "'on' forces disk offloading, 'auto' tries in-memory and falls back to "
            "disk offloading when memory is insufficient."
        ),
    )
    custom_parser.add_argument(
        "--auto-offload-gpu-mem-pct",
        type=float,
        default=-1.0,
        help=(
            "When offload-mode=auto, switch from in-memory sampler to disk-offload "
            "once current GPU memory usage reaches this percent (0-100). "
            "Set <0 to disable runtime threshold switching."
        ),
    )
    custom_parser.add_argument(
        "--tensor-store-mode",
        choices=["off", "on", "verify"],
        default="on",
        help=(
            "Message-store backend for DAATGNMemory: "
            "'on' enables optimized tensor-backed path (default), "
            "'verify' checks optimized vs reference parity at runtime, "
            "'off' uses reference path only."
        ),
    )
    custom_parser.add_argument(
        "--cache-data-on-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache frequently indexed dataset tensors (t/msg/src) on GPU once at startup "
            "to reduce per-iteration host->device transfers."
        ),
    )

    custom_args, remaining_argv = custom_parser.parse_known_args()
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
    args.offload_mode = custom_args.offload_mode
    args.auto_offload_gpu_mem_pct = custom_args.auto_offload_gpu_mem_pct
    args.tensor_store_mode = custom_args.tensor_store_mode
    args.cache_data_on_gpu = custom_args.cache_data_on_gpu
    args.num_run = 1
    args.patience = args.num_epoch

    os.environ["PRISM_TENSOR_STORE_MODE"] = args.tensor_store_mode

    return PrismRuntimeConfig(
        args=args,
        data=args.data,
        lr=args.lr,
        batch_size=args.bs,
        k_value=args.k_value,
        num_epoch=args.num_epoch,
        seed=args.seed,
        mem_dim=args.mem_dim,
        time_dim=args.time_dim,
        emb_dim=args.emb_dim,
        tolerance=args.tolerance,
        patience=args.patience,
        num_runs=args.num_run,
        model_name="SDA-TGN",
        max_train_seconds=args.mxtt * 60 * 60,
        max_exec_seconds=args.mxet * 60 * 60,
        debug=args.debug,
        apan=args.deliver_to == "neighbor",
    )


def make_run_tag(config: PrismRuntimeConfig) -> str:
    job_hint = os.environ.get("SLURM_JOB_ID", "")
    run_fingerprint = "|".join(
        [
            config.data,
            str(config.seed),
            str(os.getpid()),
            str(timeit.default_timer()),
            job_hint,
        ]
    )
    return hashlib.sha1(run_fingerprint.encode("utf-8")).hexdigest()[:8]


def build_dataset_bundle(config: PrismRuntimeConfig, device: torch.device) -> DatasetBundle:
    dataset = read_data(config.data, config.batch_size, load_neg_sampler=config.args.load_ns)
    data = dataset["data"]
    data_cache = {"t": None, "msg": None, "src": None}
    if config.args.cache_data_on_gpu and device.type == "cuda":
        try:
            data_cache["t"] = data.t.to(device)
            data_cache["msg"] = data.msg.to(device)
            data_cache["src"] = data.src.to(device)
            print("INFO: Cached dataset tensors on GPU (t/msg/src).")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(
                "WARNING: Could not cache dataset tensors on GPU (OOM). "
                "Falling back to per-iteration CPU->GPU copies."
            )
            torch.cuda.empty_cache()
    elif config.args.cache_data_on_gpu and device.type != "cuda":
        print("INFO: cache-data-on-gpu requested but CUDA is unavailable; using CPU tensors.")

    unique_destination_nodes = torch.unique(data.dst)
    items = torch.cat([data.src, data.dst])
    _, counts = torch.unique(items, return_counts=True)
    max_chunk_per_node = 1 + counts.max().item() // config.args.chunk_size

    return DatasetBundle(
        dataset=dataset,
        data=data,
        data_cache=data_cache,
        unique_destination_nodes=unique_destination_nodes,
        min_dst_idx=int(data.dst.min()),
        max_dst_idx=int(data.dst.max()),
        neg_dest_sampler=NegLinkSamplerDest(unique_destination_nodes) if config.args.custom_neg else None,
        known_dsts=unique_destination_nodes if config.data == "superuser" else None,
        max_chunk_per_node=max_chunk_per_node,
    )


def build_sampler(
    config: PrismRuntimeConfig, dataset_bundle: DatasetBundle, device: torch.device
) -> SamplerBundle:
    print("Converting data to tci data.")
    start_time = timeit.default_timer()
    result = build_sampler_backend(
        dataset_bundle.data,
        SamplerBuildSpec(
            k=config.k_value,
            chunk_size=config.args.chunk_size,
            max_chunk_per_node=dataset_bundle.max_chunk_per_node,
            offload_mode=config.args.offload_mode,
            cache_size=config.batch_size,
            device=device,
            apan=config.apan,
            skip_cnt=config.args.skip_cnt,
        ),
    )
    return SamplerBundle(
        sampler=result.sampler,
        backend=result.backend,
        build_time_seconds=timeit.default_timer() - start_time,
    )


def build_model_bundle(
    config: PrismRuntimeConfig,
    dataset_bundle: DatasetBundle,
    device: torch.device,
) -> ModelBundle:
    data = dataset_bundle.data

    if config.args.deliver_to == "self":
        memory = DAATGNMemory(
            data.num_nodes,
            data.msg.size(-1),
            config.mem_dim,
            config.time_dim,
            message_module=IdentityMessage(data.msg.size(-1), config.mem_dim, config.time_dim),
            aggregator_module=Agg(
                emb_dim=data.msg.size(-1) + 2 * config.mem_dim + config.time_dim
            ),
            layer=config.args.m_pass,
        ).to(device)
    else:
        memory = DA_APANMemory(
            data.num_nodes,
            data.msg.size(-1),
            config.mem_dim,
            config.time_dim,
            message_module=IdentityMessage(data.msg.size(-1), config.mem_dim, config.time_dim),
            aggregator_module=Agg(
                emb_dim=data.msg.size(-1) + 2 * config.mem_dim + config.time_dim
            ),
            layer=config.args.m_pass,
        ).to(device)

    if config.args.embedding == "time_emb":
        gnn = TimeEmbedding(
            in_channels=config.mem_dim,
            out_channels=config.emb_dim,
        ).to(device)
    else:
        gnn = GraphAttentionEmbedding(
            in_channels=config.mem_dim,
            out_channels=config.emb_dim,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)

    if config.args.decoder == "NCN":
        link_pred = NCNPredictor(
            in_channels=config.emb_dim,
            hidden_channels=256,
            out_channels=1,
            NCN_mode=2,
        ).to(device)
    else:
        link_pred = LinkPredictor(in_channels=config.emb_dim).to(device)

    model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}
    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=config.lr,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    return ModelBundle(
        memory=memory,
        gnn=gnn,
        link_pred=link_pred,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
    )


def build_train_args(
    config: PrismRuntimeConfig,
    model_bundle: ModelBundle,
    dataset_bundle: DatasetBundle,
    sampler: Any,
    device: torch.device,
) -> Dict[str, Any]:
    return {
        "model": model_bundle.model,
        "optimizer": model_bundle.optimizer,
        "criterion": model_bundle.criterion,
        "dataset": dataset_bundle.dataset,
        "min_dst_idx": dataset_bundle.min_dst_idx,
        "max_dst_idx": dataset_bundle.max_dst_idx,
        "device": device,
        "sampler": sampler,
        "neg_sampler": dataset_bundle.neg_dest_sampler,
        "deliver_to": config.args.deliver_to,
        "decoder": config.args.decoder,
        "embedding": config.args.embedding,
        "val_neg": config.args.val_neg,
        "known_dsts": dataset_bundle.known_dsts,
        "data_cache": dataset_bundle.data_cache,
    }
