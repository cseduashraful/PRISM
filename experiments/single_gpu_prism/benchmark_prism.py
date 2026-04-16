#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
TGB_REPO_ROOT = WORKSPACE_ROOT / "tgb"

REPO_ROOT_STR = str(REPO_ROOT)
TGB_REPO_ROOT_STR = str(TGB_REPO_ROOT)

for path_str in [REPO_ROOT_STR, TGB_REPO_ROOT_STR]:
    while path_str in sys.path:
        sys.path.remove(path_str)

# Keep PRISM ahead of TGB so `from modules...` resolves to PRISM's modules.
sys.path.insert(0, REPO_ROOT_STR)
if TGB_REPO_ROOT.exists():
    sys.path.insert(1, TGB_REPO_ROOT_STR)

PRISM_IMPORT_ERROR: Optional[Exception] = None
try:
    import preprocessor  # type: ignore
    from modules.data_utils import read_data  # type: ignore
    from modules.decoder import LinkPredictor  # type: ignore
    from modules.emb_module import GraphAttentionEmbedding, TimeEmbedding  # type: ignore
    from modules.memory_module import DAATGNMemory, DA_APANMemory  # type: ignore
    from modules.msg_agg import MeanAggregator as Agg  # type: ignore
    from modules.msg_func import IdentityMessage  # type: ignore
    from modules.NCNDecoder.NCNPred import NCNPredictor  # type: ignore
    from modules.recent_sampler import Recent_K_Sampler  # type: ignore
    from modules.train_utils import train_neg_sampler, vectorized_getMem_graph  # type: ignore
    from tgb.utils.utils import set_random_seed  # type: ignore
except Exception as exc:  # pragma: no cover - import-time environment dependent
    PRISM_IMPORT_ERROR = exc
    preprocessor = None
    read_data = None
    LinkPredictor = None
    GraphAttentionEmbedding = None
    TimeEmbedding = None
    DAATGNMemory = None
    DA_APANMemory = None
    Agg = None
    IdentityMessage = None
    NCNPredictor = None
    Recent_K_Sampler = None
    train_neg_sampler = None
    vectorized_getMem_graph = None
    set_random_seed = None


SYSTEM_NAME = "prism"
MODEL_SPECS = {
    "tgn": {"deliver_to": "self", "decoder": "fc", "embedding": "gat"},
    "apan": {"deliver_to": "neighbor", "decoder": "fc", "embedding": "gat"},
    "tncn": {"deliver_to": "self", "decoder": "NCN", "embedding": "gat"},
    "jodie": {"deliver_to": "self", "decoder": "fc", "embedding": "time_emb"},
}
DATASET_ALIASES = {
    "tgbl-wiki": ("tgbl-wiki", "tgbl-wiki"),
    "wiki": ("tgbl-wiki", "tgbl-wiki"),
    "WIKI": ("tgbl-wiki", "tgbl-wiki"),
    "tgbl-reddit": ("reddit", "tgbl-reddit"),
    "reddit": ("reddit", "tgbl-reddit"),
    "REDDIT": ("reddit", "tgbl-reddit"),
    "tgbl-lastfm": ("tgbl-lastfm", "tgbl-lastfm"),
    "lastfm": ("lastfm", "tgbl-lastfm"),
    "LASTFM": ("lastfm", "tgbl-lastfm"),
}


@dataclass
class IterationRecord:
    run_id: str
    system: str
    model: str
    dataset: str
    seed: int
    batch_size: int
    num_neighbors: int
    num_layers: int
    epoch: int
    epoch_iteration: int
    global_iteration: int
    measured_iteration: int
    total_iteration_time_s: float
    sampling_time_s: float
    feature_lookup_preprocess_time_s: float
    cpu_to_gpu_transfer_time_s: float
    forward_time_s: float
    memory_module_time_s: float
    emb_module_time_s: float
    decoder_time_s: float
    backward_time_s: float
    memory_update_time_s: float
    sampler_state_update_time_s: float
    optimizer_step_time_s: float
    peak_gpu_memory_mb: float
    memory_module_peak_gpu_memory_mb: float
    emb_module_peak_gpu_memory_mb: float
    decoder_peak_gpu_memory_mb: float
    memory_module_memory_delta_mb: float
    emb_module_memory_delta_mb: float
    decoder_memory_delta_mb: float
    gpu_utilization_pct: float
    throughput_edges_per_s: float
    samples_in_batch: int


@dataclass
class RunSummary:
    run_id: str
    system: str
    model: str
    dataset: str
    seed: int
    batch_size: int
    num_neighbors: int
    num_layers: int
    warmup_iterations: int
    measured_iterations: int
    avg_iteration_time_s: float
    std_iteration_time_s: float
    avg_sampling_time_s: float
    avg_feature_lookup_preprocess_time_s: float
    avg_cpu_to_gpu_transfer_time_s: float
    avg_forward_time_s: float
    avg_memory_module_time_s: float
    avg_emb_module_time_s: float
    avg_decoder_time_s: float
    avg_backward_time_s: float
    avg_memory_update_time_s: float
    avg_sampler_state_update_time_s: float
    avg_optimizer_step_time_s: float
    avg_peak_gpu_memory_mb: float
    peak_gpu_memory_mb: float
    avg_memory_module_peak_gpu_memory_mb: float
    avg_emb_module_peak_gpu_memory_mb: float
    avg_decoder_peak_gpu_memory_mb: float
    avg_memory_module_memory_delta_mb: float
    avg_emb_module_memory_delta_mb: float
    avg_decoder_memory_delta_mb: float
    avg_gpu_utilization_pct: float
    avg_throughput_edges_per_s: float
    total_epoch_time_s: float
    total_train_time_s: float
    final_validation_metric: float
    validation_metric_name: str


def ensure_prism_runtime_available():
    if PRISM_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "PRISM benchmark dependencies are not available. "
        "Build PRISM extensions with `python setup.py build_ext --inplace` inside `PRISM`, "
        "and make sure the TGB package is importable in the active environment. "
        f"Original import error: {PRISM_IMPORT_ERROR}"
    ) from PRISM_IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PRISM models with TGB/TGL-compatible profiling outputs."
    )
    parser.add_argument("--dataset", default="tgbl-wiki")
    parser.add_argument("--models", nargs="+", default=["tgn", "apan", "tncn"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[256])
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--neighbor-size", type=int, default=10)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="PRISM memory refinement passes (`m_pass` in the original scripts).",
    )
    parser.add_argument("--mem-dim", type=int, default=100)
    parser.add_argument("--time-dim", type=int, default=100)
    parser.add_argument("--emb-dim", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--skip-cnt", type=int, default=16)
    parser.add_argument("--sample-gpu-utilization", action="store_true")
    parser.add_argument("--skip-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache-data-on-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache frequently indexed dataset tensors (t/msg/src) on GPU once at startup "
            "to reduce per-iteration host->device transfers."
        ),
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs"))
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if set_random_seed is not None:
        set_random_seed(seed)


def sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_region_start(device: torch.device) -> float:
    sync_if_needed(device)
    return time.perf_counter()


def timed_region_end(start: float, device: torch.device) -> float:
    sync_if_needed(device)
    return time.perf_counter() - start


def memory_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def query_gpu_utilization(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    gpu_index = device.index if device.index is not None else 0
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
        return float(output.splitlines()[0])
    except Exception:
        return float("nan")


def get_system_info(device: torch.device) -> Dict[str, object]:
    cpu_model = platform.processor()
    if not cpu_model and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break

    total_ram_gb = None
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total_ram_gb = int(line.split()[1]) / (1024 ** 2)
                break

    gpu_info: Dict[str, object] = {"available": device.type == "cuda"}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu_info.update(
            {
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024 ** 3),
                "cuda_device_index": device.index if device.index is not None else 0,
            }
        )

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu_model": cpu_model,
        "ram_gb": total_ram_gb,
        "gpu": gpu_info,
    }


def write_json(path: Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_csv(path: Path, rows: Iterable[Dict[str, object]]):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def resolve_dataset_names(dataset_arg: str) -> Tuple[str, str]:
    try:
        return DATASET_ALIASES[dataset_arg]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset '{dataset_arg}'. Supported values: {sorted(DATASET_ALIASES)}"
        ) from exc


def resolve_model_spec(model_name: str) -> Dict[str, str]:
    try:
        return MODEL_SPECS[model_name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported values: {sorted(MODEL_SPECS)}"
        ) from exc


def get_device(args: argparse.Namespace) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PRISM profiling benchmark, but no CUDA device is available.")
    return torch.device("cuda:0")


def build_recent_sampler(data, args: argparse.Namespace, deliver_to: str, device: torch.device):
    if preprocessor is None or Recent_K_Sampler is None:
        ensure_prism_runtime_available()

    items = torch.cat([data.src, data.dst])
    _, counts = torch.unique(items, return_counts=True)
    max_freq = int(counts.max().item()) if counts.numel() else 0
    max_chunk_per_node = 1 + max_freq // args.chunk_size

    tci_data = preprocessor.preprocess(
        data.src.tolist(),
        data.dst.tolist(),
        data.t.double().tolist(),
        torch.arange(data.src.shape[0]).tolist(),
        data.num_nodes,
        args.chunk_size,
        max_chunk_per_node,
    )
    sampler = Recent_K_Sampler(
        tci_data,
        max_chunk_per_node,
        args.neighbor_size,
        data.num_nodes,
        device=device,
        apan=deliver_to == "neighbor",
        skip_cnt=args.skip_cnt,
    )
    return sampler, max_chunk_per_node


def build_model(data, args: argparse.Namespace, spec: Dict[str, str], device: torch.device):
    ensure_prism_runtime_available()

    if spec["deliver_to"] == "self":
        memory = DAATGNMemory(
            data.num_nodes,
            data.msg.size(-1),
            args.mem_dim,
            args.time_dim,
            message_module=IdentityMessage(data.msg.size(-1), args.mem_dim, args.time_dim),
            aggregator_module=Agg(emb_dim=data.msg.size(-1) + 2 * args.mem_dim + args.time_dim),
            layer=args.num_layers,
        ).to(device)
    else:
        memory = DA_APANMemory(
            data.num_nodes,
            data.msg.size(-1),
            args.mem_dim,
            args.time_dim,
            message_module=IdentityMessage(data.msg.size(-1), args.mem_dim, args.time_dim),
            aggregator_module=Agg(emb_dim=data.msg.size(-1) + 2 * args.mem_dim + args.time_dim),
            layer=args.num_layers,
        ).to(device)

    if spec["embedding"] == "time_emb":
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

    if spec["decoder"] == "NCN":
        link_pred = NCNPredictor(
            in_channels=args.emb_dim,
            hidden_channels=256,
            out_channels=1,
            NCN_mode=2,
        ).to(device)
    else:
        link_pred = LinkPredictor(in_channels=args.emb_dim).to(device)

    return {
        "memory": memory,
        "gnn": gnn,
        "link_pred": link_pred,
    }


def build_data_cache(data, args: argparse.Namespace, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    cache: Dict[str, Optional[torch.Tensor]] = {
        "t": None,
        "msg": None,
        "src": None,
    }
    if not args.cache_data_on_gpu:
        return cache

    try:
        cache["t"] = data.t.to(device)
        cache["msg"] = data.msg.to(device)
        cache["src"] = data.src.to(device)
        print("[info] Cached dataset tensors on GPU for PRISM transfer reduction.")
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(
            "[warning] Could not cache full dataset tensors on GPU (OOM). "
            "Falling back to per-iteration CPU->GPU copies."
        )
        torch.cuda.empty_cache()
    return cache


def load_runtime(
    dataset_name: str,
    model_name: str,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
):
    ensure_prism_runtime_available()

    spec = resolve_model_spec(model_name)
    dataset = read_data(dataset_name, batch_size, load_neg_sampler=False)
    data = dataset["data"]
    sampler, max_chunk_per_node = build_recent_sampler(data, args, spec["deliver_to"], device)
    model = build_model(data, args, spec, device)
    data_cache = build_data_cache(data, args, device)
    optimizer = torch.optim.Adam(
        set(model["memory"].parameters())
        | set(model["gnn"].parameters())
        | set(model["link_pred"].parameters()),
        lr=args.lr,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    unique_destination_nodes = torch.unique(data.dst)
    min_dst_idx = int(data.dst.min())
    max_dst_idx = int(data.dst.max())
    known_dsts = unique_destination_nodes if dataset_name == "superuser" else None

    return {
        "dataset": dataset,
        "data": data,
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "sampler": sampler,
        "data_cache": data_cache,
        "spec": spec,
        "min_dst_idx": min_dst_idx,
        "max_dst_idx": max_dst_idx,
        "known_dsts": known_dsts,
        "max_chunk_per_node": max_chunk_per_node,
        "max_seen_eid": -1,
    }


def get_num_neighbors(args: argparse.Namespace) -> int:
    return int(args.neighbor_size)


def get_num_layers(args: argparse.Namespace) -> int:
    return int(args.num_layers)


def make_profiler(
    trace_dir: Path,
    warmup_iters: int,
    measured_iters: int,
    enable_cuda: bool,
):
    activities = [ProfilerActivity.CPU]
    if enable_cuda:
        activities.append(ProfilerActivity.CUDA)
    return profile(
        activities=activities,
        schedule=schedule(
            wait=max(warmup_iters - 1, 0),
            warmup=1 if warmup_iters > 0 else 0,
            active=measured_iters,
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def reset_sampler_state(sampler):
    sampler.assoc = torch.arange(sampler.num_nodes, device=sampler.device)
    sampler.current_prefetch_idx = 0
    sampler.prefetch_buffer_ts = [None, None]
    sampler.prefetch_buffer_eid = [None, None]
    sampler.prefetch_buffer_other_node = [None, None]


def reset_runtime_state(runtime: Dict[str, object]):
    model = runtime["model"]
    model["memory"].reset_state()
    reset_sampler_state(runtime["sampler"])
    runtime["max_seen_eid"] = -1


def make_self_mem_graph(model, sampler, n_id, e_id, edge_index, bs, max_seen_eid, src, pos_dst, device):
    bmsk_cpu = e_id > max_seen_eid
    bmsk = bmsk_cpu.to(device)

    bdst = torch.arange(bs * 2, device=device)
    bsrc = torch.cat([torch.arange(bs, 2 * bs, device=device), torch.arange(bs, device=device)])
    bedge = torch.stack([bsrc, bdst], dim=0)
    bedge_nodes = n_id[torch.cat([bedge[0], bedge[1]], dim=0)]
    bedge_targets = torch.cat([bedge[1], bedge[1]], dim=0)

    bedge_all = model["memory"].mem_graph(bedge_nodes, bedge_targets, src, pos_dst)
    fall_back = sampler.assoc[bedge_nodes]
    updated_ball = torch.where(bedge_all != -1, bedge_all, fall_back)
    mem_graph_triplet = torch.stack(
        [updated_ball[: bedge.shape[1]], updated_ball[bedge.shape[1] :], bedge[1]],
        dim=0,
    )

    b_edge_index = edge_index[:, bmsk]
    new_eids = torch.arange(max_seen_eid + 1, max_seen_eid + 1 + bs, device=device)
    mem_eid = torch.cat([new_eids, new_eids, e_id[bmsk_cpu].to(device)], dim=0)
    del_addr = torch.cat([mem_graph_triplet[2], b_edge_index[1]], dim=0)
    relative_mem_id = mem_eid - (max_seen_eid + 1)

    mem_graph_quad_tmp = torch.vstack(
        [mem_graph_triplet[:2, relative_mem_id], del_addr, mem_eid]
    )
    direction = del_addr < bs
    src_part = mem_graph_quad_tmp[:, direction]
    dst_part = mem_graph_quad_tmp[:, ~direction].clone()
    dst_part[[0, 1]] = dst_part[[1, 0]]
    mem_graph_quad = torch.cat([src_part, dst_part], dim=1)

    remap_partial = model["memory"].mem_graph(
        n_id[: 3 * bs],
        torch.arange(3 * bs, device=device),
        src,
        pos_dst,
    )
    remap_fall_back = sampler.assoc[n_id[: 3 * bs]]
    remap = torch.where(remap_partial != -1, remap_partial, remap_fall_back)
    return mem_graph_quad, remap


def make_self_embedding_inputs(model, n_id, edge_index, src, pos_dst):
    ei_src_all = model["memory"].mem_graph(n_id[edge_index[0]], edge_index[1], src, pos_dst)
    updated_src = torch.where(ei_src_all != -1, ei_src_all, edge_index[0])
    return torch.stack([updated_src, edge_index[1]], dim=0)


def prepare_batch(
    runtime: Dict[str, object],
    batch,
    device: torch.device,
) -> Tuple[Dict[str, object], float, float, float, List[float]]:
    model = runtime["model"]
    data = runtime["data"]
    data_cache = runtime["data_cache"]
    cached_t = data_cache.get("t")
    cached_msg = data_cache.get("msg")
    cached_src = data_cache.get("src")
    sampler = runtime["sampler"]
    spec = runtime["spec"]
    min_dst_idx = runtime["min_dst_idx"]
    max_dst_idx = runtime["max_dst_idx"]
    known_dsts = runtime["known_dsts"]
    max_seen_eid = runtime["max_seen_eid"]

    transfer_time = 0.0
    preprocess_time = 0.0
    transfer_peaks: List[float] = []

    with record_function("stage_cpu_to_gpu_transfer"):
        reset_peak_memory(device)
        transfer_start = timed_region_start(device)
        batch = batch.to(device)
        transfer_time += timed_region_end(transfer_start, device)
        transfer_peaks.append(peak_memory_mb(device))

    src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
    batch_len = int(batch.num_events)

    with record_function("stage_sampling"):
        sampling_start = timed_region_start(device)
        neg_dst = train_neg_sampler(
            min_dst_idx,
            max_dst_idx,
            pos_dst,
            device,
            None,
            known_dsts=known_dsts,
        )
        root_ts = torch.cat([t, t, t], dim=0).double()
        root_nodes = torch.cat([src, pos_dst, neg_dst], dim=0)
        n_id, e_id, edge_index = sampler.sample(root_nodes, root_ts)
        if spec["decoder"] == "NCN":
            ncn_nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
            ncn_nid_ts[: root_nodes.shape[0]] = root_ts
            n_id, e_id, edge_index = sampler.sample(n_id, ncn_nid_ts)
        sampling_time = timed_region_end(sampling_start, device)

    nid_ts = None
    if spec["embedding"] == "time_emb":
        nid_ts = root_ts.new_full((n_id.shape[0],), root_ts.min())
        nid_ts[: root_nodes.shape[0]] = root_ts

    with record_function("stage_feature_lookup_preprocess"):
        preprocess_start = timed_region_start(device)
        if spec["deliver_to"] == "neighbor":
            bsrc = torch.stack([torch.arange(batch_len, device=device), pos_dst], dim=0)
            bdst = torch.stack([torch.arange(batch_len, 2 * batch_len, device=device), src], dim=0)
            bndst = torch.stack(
                [
                    torch.arange(2 * batch_len, 3 * batch_len, device=device),
                    torch.full((batch_len,), -1, dtype=torch.long, device=device),
                ],
                dim=0,
            )
            od = torch.cat([bsrc.T, bdst.T, bndst.T, sampler.id_to_pair], dim=0)
            od_updated = torch.cat([od, n_id.unsqueeze(1)], dim=1)
            tmp = od_updated[:, 0] % batch_len
            od_updated = torch.cat([od_updated, tmp.unsqueeze(1)], dim=1)
            mem_graph_quad, store_quad = vectorized_getMem_graph(od_updated, batch_len, max_seen_eid)
            unique_keys, inverse_indices = torch.unique(mem_graph_quad[[0, 1, 3]].T, dim=0, return_inverse=True)
            store_eid = store_quad[3].long()
            unique_eids = unique_keys[:, 2].long()
            store_eid_cpu = store_eid.cpu().long()
            unique_eids_cpu = unique_eids.cpu().long()
            if cached_src is not None:
                dirs = cached_src[store_eid] == n_id[store_quad[0]]
                dirs_cpu = None
            else:
                dirs = None
                dirs_cpu = data.src[store_eid_cpu] == n_id[store_quad[0]].cpu()
            preprocess_payload = {
                "mem_graph_quad": mem_graph_quad,
                "store_quad": store_quad,
                "unique_keys": unique_keys,
                "inverse_indices": inverse_indices,
                "store_eid": store_eid,
                "unique_eids": unique_eids,
                "store_eid_cpu": store_eid_cpu,
                "unique_eids_cpu": unique_eids_cpu,
                "dirs": dirs,
                "dirs_cpu": dirs_cpu,
            }
        else:
            mem_graph_quad, remap = make_self_mem_graph(
                model,
                sampler,
                n_id,
                e_id,
                edge_index,
                batch_len,
                max_seen_eid,
                src,
                pos_dst,
                device,
            )
            b_eid = mem_graph_quad[3].long()
            b_eid_cpu = b_eid.cpu().long()
            if cached_src is not None:
                b_isrc = n_id[mem_graph_quad[1]] == cached_src[b_eid]
            else:
                b_isrc = (n_id[mem_graph_quad[1]].cpu() == data.src[b_eid_cpu]).to(device)
            preprocess_payload = {
                "mem_graph_quad": mem_graph_quad,
                "mem_graph_quad_uf": mem_graph_quad,
                "remap": remap,
                "b_eid": b_eid,
                "b_eid_cpu": b_eid_cpu,
                "b_isrc": b_isrc,
            }
        preprocess_time = timed_region_end(preprocess_start, device)

    with record_function("stage_cpu_to_gpu_transfer"):
        reset_peak_memory(device)
        transfer_start = timed_region_start(device)
        if spec["deliver_to"] == "neighbor":
            if cached_t is not None and cached_msg is not None:
                unique_eids = preprocess_payload["unique_eids"]
                preprocess_payload["b_t_unique"] = cached_t[unique_eids]
                preprocess_payload["b_raw_msg_unique"] = cached_msg[unique_eids]
            else:
                unique_eids_cpu = preprocess_payload["unique_eids_cpu"]
                preprocess_payload["b_t_unique"] = data.t[unique_eids_cpu].to(device)
                preprocess_payload["b_raw_msg_unique"] = data.msg[unique_eids_cpu].to(device)
            if preprocess_payload["dirs"] is None:
                preprocess_payload["dirs"] = preprocess_payload["dirs_cpu"].to(device)
        else:
            if cached_t is not None and cached_msg is not None:
                b_eid = preprocess_payload["b_eid"]
                preprocess_payload["b_t"] = cached_t[b_eid]
                preprocess_payload["b_raw_msg"] = cached_msg[b_eid]
            else:
                b_eid_cpu = preprocess_payload["b_eid_cpu"]
                preprocess_payload["b_t"] = data.t[b_eid_cpu].to(device)
                preprocess_payload["b_raw_msg"] = data.msg[b_eid_cpu].to(device)
        if spec["embedding"] != "time_emb":
            e_id_idx = e_id.long()
            if cached_t is not None and cached_msg is not None:
                preprocess_payload["edge_t"] = cached_t[e_id_idx]
                preprocess_payload["edge_msg"] = cached_msg[e_id_idx]
            else:
                preprocess_payload["edge_t"] = data.t[e_id_idx].to(device)
                preprocess_payload["edge_msg"] = data.msg[e_id_idx].to(device)
        transfer_time += timed_region_end(transfer_start, device)
        transfer_peaks.append(peak_memory_mb(device))

    prepared = {
        "batch": batch,
        "src": src,
        "pos_dst": pos_dst,
        "neg_dst": neg_dst,
        "t": t,
        "msg": msg,
        "batch_len": batch_len,
        "root_ts": root_ts,
        "root_nodes": root_nodes,
        "n_id": n_id,
        "e_id": e_id,
        "edge_index": edge_index,
        "nid_ts": nid_ts,
    }
    prepared.update(preprocess_payload)
    return prepared, sampling_time, preprocess_time, transfer_time, transfer_peaks


def run_forward_components(
    runtime: Dict[str, object],
    prepared: Dict[str, object],
    criterion,
    device: torch.device,
):
    model = runtime["model"]
    data = runtime["data"]
    spec = runtime["spec"]
    component_peaks: List[float] = []

    with record_function("stage_memory_module"):
        before_alloc = memory_allocated_mb(device)
        reset_peak_memory(device)
        start = timed_region_start(device)
        if spec["deliver_to"] == "neighbor":
            z, last_update = model["memory"](
                prepared["n_id"],
                prepared["mem_graph_quad"],
                prepared["b_t_unique"],
                prepared["b_raw_msg_unique"],
                prepared["unique_keys"],
                prepared["inverse_indices"],
                data=data,
            )
        else:
            z, last_update = model["memory"](
                prepared["n_id"],
                prepared["mem_graph_quad"][0:2, :],
                prepared["b_t"],
                prepared["b_raw_msg"],
                prepared["b_isrc"],
                delivery_addr=prepared["mem_graph_quad"][2],
            )
            z = torch.cat([z[prepared["remap"]], z[3 * prepared["batch_len"] :]], dim=0)
            last_update = torch.cat(
                [last_update[prepared["remap"]], last_update[3 * prepared["batch_len"] :]],
                dim=0,
            )
        memory_module_time = timed_region_end(start, device)
        after_alloc = memory_allocated_mb(device)
        memory_module_peak = peak_memory_mb(device)
        memory_module_delta = after_alloc - before_alloc
    component_peaks.append(memory_module_peak)

    with record_function("stage_emb_module"):
        before_alloc = memory_allocated_mb(device)
        reset_peak_memory(device)
        start = timed_region_start(device)
        if spec["embedding"] == "time_emb":
            embeddings = model["gnn"](z, last_update, prepared["nid_ts"])
        elif spec["deliver_to"] == "neighbor":
            embeddings = model["gnn"](
                z,
                last_update,
                prepared["edge_index"],
                prepared["edge_t"],
                prepared["edge_msg"],
            )
        else:
            edge_index_for_gnn = make_self_embedding_inputs(
                model,
                prepared["n_id"],
                prepared["edge_index"],
                prepared["src"],
                prepared["pos_dst"],
            )
            embeddings = model["gnn"](
                z,
                last_update,
                edge_index_for_gnn,
                prepared["edge_t"],
                prepared["edge_msg"],
            )
            prepared["edge_index_for_gnn"] = edge_index_for_gnn
        emb_module_time = timed_region_end(start, device)
        after_alloc = memory_allocated_mb(device)
        emb_module_peak = peak_memory_mb(device)
        emb_module_delta = after_alloc - before_alloc
    component_peaks.append(emb_module_peak)

    with record_function("stage_decoder_module"):
        before_alloc = memory_allocated_mb(device)
        reset_peak_memory(device)
        start = timed_region_start(device)
        if spec["decoder"] == "NCN":
            src_re = torch.arange(prepared["batch_len"], device=device)
            pos_re = torch.arange(prepared["batch_len"], 2 * prepared["batch_len"], device=device)
            neg_re = torch.arange(2 * prepared["batch_len"], 3 * prepared["batch_len"], device=device)
            time_info = (last_update, prepared["t"])
            decoder_edge_index = prepared.get("edge_index_for_gnn", prepared["edge_index"])
            pos_out = model["link_pred"](
                embeddings,
                decoder_edge_index,
                torch.stack([src_re, pos_re], dim=0),
                2,
                cn_time_decay=False,
                time_info=time_info,
            )
            neg_out = model["link_pred"](
                embeddings,
                decoder_edge_index,
                torch.stack([src_re, neg_re], dim=0),
                2,
                cn_time_decay=False,
                time_info=time_info,
            )
        else:
            pos_out = model["link_pred"](embeddings[0 : prepared["batch_len"]], embeddings[prepared["batch_len"] : 2 * prepared["batch_len"]])
            neg_out = model["link_pred"](embeddings[0 : prepared["batch_len"]], embeddings[2 * prepared["batch_len"] : 3 * prepared["batch_len"]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss = loss + criterion(neg_out, torch.zeros_like(neg_out))
        decoder_time = timed_region_end(start, device)
        after_alloc = memory_allocated_mb(device)
        decoder_peak = peak_memory_mb(device)
        decoder_delta = after_alloc - before_alloc
    component_peaks.append(decoder_peak)

    return {
        "loss": loss,
        "memory_module_time": memory_module_time,
        "emb_module_time": emb_module_time,
        "decoder_time": decoder_time,
        "memory_module_peak": memory_module_peak,
        "emb_module_peak": emb_module_peak,
        "decoder_peak": decoder_peak,
        "memory_module_delta": memory_module_delta,
        "emb_module_delta": emb_module_delta,
        "decoder_delta": decoder_delta,
        "forward_peak": max(component_peaks) if component_peaks else 0.0,
    }


def execute_train_step(
    model_name: str,
    runtime: Dict[str, object],
    batch,
    batch_size: int,
    device: torch.device,
    sample_gpu_utilization: bool,
) -> Tuple[IterationRecord, int]:
    model = runtime["model"]
    optimizer = runtime["optimizer"]
    criterion = runtime["criterion"]
    spec = runtime["spec"]
    max_seen_eid = runtime["max_seen_eid"]

    optimizer.zero_grad(set_to_none=True)

    total_start = timed_region_start(device)
    overall_peak_candidates: List[float] = []

    prepared, sampling_time, preprocess_time, transfer_time, transfer_peaks = prepare_batch(
        runtime,
        batch,
        device,
    )
    overall_peak_candidates.extend(transfer_peaks)

    with record_function("stage_forward"):
        forward_start = timed_region_start(device)
        forward_parts = run_forward_components(
            runtime=runtime,
            prepared=prepared,
            criterion=criterion,
            device=device,
        )
        loss = forward_parts["loss"]
        forward_time = timed_region_end(forward_start, device)
        overall_peak_candidates.append(forward_parts["forward_peak"])

    with record_function("stage_backward"):
        reset_peak_memory(device)
        backward_start = timed_region_start(device)
        loss.backward()
        backward_time = timed_region_end(backward_start, device)
        overall_peak_candidates.append(peak_memory_mb(device))

    with record_function("stage_optimizer_step"):
        reset_peak_memory(device)
        optimizer_step_start = timed_region_start(device)
        optimizer.step()
        optimizer_step_time = timed_region_end(optimizer_step_start, device)
        overall_peak_candidates.append(peak_memory_mb(device))

    with record_function("stage_memory_update"):
        reset_peak_memory(device)
        memory_update_start = timed_region_start(device)
        if spec["deliver_to"] == "neighbor":
            z_update, last_update_update = model["memory"](
                prepared["n_id"],
                prepared["mem_graph_quad"],
                prepared["b_t_unique"],
                prepared["b_raw_msg_unique"],
                prepared["unique_keys"],
                prepared["inverse_indices"],
                data=runtime["data"],
            )
            model["memory"].update_state(
                prepared["n_id"],
                z_update,
                last_update_update,
                prepared["store_quad"],
                prepared["dirs"],
            )
        else:
            z_update, last_update_update = model["memory"](
                prepared["n_id"],
                prepared["mem_graph_quad"][0:2, :],
                prepared["b_t"],
                prepared["b_raw_msg"],
                prepared["b_isrc"],
                delivery_addr=prepared["mem_graph_quad"][2],
            )
            z_update = torch.cat([z_update[prepared["remap"]], z_update[3 * prepared["batch_len"] :]], dim=0)
            last_update_update = torch.cat(
                [last_update_update[prepared["remap"]], last_update_update[3 * prepared["batch_len"] :]],
                dim=0,
            )
            model["memory"].update_state_v2(
                prepared["mem_graph_quad_uf"][2],
                prepared["remap"],
                prepared["batch_len"],
                prepared["src"],
                prepared["pos_dst"],
                prepared["t"],
                prepared["msg"],
                prepared["n_id"],
                last_update_update,
                z_update,
            )
        model["memory"].detach()
        memory_update_time = timed_region_end(memory_update_start, device)
        overall_peak_candidates.append(peak_memory_mb(device))

    with record_function("stage_neighbor_state_update"):
        sampler_state_update_time = 0.0

    total_time = timed_region_end(total_start, device)
    peak_gpu_memory = max(overall_peak_candidates) if overall_peak_candidates else 0.0
    gpu_util = query_gpu_utilization(device) if sample_gpu_utilization else float("nan")
    throughput = prepared["batch_len"] / total_time if total_time > 0 else float("nan")

    record = IterationRecord(
        run_id="",
        system=SYSTEM_NAME,
        model=model_name,
        dataset="",
        seed=-1,
        batch_size=batch_size,
        num_neighbors=-1,
        num_layers=-1,
        epoch=-1,
        epoch_iteration=-1,
        global_iteration=-1,
        measured_iteration=-1,
        total_iteration_time_s=total_time,
        sampling_time_s=sampling_time,
        feature_lookup_preprocess_time_s=preprocess_time,
        cpu_to_gpu_transfer_time_s=transfer_time,
        forward_time_s=forward_time,
        memory_module_time_s=forward_parts["memory_module_time"],
        emb_module_time_s=forward_parts["emb_module_time"],
        decoder_time_s=forward_parts["decoder_time"],
        backward_time_s=backward_time,
        memory_update_time_s=memory_update_time,
        sampler_state_update_time_s=sampler_state_update_time,
        optimizer_step_time_s=optimizer_step_time,
        peak_gpu_memory_mb=peak_gpu_memory,
        memory_module_peak_gpu_memory_mb=forward_parts["memory_module_peak"],
        emb_module_peak_gpu_memory_mb=forward_parts["emb_module_peak"],
        decoder_peak_gpu_memory_mb=forward_parts["decoder_peak"],
        memory_module_memory_delta_mb=forward_parts["memory_module_delta"],
        emb_module_memory_delta_mb=forward_parts["emb_module_delta"],
        decoder_memory_delta_mb=forward_parts["decoder_delta"],
        gpu_utilization_pct=gpu_util,
        throughput_edges_per_s=throughput,
        samples_in_batch=prepared["batch_len"],
    )
    next_max_seen_eid = max_seen_eid + prepared["batch_len"]
    return record, next_max_seen_eid


def summarize_run(
    run_id: str,
    model_name: str,
    dataset_name: str,
    seed: int,
    batch_size: int,
    num_neighbors: int,
    num_layers: int,
    warmup_iters: int,
    measured_iters: int,
    iteration_records: Sequence[IterationRecord],
    total_epoch_time_s: float,
    total_train_time_s: float,
) -> RunSummary:
    total_iter = np.array([rec.total_iteration_time_s for rec in iteration_records], dtype=float)
    sampling = np.array([rec.sampling_time_s for rec in iteration_records], dtype=float)
    preprocess = np.array([rec.feature_lookup_preprocess_time_s for rec in iteration_records], dtype=float)
    transfer = np.array([rec.cpu_to_gpu_transfer_time_s for rec in iteration_records], dtype=float)
    forward = np.array([rec.forward_time_s for rec in iteration_records], dtype=float)
    memory_forward = np.array([rec.memory_module_time_s for rec in iteration_records], dtype=float)
    emb_forward = np.array([rec.emb_module_time_s for rec in iteration_records], dtype=float)
    decoder_forward = np.array([rec.decoder_time_s for rec in iteration_records], dtype=float)
    backward = np.array([rec.backward_time_s for rec in iteration_records], dtype=float)
    memory = np.array([rec.memory_update_time_s for rec in iteration_records], dtype=float)
    sampler_state = np.array([rec.sampler_state_update_time_s for rec in iteration_records], dtype=float)
    optimizer = np.array([rec.optimizer_step_time_s for rec in iteration_records], dtype=float)
    gpu_mem = np.array([rec.peak_gpu_memory_mb for rec in iteration_records], dtype=float)
    memory_module_peak = np.array([rec.memory_module_peak_gpu_memory_mb for rec in iteration_records], dtype=float)
    emb_module_peak = np.array([rec.emb_module_peak_gpu_memory_mb for rec in iteration_records], dtype=float)
    decoder_peak = np.array([rec.decoder_peak_gpu_memory_mb for rec in iteration_records], dtype=float)
    memory_module_delta = np.array([rec.memory_module_memory_delta_mb for rec in iteration_records], dtype=float)
    emb_module_delta = np.array([rec.emb_module_memory_delta_mb for rec in iteration_records], dtype=float)
    decoder_delta = np.array([rec.decoder_memory_delta_mb for rec in iteration_records], dtype=float)
    gpu_util = np.array([rec.gpu_utilization_pct for rec in iteration_records], dtype=float)
    throughput = np.array([rec.throughput_edges_per_s for rec in iteration_records], dtype=float)

    return RunSummary(
        run_id=run_id,
        system=SYSTEM_NAME,
        model=model_name,
        dataset=dataset_name,
        seed=seed,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        num_layers=num_layers,
        warmup_iterations=warmup_iters,
        measured_iterations=measured_iters,
        avg_iteration_time_s=float(np.mean(total_iter)),
        std_iteration_time_s=float(np.std(total_iter)),
        avg_sampling_time_s=float(np.mean(sampling)),
        avg_feature_lookup_preprocess_time_s=float(np.mean(preprocess)),
        avg_cpu_to_gpu_transfer_time_s=float(np.mean(transfer)),
        avg_forward_time_s=float(np.mean(forward)),
        avg_memory_module_time_s=float(np.mean(memory_forward)),
        avg_emb_module_time_s=float(np.mean(emb_forward)),
        avg_decoder_time_s=float(np.mean(decoder_forward)),
        avg_backward_time_s=float(np.mean(backward)),
        avg_memory_update_time_s=float(np.mean(memory)),
        avg_sampler_state_update_time_s=float(np.mean(sampler_state)),
        avg_optimizer_step_time_s=float(np.mean(optimizer)),
        avg_peak_gpu_memory_mb=float(np.mean(gpu_mem)),
        peak_gpu_memory_mb=float(np.max(gpu_mem)),
        avg_memory_module_peak_gpu_memory_mb=float(np.mean(memory_module_peak)),
        avg_emb_module_peak_gpu_memory_mb=float(np.mean(emb_module_peak)),
        avg_decoder_peak_gpu_memory_mb=float(np.mean(decoder_peak)),
        avg_memory_module_memory_delta_mb=float(np.mean(memory_module_delta)),
        avg_emb_module_memory_delta_mb=float(np.mean(emb_module_delta)),
        avg_decoder_memory_delta_mb=float(np.mean(decoder_delta)),
        avg_gpu_utilization_pct=float(np.nanmean(gpu_util)) if not np.isnan(gpu_util).all() else float("nan"),
        avg_throughput_edges_per_s=float(np.mean(throughput)),
        total_epoch_time_s=total_epoch_time_s,
        total_train_time_s=total_train_time_s,
        final_validation_metric=float("nan"),
        validation_metric_name="mrr",
    )


def main():
    args = parse_args()
    ensure_prism_runtime_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.chdir(REPO_ROOT)

    device = get_device(args)
    dataset_internal, dataset_output = resolve_dataset_names(args.dataset)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"{dataset_output}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.json", vars(args))
    write_json(output_root / "system_info.json", get_system_info(device))

    for model_name in args.models:
        for batch_size in args.batch_sizes:
            for run_offset in range(args.runs):
                seed = args.seed + run_offset
                set_seed(seed)
                runtime = load_runtime(dataset_internal, model_name, batch_size, args, device)
                num_neighbors = get_num_neighbors(args)
                num_layers = get_num_layers(args)
                reset_runtime_state(runtime)

                run_id = f"{model_name}_bs{batch_size}_nbr{num_neighbors}_layers{num_layers}_seed{seed}"
                print(f"[benchmark] {SYSTEM_NAME} {run_id}")

                measured_records: List[IterationRecord] = []
                epoch_times: List[float] = []
                global_iteration = 0
                total_train_start = time.perf_counter()

                trace_dir = output_root / "profiler_traces" / run_id
                with make_profiler(
                    trace_dir=trace_dir,
                    warmup_iters=args.warmup_iters,
                    measured_iters=args.measure_iters,
                    enable_cuda=device.type == "cuda",
                ) as prof:
                    epoch = 0
                    while len(measured_records) < args.measure_iters:
                        epoch += 1
                        epoch_start = time.perf_counter()
                        reset_runtime_state(runtime)
                        model = runtime["model"]
                        model["memory"].train()
                        model["gnn"].train()
                        model["link_pred"].train()

                        for epoch_iteration, batch in enumerate(runtime["dataset"]["train_dataloader"], start=1):
                            global_iteration += 1
                            record, next_max_seen_eid = execute_train_step(
                                model_name=model_name,
                                runtime=runtime,
                                batch=batch,
                                batch_size=batch_size,
                                device=device,
                                sample_gpu_utilization=args.sample_gpu_utilization,
                            )
                            runtime["max_seen_eid"] = next_max_seen_eid
                            prof.step()

                            if global_iteration > args.warmup_iters:
                                measured_idx = len(measured_records) + 1
                                record.run_id = run_id
                                record.dataset = dataset_output
                                record.seed = seed
                                record.num_neighbors = num_neighbors
                                record.num_layers = num_layers
                                record.epoch = epoch
                                record.epoch_iteration = epoch_iteration
                                record.global_iteration = global_iteration
                                record.measured_iteration = measured_idx
                                measured_records.append(record)

                            if len(measured_records) >= args.measure_iters:
                                break

                        epoch_times.append(time.perf_counter() - epoch_start)

                total_train_time_s = time.perf_counter() - total_train_start
                profiler_table = prof.key_averages().table(
                    sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
                )
                (output_root / "profiler_tables").mkdir(parents=True, exist_ok=True)
                (output_root / "profiler_tables" / f"{run_id}.txt").write_text(profiler_table)

                if not args.skip_validation:
                    print("[warning] Validation is not implemented in benchmark_prism.py yet; writing NaN metrics.")

                summary = summarize_run(
                    run_id=run_id,
                    model_name=model_name,
                    dataset_name=dataset_output,
                    seed=seed,
                    batch_size=batch_size,
                    num_neighbors=num_neighbors,
                    num_layers=num_layers,
                    warmup_iters=args.warmup_iters,
                    measured_iters=args.measure_iters,
                    iteration_records=measured_records,
                    total_epoch_time_s=float(np.mean(epoch_times)) if epoch_times else float("nan"),
                    total_train_time_s=total_train_time_s,
                )

                append_csv(
                    output_root / "iteration_metrics.csv",
                    [asdict(record) for record in measured_records],
                )
                append_csv(output_root / "run_summary.csv", [asdict(summary)])

                print(
                    f"[done] {SYSTEM_NAME} {run_id} avg_iter={summary.avg_iteration_time_s:.6f}s "
                    f"throughput={summary.avg_throughput_edges_per_s:.2f} edges/s "
                    f"peak_mem={summary.peak_gpu_memory_mb:.2f} MB"
                )

    print(f"Results written to: {output_root}")


if __name__ == "__main__":
    main()
