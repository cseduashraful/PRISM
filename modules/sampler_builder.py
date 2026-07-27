from dataclasses import dataclass
from uuid import uuid4
from typing import Any

import torch
import chunkio
import preprocessor

from modules.recent_sampler import Recent_K_Sampler


@dataclass
class SamplerBuildSpec:
    k: int
    chunk_size: int
    max_chunk_per_node: int
    offload_mode: str = "auto"
    cache_size: int = 100
    device: Any = "cuda"
    apan: bool = False
    skip_cnt: int = 1


@dataclass
class SamplerBuildResult:
    sampler: Any
    backend: str
    preproc_dir: str | None = None


def build_sampler_backend(data, spec: SamplerBuildSpec) -> SamplerBuildResult:
    if spec.offload_mode == "on":
        return _build_disk_sampler(data, spec)
    if spec.offload_mode == "off":
        return _build_in_memory_sampler(data, spec)
    if spec.offload_mode != "auto":
        raise ValueError(f"Unsupported offload mode: {spec.offload_mode}")

    try:
        return _build_in_memory_sampler(data, spec)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _build_disk_sampler(data, spec)


def _build_in_memory_sampler(data, spec: SamplerBuildSpec) -> SamplerBuildResult:
    tci_data = preprocessor.preprocess(
        data.src.tolist(),
        data.dst.tolist(),
        data.t.double().tolist(),
        torch.arange(data.src.shape[0]).tolist(),
        data.num_nodes,
        spec.chunk_size,
        spec.max_chunk_per_node,
    )
    sampler = Recent_K_Sampler(
        tci_data,
        spec.max_chunk_per_node,
        spec.k,
        data.num_nodes,
        device=spec.device,
        apan=spec.apan,
        skip_cnt=spec.skip_cnt,
    )
    return SamplerBuildResult(sampler=sampler, backend="in-memory")


def _build_disk_sampler(data, spec: SamplerBuildSpec) -> SamplerBuildResult:
    from modules.grnstream import GRN_Stream

    folder_name = f"cache_{uuid4().hex[:8]}"
    outdir = "preproc_out/" + folder_name
    tci_stream = chunkio.preprocess_streaming(
        data.src.tolist(),
        data.dst.tolist(),
        data.t.double().tolist(),
        torch.arange(data.src.shape[0]).tolist(),
        data.num_nodes,
        chunk_size=spec.chunk_size,
        max_chunk_per_node=spec.max_chunk_per_node,
        duplicate_undirected=True,
        out_dir=outdir,
        num_shards=256,
    )
    sampler = GRN_Stream(
        tci_stream,
        spec.max_chunk_per_node,
        spec.k,
        data.num_nodes,
        spec.chunk_size,
        outdir,
        cache_size=spec.cache_size,
        device=spec.device,
        apan=spec.apan,
        skip_cnt=spec.skip_cnt,
    )
    return SamplerBuildResult(
        sampler=sampler,
        backend="disk-offload",
        preproc_dir=outdir,
    )
