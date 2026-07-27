from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class DatasetRuntime:
    dataset: Dict[str, Any]
    data: Any
    data_cache: Dict[str, Optional[torch.Tensor]]
    min_dst_idx: int
    max_dst_idx: int
    known_dsts: Optional[torch.Tensor]


@dataclass
class SamplerRuntime:
    sampler: Any
    backend: Optional[str] = None


@dataclass
class TrainRuntime:
    model_bundle: Any
    dataset_runtime: DatasetRuntime
    sampler_runtime: SamplerRuntime
    device: torch.device
    neg_sampler: Any
    deliver_to: str
    decoder: str
    embedding: str
    val_neg: int
