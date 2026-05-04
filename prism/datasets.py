from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader


def dataset_from_temporal_data(
    data: TemporalData,
    batch_size: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    n = data.num_events
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    observed_nodes = torch.cat([train_data.src, train_data.dst]).unique()
    transductive_mask = {
        "val": torch.isin(val_data.src, observed_nodes).cpu().numpy(),
        "test": torch.isin(test_data.src, observed_nodes).cpu().numpy(),
    }
    inductive_mask = {
        "val": ~transductive_mask["val"],
        "test": ~transductive_mask["test"],
    }

    return {
        "data": data,
        "train_dataloader": TemporalDataLoader(train_data, batch_size=batch_size),
        "val_dataloader": TemporalDataLoader(val_data, batch_size=batch_size),
        "test_dataloader": TemporalDataLoader(test_data, batch_size=batch_size),
        "neg_sampler": None,
        "evaluator": None,
        "metric": "mrr",
        "train_length": train_data.num_events,
        "transductive_mask": transductive_mask,
        "inductive_mask": inductive_mask,
    }


def dataset_from_csv(
    csv_path: str,
    batch_size: int,
    src_col: str = "src",
    dst_col: str = "dst",
    t_col: str = "t",
    msg_cols: list[str] | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    df = pd.read_csv(csv_path)
    if msg_cols is None:
        msg_cols = [c for c in df.columns if c.startswith("msg")]

    if msg_cols:
        msg = torch.tensor(df[msg_cols].values, dtype=torch.float32)
    else:
        msg = torch.zeros((len(df), 1), dtype=torch.float32)

    src = torch.tensor(df[src_col].values, dtype=torch.long)
    dst = torch.tensor(df[dst_col].values, dtype=torch.long)
    t = torch.tensor(df[t_col].values, dtype=torch.long)

    # keep chronological order
    order = torch.argsort(t)
    data = TemporalData(
        src=src[order],
        dst=dst[order],
        t=t[order],
        msg=msg[order],
    )
    data.num_nodes = int(torch.max(torch.cat([data.src, data.dst])).item()) + 1
    return dataset_from_temporal_data(data, batch_size=batch_size, val_ratio=val_ratio, test_ratio=test_ratio)


def dataset_from_directory(
    dataset_dir: str,
    batch_size: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Load a temporal dataset from a directory like:
      - edges.csv (required): columns include src, dst, time
      - edge_features.pt (optional): [num_edges, feat_dim]
    This matches the REDDIT layout under GNNFlow data directories.
    """
    edges_path = f"{dataset_dir}/edges.csv"
    df = pd.read_csv(edges_path)

    # Drop typical unnamed index column if present.
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Support common aliases.
    src_col = "src" if "src" in df.columns else "u"
    dst_col = "dst" if "dst" in df.columns else "i"
    t_col = "time" if "time" in df.columns else ("ts" if "ts" in df.columns else "t")

    src = torch.tensor(df[src_col].values, dtype=torch.float32).long()
    dst = torch.tensor(df[dst_col].values, dtype=torch.float32).long()
    t = torch.tensor(df[t_col].values, dtype=torch.float64)

    feat_path = f"{dataset_dir}/edge_features.pt"
    try:
        msg = torch.load(feat_path, map_location="cpu")
        if not isinstance(msg, torch.Tensor):
            msg = torch.tensor(msg, dtype=torch.float32)
        msg = msg.float()
        if msg.ndim == 1:
            msg = msg.unsqueeze(1)
    except Exception:
        msg = torch.zeros((len(df), 1), dtype=torch.float32)

    if msg.shape[0] != len(df):
        raise ValueError(
            f"edge_features.pt row count {msg.shape[0]} does not match edges.csv rows {len(df)}"
        )

    order = torch.argsort(t)
    data = TemporalData(
        src=src[order],
        dst=dst[order],
        t=t[order].long(),
        msg=msg[order],
    )
    data.num_nodes = int(torch.max(torch.cat([data.src, data.dst])).item()) + 1
    return dataset_from_temporal_data(data, batch_size=batch_size, val_ratio=val_ratio, test_ratio=test_ratio)
