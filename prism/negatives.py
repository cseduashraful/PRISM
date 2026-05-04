import argparse
import os
import time

from modules.neg_gen import NegativeEdgeGenerator
from modules.dataset_pyg import PyGLinkPropPredDataset


def generate_offline_negative_samples(
    dataset_name: str,
    root: str = "datasets",
    num_neg_per_pos: int = 100,
    strategy: str = "rnd",
    seed: int = 42,
    partial_path: str = "./",
    if_exists: str = "skip",
):
    if if_exists not in {"skip", "force", "error"}:
        raise ValueError("if_exists must be one of: 'skip', 'force', 'error'")

    dataset = PyGLinkPropPredDataset(name=dataset_name, root=root)
    data = dataset.get_TemporalData()
    train_data = data[dataset.train_mask]
    val_data = data[dataset.val_mask]
    test_data = data[dataset.test_mask]

    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    historical_data = train_data if strategy == "hist_rnd" else None

    neg_sampler = NegativeEdgeGenerator(
        dataset_name=dataset_name,
        first_dst_id=min_dst_idx,
        last_dst_id=max_dst_idx,
        num_neg_e=num_neg_per_pos,
        strategy=strategy,
        rnd_seed=seed,
        historical_data=historical_data,
    )

    out_files = {
        "val": f"{partial_path}/{dataset_name}_val_ns.pkl",
        "test": f"{partial_path}/{dataset_name}_test_ns.pkl",
    }

    if if_exists == "error":
        conflicts = [p for p in out_files.values() if os.path.exists(p)]
        if conflicts:
            raise FileExistsError(
                "Negative sample files already exist: " + ", ".join(conflicts)
            )
    elif if_exists == "force":
        for p in out_files.values():
            if os.path.exists(p):
                os.remove(p)
                print(f"INFO: Removed existing negatives file: {p}")

    for split_mode, split_data in [("val", val_data), ("test", test_data)]:
        if if_exists == "skip" and os.path.exists(out_files[split_mode]):
            print(f"INFO: Skipping {split_mode}; existing file found: {out_files[split_mode]}")
            continue
        start = time.time()
        neg_sampler.generate_negative_samples(
            data=split_data, split_mode=split_mode, partial_path=partial_path
        )
        print(
            f"INFO: Generated {split_mode} negatives with strategy={strategy} "
            f"in {time.time() - start:.2f}s"
        )


def main():
    parser = argparse.ArgumentParser("prism-negatives")
    parser.add_argument("--data", required=True, help="Dataset name, e.g., tgbl-wiki")
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--num-neg", type=int, default=100)
    parser.add_argument("--strategy", choices=["rnd", "hist_rnd"], default="rnd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partial-path", default="./")
    parser.add_argument("--if-exists", choices=["skip", "force", "error"], default="skip")
    args = parser.parse_args()
    generate_offline_negative_samples(
        dataset_name=args.data,
        root=args.root,
        num_neg_per_pos=args.num_neg,
        strategy=args.strategy,
        seed=args.seed,
        partial_path=args.partial_path,
        if_exists=args.if_exists,
    )


if __name__ == "__main__":
    main()
