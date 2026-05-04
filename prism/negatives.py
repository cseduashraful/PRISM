import argparse
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
):
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

    for split_mode, split_data in [("val", val_data), ("test", test_data)]:
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
    args = parser.parse_args()
    generate_offline_negative_samples(
        dataset_name=args.data,
        root=args.root,
        num_neg_per_pos=args.num_neg,
        strategy=args.strategy,
        seed=args.seed,
        partial_path=args.partial_path,
    )


if __name__ == "__main__":
    main()
