import numpy as np
import torch
import timeit
import os
import os.path as osp
from pathlib import Path
import preprocessor #openmp
from modules.recent_sampler import Recent_K_Sampler


def main():
    src = torch.tensor([1, 2, 3, 1, 2, 3, 4, 0, 1, 2, 0], dtype=torch.long)
    dst = torch.tensor([0, 0, 1, 2, 3, 4, 1, 2, 3, 2, 2], dtype=torch.long)
    ts = torch.tensor([20.1, 50.9, 60.11, 67.45, 89.28, 98.32, 112.89, 123.67, 145.52, 187.12, 190.5], dtype=torch.float64)
    eid = torch.arange(len(src), dtype=torch.long)

    num_nodes = 5
    chunk_size = 4
    max_chunk_per_node = 4
    k = 3

    src_list = src.tolist()
    dst_list = dst.tolist()
    ts_list = ts.tolist()
    eid_list = eid.tolist()


    output = preprocessor.preprocess(
        src_list,
        dst_list,
        ts_list,
        eid_list,
        num_nodes,
        chunk_size,
        max_chunk_per_node
    )

    sampler = Recent_K_Sampler(output, max_chunk_per_node, k)
    root_node = torch.tensor([0, 1, 2, 3, 4, 1, 1, 0], dtype=torch.long, device='cuda')
    root_ts = torch.tensor([200.0, 115.0, 130.0, 140.0, 10.0, 56.0, 150.0, 10.0], dtype=torch.float64, device='cuda')

    samples = sampler.sample(root_node, root_ts)
    print(samples)


    

if __name__ == "__main__":
    main()
