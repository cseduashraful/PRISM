
from uuid import uuid4
import torch
import os
from modules.data_utils import save_tci_data, load_tci_data
import pandas as pd
import chunkio

chunk_size = 2
max_chunk_per_node = 3
folder_name = f"cache_test"
outdir = "inference_preproc_out/"+folder_name






df = pd.read_csv("../datasets/toy/toy_edgelist_chunk.csv")
src  = df["src"].astype(int).tolist()
dst  = df["dst"].astype(int).tolist()
time = df["time"].astype(float).tolist()
num_nodes = 5

print(src)
print(dst)
print(time)
trvl=6


tci_data = chunkio.preprocess_streaming(
    src[:trvl],
    dst[:trvl],#dst_list,
    time[:trvl],#ts_list,
    torch.arange(len(src[:trvl])).tolist(),#eid_list,
    num_nodes,
    chunk_size=chunk_size,
    max_chunk_per_node=max_chunk_per_node,
    duplicate_undirected=True,
    out_dir=outdir,#"preproc_out",
    num_shards=256,
)
pkl_path = "inference_preproc_out/"+folder_name+"_tci.pkl"

save_tci_data(tci_data, pkl_path)

tci = load_tci_data(pkl_path)
cache_size  = 100
tci_reader = chunkio.TorchChunkCache(tci.out_dir, tci.chunk_size, cache_size)


ts_chunks_selected_cpu, eid_chunks_selected_cpu, other_node_chunks_selected_cpu = tci_reader.get_chunks_torch(torch.tensor([0, 1, 2, 3, 4, 5, 6]))

print(tci.chunk_map)
print(tci.chunk_last_ts)
print(ts_chunks_selected_cpu)
print(eid_chunks_selected_cpu)
print(other_node_chunks_selected_cpu)

breakpoint()

tci_ex = chunkio.extend_streaming_latestk_ordered_reuse(
    tci,
    src[trvl:],
    dst[trvl:],
    time[trvl:],
    torch.arange(trvl, len(src)).tolist(),
    duplicate_undirected=True,
)
tci_reader = chunkio.TorchChunkCache(tci_ex.out_dir, tci_ex.chunk_size, cache_size)




breakpoint()

ts_chunks_selected_cpu, eid_chunks_selected_cpu, other_node_chunks_selected_cpu = tci_reader.get_chunks_torch(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

print(tci_ex.chunk_map)
print(tci_ex.chunk_last_ts)
print(ts_chunks_selected_cpu)
print(eid_chunks_selected_cpu)
print(other_node_chunks_selected_cpu)

breakpoint()