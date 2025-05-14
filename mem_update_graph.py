import torch

max_node = 10
batch = torch.tensor([[1, 2, 3, 2],[2, 3, 1, 4]])
batch_t = torch.tensor([1, 2, 3, 4], dtype = torch.float64)
assoc = torch.arange(max_node)


def memory_update_graph(buff):
    uniques = torch.unique(batch)
    assoc[uniques] = torch.arange(uniques.shape[0])
    node_ids = torch.cat([uniques, batch[0, :], batch[1,:]])

    breakpoint()

if __name__ == "__main__":
    memory_update_graph(batch)

# import torch

# edges = torch.tensor([
#     [1, 2, 1],
#     [2, 3, 2],
#     [3, 1, 3],
#     [2, 4, 4]
# ], dtype=torch.long)

# src_nodes = edges[:, 0]
# dst_nodes = edges[:, 1]

# unique_nodes = torch.unique(torch.cat([src_nodes, dst_nodes]))
# old_mem = {node.item(): torch.rand(8) for node in unique_nodes}  # Example: 8-dim memory

# # 1. Collect node ids for extended memory
# node_ids = torch.cat([
#     unique_nodes,        # Old memory (1,2,3,4)
#     src_nodes,           # Batch src nodes
#     dst_nodes            # Batch dst nodes
# ])

# # 2. Map node_id -> positions in extended memory
# node_pos_map = {}
# for i, nid in enumerate(node_ids.tolist()):
#     node_pos_map.setdefault(nid, []).append(i)

# # 3. Construct extended memory tensor
# extend_mem = torch.stack([old_mem[nid.item()] for nid in unique_nodes] +
#                          [old_mem[nid.item()] for nid in src_nodes] +
#                          [old_mem[nid.item()] for nid in dst_nodes])

# breakpoint()
# src_tensor = []
# dst_tensor = []

# # Create a reverse index: for each node, record the latest position in node_ids
# latest_mem_index = {}

# # Iterate chronologically through the batch
# for i in range(edges.size(0)):
#     s, d = edges[i, 0].item(), edges[i, 1].item()

#     # Get positions in extended memory
#     s_idx = node_pos_map[s][i+1]  # +1 because first is from old_mem
#     d_idx = node_pos_map[d][i+1]

#     # Both src and dst memory should be updated, we treat both as destination

#     # For src node
#     if s in latest_mem_index:
#         src_tensor.append(latest_mem_index[s])
#         dst_tensor.append(s_idx)
#     latest_mem_index[s] = s_idx

#     # For dst node
#     if d in latest_mem_index:
#         src_tensor.append(latest_mem_index[d])
#         dst_tensor.append(d_idx)
#     latest_mem_index[d] = d_idx


# import torch
# from collections import defaultdict
# from bisect import bisect_right

# def build_sparse_index(t: torch.Tensor):
#     value_indices = defaultdict(list)
#     n = t.size(0)

#     for row in range(n):
#         for col in range(2):
#             val = t[row, col].item()
#             value_indices[val].append(row)

#     return value_indices


# def sparse_query(value_indices, x, idx):
#     if x not in value_indices:
#         return 0
#     return bisect_right(value_indices[x], idx)


# t = torch.tensor([
#     [1, 2],
#     [3, 4],
#     [1, 5],
#     [6, 1],
#     [7, 8]
# ])

# value_indices = build_sparse_index(t)

# print(sparse_query(value_indices, 1, 3))  # 3 (rows 0, 2, 3)
# print(sparse_query(value_indices, 5, 4))  # 1
# print(sparse_query(value_indices, 9, 4))  # 0
