#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void recent_index_kernel(
    const int64_t* ei_src,
    const int64_t* ei_dst,
    const int64_t* pos_node_s,
    const int64_t* pos_node_d,
    int64_t* recent_indices,
    int batch_size,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    int64_t target = ei_src[idx];
    int64_t max_idx = ei_dst[idx] % batch_size;
    int64_t found = -1;

    // Search backwards up to max_idx
    for (int j = max_idx - 1; j >= 0; --j) {
        if (pos_node_s[j] == target) {
            found = j;
            break;
        } else if (pos_node_d[j] == target) {
            found = j + batch_size;
            break;
        }
    }

    recent_indices[idx] = found;
}



__device__ inline int64_t compute_dst(int64_t cond, int64_t bs) {
    return (cond < bs) ? cond + bs : cond % bs;
}

__global__ void build_mem_graph_kernel(
    const int64_t* __restrict__ conds,
    const int64_t* __restrict__ keys,
    const int64_t* __restrict__ bidxs,
    const int64_t* __restrict__ od_updated,
    const int64_t* __restrict__ match_starts,
    const int64_t* __restrict__ match_ends,
    const int64_t* __restrict__ match_indices,
    int64_t max_seen_eid,
    int64_t bs,
    int64_t* ei_src,
    int64_t* ei_dst,
    int64_t* ei_dla,
    int64_t* ei_eid,
    int64_t* msg_store_src,
    int64_t* msg_store_dst,
    int64_t* msg_store_nid,
    int64_t* msg_store_eid,
    int64_t* counter_edge,
    int64_t* counter_store,
    int64_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int64_t key = keys[i];
    if (key == -1) return;

    int64_t cond = conds[i];
    int64_t bidx = bidxs[i];
    int64_t dst = compute_dst(cond, bs);

    int start = match_starts[key];
    int end   = match_ends[key];
    bool has_valid = false;

    for (int j = start; j < end; j++) {
        int64_t idx = match_indices[j];
        int64_t match_bidx = od_updated[idx * 4 + 3];  // od_updated shape: (N, 4)

        if (match_bidx > bidx) {
            int write_idx = atomicAdd(counter_edge, 1);
            ei_src[write_idx] = cond;
            ei_dst[write_idx] = dst;
            ei_dla[write_idx] = idx;
            ei_eid[write_idx] = bidx + max_seen_eid + 1;
            has_valid = true;
        }
    }

    if (!has_valid) {
        int store_idx = atomicAdd(counter_store, 1);
        msg_store_src[store_idx] = cond;
        msg_store_dst[store_idx] = dst;
        msg_store_nid[store_idx] = key;
        msg_store_eid[store_idx] = bidx + max_seen_eid + 1;
    }
}


#include <torch/extension.h>

torch::Tensor recent_index_cuda(
    torch::Tensor ei_src,
    torch::Tensor ei_dst,
    torch::Tensor pos_node_s,
    torch::Tensor pos_node_d,
    int batch_size
) {
    auto recent_indices = torch::full(
        {ei_src.size(0)}, -1, ei_src.options());

    const int threads = 128;
    const int blocks = (ei_src.size(0) + threads - 1) / threads;

    recent_index_kernel<<<blocks, threads>>>(
        ei_src.data_ptr<int64_t>(),
        ei_dst.data_ptr<int64_t>(),
        pos_node_s.data_ptr<int64_t>(),
        pos_node_d.data_ptr<int64_t>(),
        recent_indices.data_ptr<int64_t>(),
        batch_size,
        ei_src.size(0)
    );

    return recent_indices;
}

void build_mem_graph_cuda(
    torch::Tensor conds,
    torch::Tensor keys,
    torch::Tensor bidxs,
    torch::Tensor od_updated,
    torch::Tensor match_starts,
    torch::Tensor match_ends,
    torch::Tensor match_indices,
    int64_t max_seen_eid,
    int64_t bs,
    torch::Tensor ei_src,
    torch::Tensor ei_dst,
    torch::Tensor ei_dla,
    torch::Tensor ei_eid,
    torch::Tensor msg_store_src,
    torch::Tensor msg_store_dst,
    torch::Tensor msg_store_nid,
    torch::Tensor msg_store_eid,
    torch::Tensor counter_edge,
    torch::Tensor counter_store
) {
    int N = conds.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    build_mem_graph_kernel<<<blocks, threads>>>(
        conds.data_ptr<int64_t>(),
        keys.data_ptr<int64_t>(),
        bidxs.data_ptr<int64_t>(),
        od_updated.data_ptr<int64_t>(),
        match_starts.data_ptr<int64_t>(),
        match_ends.data_ptr<int64_t>(),
        match_indices.data_ptr<int64_t>(),
        max_seen_eid,
        bs,
        ei_src.data_ptr<int64_t>(),
        ei_dst.data_ptr<int64_t>(),
        ei_dla.data_ptr<int64_t>(),
        ei_eid.data_ptr<int64_t>(),
        msg_store_src.data_ptr<int64_t>(),
        msg_store_dst.data_ptr<int64_t>(),
        msg_store_nid.data_ptr<int64_t>(),
        msg_store_eid.data_ptr<int64_t>(),
        counter_edge.data_ptr<int64_t>(),
        counter_store.data_ptr<int64_t>(),
        N
    );

    cudaDeviceSynchronize();  // Optional: debugging
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mem_graph", &recent_index_cuda, "Recent index CUDA for mem update graph");
    m.def("build_mem_graph", &build_mem_graph_cuda, "Build memory graph edges and store entries");

}
