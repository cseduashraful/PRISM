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
            int write_idx = atomicAdd((unsigned long long int*)counter_edge, 1ULL);//atomicAdd(counter_edge, 1);
            ei_src[write_idx] = cond;
            ei_dst[write_idx] = dst;
            ei_dla[write_idx] = idx;
            ei_eid[write_idx] = bidx + max_seen_eid + 1;
            has_valid = true;
        }
    }

    if (!has_valid) {
        int store_idx = atomicAdd((unsigned long long int*)counter_store, 1ULL);//atomicAdd(counter_store, 1);
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

/////APAN Start
// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// __global__ void build_mem_graph_apan_kernel(
//     const int64_t* __restrict__ od_data,          // [num_rows * 5]
//     const int64_t* __restrict__ match_ptr,        // [num_keys + 1]
//     const int64_t* __restrict__ match_indices,    // flattened match index list
//     int64_t num_rows,
//     int64_t bs,
//     int64_t max_seen_eid,
//     int64_t* __restrict__ mem_graph_out,          // [4, max_edges]
//     int64_t* __restrict__ store_quad_out,         // [4, max_msgs]
//     // int64_t* __restrict__ edge_counter,
//     // int64_t* __restrict__ msg_counter
//     int32_t* __restrict__ edge_counter,
//     int32_t* __restrict__ msg_counter
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_rows) return;

//     int64_t cond = od_data[tid * 5 + 0];
//     int64_t key  = od_data[tid * 5 + 1];
//     int64_t bidx = od_data[tid * 5 + 3];

//     if (key == -1) return;

//     int64_t eid_val = bidx + max_seen_eid + 1;
//     int64_t dst_val = (cond < bs) ? cond + bs : cond % bs;

//     int64_t start = match_ptr[key];
//     int64_t end   = match_ptr[key + 1];

//     bool has_valid = false;
//     for (int64_t j = start; j < end; ++j) {
//         int64_t idx = match_indices[j];
//         int64_t other_bidx = od_data[idx * 5 + 3];
//         if (other_bidx > bidx) {
//             int pos = atomicAdd(edge_counter, 1);

//             mem_graph_out[pos + 0 * num_rows] = cond;
//             mem_graph_out[pos + 1 * num_rows] = dst_val;
//             mem_graph_out[pos + 2 * num_rows] = idx;
//             mem_graph_out[pos + 3 * num_rows] = eid_val;

//             has_valid = true;
//         }
//     }

//     if (!has_valid) {
//         int pos = atomicAdd(msg_counter, 1);

//         store_quad_out[pos + 0 * num_rows] = cond;
//         store_quad_out[pos + 1 * num_rows] = dst_val;
//         store_quad_out[pos + 2 * num_rows] = key;
//         store_quad_out[pos + 3 * num_rows] = eid_val;
//     }
// }



// // Forward declaration for PyBind
// void launch_build_mem_graph_apan_kernel(
//     at::Tensor od_data,
//     at::Tensor match_ptr,
//     at::Tensor match_indices,
//     int64_t bs,
//     int64_t max_seen_eid,
//     at::Tensor mem_graph_out,
//     at::Tensor store_quad_out,
//     at::Tensor edge_counter,
//     at::Tensor msg_counter
// ) {
//     const int num_rows = od_data.size(0);
//     const int threads = 256;
//     const int blocks = (num_rows + threads - 1) / threads;

//     build_mem_graph_apan_kernel<<<blocks, threads>>>(
//         od_data.data_ptr<int64_t>(),
//         match_ptr.data_ptr<int64_t>(),
//         match_indices.data_ptr<int64_t>(),
//         num_rows,
//         bs,
//         max_seen_eid,
//         mem_graph_out.data_ptr<int64_t>(),
//         store_quad_out.data_ptr<int64_t>(),
//         edge_counter.data_ptr<int>(),
//         msg_counter.data_ptr<int>()
//     );
// }

// /////APAN End




// __global__ void find_matches_kernel(
//     const int64_t* __restrict__ key_vals_all,  // [M]
//     const int64_t* __restrict__ match_vals_all,  // [N]
//     int64_t* out_i_valid,  // [MAX_MATCHES]
//     int64_t* out_match_idx,  // [MAX_MATCHES]
//     int64_t* match_count,  // single int64_t counter (on device, use atomicAdd)
//     int M, int N
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= M) return;

//     int64_t key = key_vals_all[tid];
//     for (int j = 0; j < N; ++j) {
//         if (key == match_vals_all[j]) {
//             // int idx = atomicAdd(match_count, 1);
//             // long long int idx = atomicAdd((long long int*)match_count, 1);
//             // long long int idx = atomicAdd((long long int*)match_count, (long long int)1);
//             unsigned long long int idx = atomicAdd(reinterpret_cast<unsigned long long int*>(match_count), 1ULL);


//             out_i_valid[idx] = tid;
//             out_match_idx[idx] = j;
//         }
//     }
// }

// void launch_match_kernel(
//     torch::Tensor key_vals_all,
//     torch::Tensor match_vals_all,
//     torch::Tensor out_i_valid,
//     torch::Tensor out_match_idx,
//     torch::Tensor match_count) {

//     const int M = key_vals_all.size(0);
//     const int N = match_vals_all.size(0);

//     const int threads = 256;
//     const int blocks = (M + threads - 1) / threads;

//     find_matches_kernel<<<blocks, threads>>>(
//         key_vals_all.data_ptr<int64_t>(),
//         match_vals_all.data_ptr<int64_t>(),
//         out_i_valid.data_ptr<int64_t>(),
//         out_match_idx.data_ptr<int64_t>(),
//         match_count.data_ptr<int64_t>(),
//         M, N
//     );
// }



// std::vector<torch::Tensor> match_indices(
//     torch::Tensor key_vals_all,
//     torch::Tensor match_vals_all,
//     int64_t max_matches) {

//     auto options = key_vals_all.options().dtype(torch::kInt64);
//     auto out_i_valid = torch::empty({max_matches}, options);
//     auto out_match_idx = torch::empty({max_matches}, options);
//     auto match_count = torch::zeros({1}, options.device(key_vals_all.device()));

//     launch_match_kernel(key_vals_all, match_vals_all, out_i_valid, out_match_idx, match_count);

//     auto actual_count = match_count.item<int64_t>();
//     return {
//         out_i_valid.slice(0, 0, actual_count),
//         out_match_idx.slice(0, 0, actual_count)
//     };
// }















PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mem_graph", &recent_index_cuda, "Recent index CUDA for mem update graph");
    // m.def("match_indices", &match_indices, "Efficient matching indices (CUDA)");
    // m.def("find_matches", &find_matches, "find matches kernel for apan");
    // m.def("build_mem_graph", &build_mem_graph_cuda, "Build memory graph edges and store entries");
    // m.def("apan_mem_graph", &launch_build_mem_graph_apan_kernel, "Recent index CUDA for mem update graph");

}
