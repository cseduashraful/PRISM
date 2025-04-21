#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void find_chunk_from_last_ts_kernel(
    const int64_t* node_ids,
    const double* timestamps,
    const int64_t* chunk_map,
    const double* chunk_last_ts,
    int64_t* output_chunk_ids,
    int64_t* output_previous_chunk_ids,
    int num_queries,
    int max_chunk_per_node
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;

    int64_t node = node_ids[idx];
    double ts = timestamps[idx];

    int left = 0, right = max_chunk_per_node - 1;
    int64_t found_chunk = -1;
    int found_slot = -1;
    int64_t last_valid_chunk = -1;
    int last_valid_slot = -1;

    while (left <= right) {
        int mid = (left + right) / 2;
        int64_t chunk_id = chunk_map[node * max_chunk_per_node + mid];
        double last_ts = chunk_last_ts[node * max_chunk_per_node + mid];

        if (chunk_id == -1) {
            right = mid - 1;
            continue;
        }

        last_valid_chunk = chunk_id;
        last_valid_slot = mid;

        if (last_ts >= ts) {
            found_chunk = chunk_id;
            found_slot = mid;
            right = mid - 1;  // try earlier satisfying chunk
        } else {
            left = mid + 1;
        }
    }

    // If we found a chunk satisfying last_ts >= ts
    if (found_chunk != -1) {
        output_chunk_ids[idx] = found_chunk;

        if (found_slot > 0) {
            output_previous_chunk_ids[idx] = chunk_map[node * max_chunk_per_node + (found_slot - 1)];
        } else {
            output_previous_chunk_ids[idx] = -1;
        }
    }
    else if (last_valid_chunk != -1) {
        // fallback to last valid chunk
        output_chunk_ids[idx] = last_valid_chunk;

        if (last_valid_slot > 0) {
            output_previous_chunk_ids[idx] = chunk_map[node * max_chunk_per_node + (last_valid_slot - 1)];
        } else {
            output_previous_chunk_ids[idx] = -1;
        }
    }
    else {
        // no chunk at all
        output_chunk_ids[idx] = -1;
        output_previous_chunk_ids[idx] = -1;
    }
}

// __global__ void find_chunk_from_last_ts_kernel(
//     const int64_t* node_ids,
//     const double* timestamps,
//     const int64_t* chunk_map,
//     const double* chunk_last_ts,
//     int64_t* output_chunk_ids,
//     int64_t* output_previous_chunk_ids,
//     int num_queries,
//     int max_chunk_per_node
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= num_queries) return;

//     int64_t node = node_ids[idx];
//     double ts = timestamps[idx];

//     int left = 0, right = max_chunk_per_node - 1;
//     int64_t found_chunk = -1;
//     int found_slot = -1;

//     while (left <= right) {
//         int mid = (left + right) / 2;
//         int64_t chunk_id = chunk_map[node * max_chunk_per_node + mid];
//         double last_ts = chunk_last_ts[node * max_chunk_per_node + mid];

//         if (chunk_id == -1) {
//             right = mid - 1;
//             continue;
//         }

//         if (last_ts >= ts) {
//             found_chunk = chunk_id;
//             found_slot = mid;
//             right = mid - 1;
//         } else {
//             left = mid + 1;
//         }
//     }

//     output_chunk_ids[idx] = found_chunk;

//     if (found_slot > 0) {
//         output_previous_chunk_ids[idx] = chunk_map[node * max_chunk_per_node + (found_slot - 1)];
//     } else {
//         output_previous_chunk_ids[idx] = -1;
//     }
// }


std::vector<at::Tensor> find_chunk_from_last_ts(
    at::Tensor node_ids,
    at::Tensor timestamps,
    at::Tensor chunk_map,
    at::Tensor chunk_last_ts,
    int max_chunk_per_node
) {
    const int num_queries = node_ids.size(0);

    auto output_chunk_ids = torch::full({num_queries}, -1, torch::dtype(torch::kInt64).device(node_ids.device()));
    auto output_previous_chunk_ids = torch::full({num_queries}, -1, torch::dtype(torch::kInt64).device(node_ids.device()));

    const int threads = 256;
    const int blocks = (num_queries + threads - 1) / threads;

    find_chunk_from_last_ts_kernel<<<blocks, threads>>>(
        node_ids.data_ptr<int64_t>(),
        timestamps.data_ptr<double>(),
        chunk_map.data_ptr<int64_t>(),
        chunk_last_ts.data_ptr<double>(),
        output_chunk_ids.data_ptr<int64_t>(),
        output_previous_chunk_ids.data_ptr<int64_t>(),
        num_queries,
        max_chunk_per_node
    );

    return {output_chunk_ids, output_previous_chunk_ids};
}




// __global__ void find_chunk_from_last_ts_kernel(
//     const int64_t* node_ids,
//     const double* timestamps,
//     const int64_t* chunk_map,
//     const double* chunk_last_ts,
//     int64_t* output_chunk_ids,
//     int num_queries,
//     int max_chunk_per_node
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= num_queries) return;

//     int64_t node = node_ids[idx];
//     double ts = timestamps[idx];

//     int left = 0, right = max_chunk_per_node - 1;
//     int64_t candidate_chunk = -1;  // chunk satisfying last_ts >= ts
//     int64_t last_valid_chunk = -1; // last valid chunk assigned to this node

//     while (left <= right) {
//         int mid = (left + right) / 2;
//         int64_t chunk_id = chunk_map[node * max_chunk_per_node + mid];
//         double last_ts = chunk_last_ts[node * max_chunk_per_node + mid];

//         if (chunk_id == -1) {
//             right = mid - 1; // No chunk here
//             continue;
//         }

//         last_valid_chunk = chunk_id; // update last valid assigned chunk

//         if (last_ts >= ts) {
//             candidate_chunk = chunk_id; // found a chunk satisfying
//             right = mid - 1; // but try to find earlier satisfying chunk
//         } else {
//             left = mid + 1;
//         }
//     }

//     // Decide the output
//     if (candidate_chunk != -1) {
//         output_chunk_ids[idx] = candidate_chunk;
//     } else {
//         output_chunk_ids[idx] = last_valid_chunk; // fallback
//     }
// }

// at::Tensor find_chunk_from_last_ts(
//     at::Tensor node_ids,
//     at::Tensor timestamps,
//     at::Tensor chunk_map,
//     at::Tensor chunk_last_ts,
//     int max_chunk_per_node
// ) {
//     const int num_queries = node_ids.size(0);

//     auto output_chunk_ids = torch::full({num_queries}, -1, torch::dtype(torch::kInt64).device(node_ids.device()));

//     const int threads = 256;
//     const int blocks = (num_queries + threads - 1) / threads;

//     find_chunk_from_last_ts_kernel<<<blocks, threads>>>(
//         node_ids.data_ptr<int64_t>(),
//         timestamps.data_ptr<double>(),
//         chunk_map.data_ptr<int64_t>(),
//         chunk_last_ts.data_ptr<double>(),
//         output_chunk_ids.data_ptr<int64_t>(),
//         num_queries,
//         max_chunk_per_node
//     );

//     return output_chunk_ids;
// }



// __global__ void find_index_in_chunk_kernel(
//     const double* ts_chunks_batch,
//     const double* root_ts,
//     int64_t* output_indices,
//     int batch_size,
//     int chunk_size
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= batch_size) return;

//     const double ts = root_ts[idx];
//     const double* chunk_ptr = ts_chunks_batch + idx * chunk_size;

//     int left = 0, right = chunk_size - 1;
//     int64_t best_idx = -1;

//     while (left <= right) {
//         int mid = (left + right) / 2;
//         double val = chunk_ptr[mid];

//         if (val < ts) {
//             best_idx = mid; // candidate
//             left = mid + 1;
//         } else {
//             right = mid - 1;
//         }
//     }

//     output_indices[idx] = best_idx;
// }

// at::Tensor find_index_in_chunk(
//     at::Tensor ts_chunks_batch,
//     at::Tensor root_ts,
//     int chunk_size
// ) {
//     const int batch_size = ts_chunks_batch.size(0);

//     auto output_indices = torch::full({batch_size}, -1, torch::dtype(torch::kInt64).device(ts_chunks_batch.device()));

//     const int threads = 256;
//     const int blocks = (batch_size + threads - 1) / threads;

//     find_index_in_chunk_kernel<<<blocks, threads>>>(
//         ts_chunks_batch.data_ptr<double>(),
//         root_ts.data_ptr<double>(),
//         output_indices.data_ptr<int64_t>(),
//         batch_size,
//         chunk_size
//     );

//     return output_indices;
// }

__global__ void find_index_in_chunk_kernel(
    const double* ts_chunks_batch,
    const double* root_ts,
    int64_t* output_indices,
    bool* is_previous_chunk_needed,
    int batch_size,
    int chunk_size,
    int k  // <-- new
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const double ts = root_ts[idx];
    const double* chunk_ptr = ts_chunks_batch + idx * chunk_size;

    int left = 0, right = chunk_size - 1;
    int64_t best_idx = -1;

    while (left <= right) {
        int mid = (left + right) / 2;
        double val = chunk_ptr[mid];

        if (val < ts) {
            best_idx = mid; // candidate
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    output_indices[idx] = best_idx;

    // New: decide if previous chunk is needed
    if (best_idx == -1) {
        is_previous_chunk_needed[idx] = true;  // No ts found, need previous
    } else if (best_idx < k) {
        is_previous_chunk_needed[idx] = true;  // Too close to start, need to look back
    } else {
        is_previous_chunk_needed[idx] = false; // Good enough, no previous chunk needed
    }
}


at::Tensor find_index_in_chunk(
    at::Tensor ts_chunks_batch,
    at::Tensor root_ts,
    int chunk_size,
    int k  // <-- new
) {
    const int batch_size = ts_chunks_batch.size(0);

    auto output_indices = torch::full({batch_size}, -1, torch::dtype(torch::kInt64).device(ts_chunks_batch.device()));
    auto prev_chunk_flags = torch::full({batch_size}, false, torch::dtype(torch::kBool).device(ts_chunks_batch.device()));

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    find_index_in_chunk_kernel<<<blocks, threads>>>(
        ts_chunks_batch.data_ptr<double>(),
        root_ts.data_ptr<double>(),
        output_indices.data_ptr<int64_t>(),
        prev_chunk_flags.data_ptr<bool>(),  // new
        batch_size,
        chunk_size,
        k
    );

    return torch::stack({output_indices, prev_chunk_flags.to(torch::kInt64)}, 1);  // Return both
}

// __global__ void collect_prev_k_ts_kernel(
//     const int64_t* chunk_ids,               // Current chunk id (mapped to local ids in ts_chunks_selected)
//     const int64_t* previous_chunk_ids,       // Previous chunk id (mapped to local ids)
//     const int64_t* indices_in_chunk,          // Start index inside current chunk
//     const double* ts_chunks_selected,        // Flattened selected ts_chunks
//     int64_t* output_ts_indices,
//     int batch_size,
//     int chunk_size,
//     int k
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= batch_size) return;

//     int64_t current_chunk_id = chunk_ids[idx];
//     int64_t prev_chunk_id = previous_chunk_ids[idx];
//     int64_t current_idx = indices_in_chunk[idx];

//     int collected = 0;

//     while (collected < k && current_chunk_id != -1) {
//         while (current_idx >= 0 && collected < k) {
//             output_ts_indices[idx * k + collected] = current_chunk_id * chunk_size + current_idx;
//             current_idx--;
//             collected++;
//         }

//         if (collected < k && prev_chunk_id != -1) {
//             current_chunk_id = prev_chunk_id;
//             current_idx = chunk_size - 1;
//             prev_chunk_id = -1;  // Only fallback once
//         } else {
//             break;
//         }
//     }

//     while (collected < k) {
//         output_ts_indices[idx * k + collected] = -1;
//         collected++;
//     }
// }
// __global__ void collect_prev_k_ts_kernel(
//     const int64_t* chunk_ids,
//     const int64_t* previous_chunk_ids,
//     const int64_t* indices_in_chunk,
//     const double* ts_chunks_selected,
//     int64_t* output_ts_indices,
//     int batch_size,
//     int chunk_size,
//     int k
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= batch_size) return;

//     int64_t current_chunk_id = chunk_ids[idx];
//     int64_t prev_chunk_id = previous_chunk_ids[idx];
//     int64_t current_idx = indices_in_chunk[idx];

//     int collected = 0;

//     if (current_chunk_id != -1) {
//         // 1. Collect from current chunk
//         while (current_idx >= 0 && collected < k) {
//             output_ts_indices[idx * k + collected] = current_chunk_id * chunk_size + current_idx;
//             current_idx--;
//             collected++;
//         }
//     }

//     if (collected < k && prev_chunk_id != -1) {
//         // 2. Collect from previous chunk
//         current_idx = chunk_size - 1;
//         while (current_idx >= 0 && collected < k) {
//             output_ts_indices[idx * k + collected] = prev_chunk_id * chunk_size + current_idx;
//             current_idx--;
//             collected++;
//         }
//     }

//     // 3. Pad the rest with -1 if not enough
//     while (collected < k) {
//         output_ts_indices[idx * k + collected] = -1;
//         collected++;
//     }
// }

// at::Tensor collect_prev_k_ts(
//     at::Tensor chunk_ids,
//     at::Tensor previous_chunk_ids,
//     at::Tensor indices_in_chunk,
//     at::Tensor ts_chunks_selected,
//     int chunk_size,
//     int k
// ) {
//     const int batch_size = chunk_ids.size(0);

//     auto output_ts_indices = torch::full({batch_size, k}, -1, torch::dtype(torch::kInt64).device(chunk_ids.device()));

//     const int threads = 256;
//     const int blocks = (batch_size + threads - 1) / threads;

//     collect_prev_k_ts_kernel<<<blocks, threads>>>(
//         chunk_ids.data_ptr<int64_t>(),
//         previous_chunk_ids.data_ptr<int64_t>(),
//         indices_in_chunk.data_ptr<int64_t>(),
//         ts_chunks_selected.data_ptr<double>(),   // Corrected here ✅
//         output_ts_indices.data_ptr<int64_t>(),
//         batch_size,
//         chunk_size,
//         k
//     );

//     return output_ts_indices;
// }

// __global__ void collect_prev_k_ts_kernel(
//     const int64_t* chunk_ids,
//     const int64_t* previous_chunk_ids,
//     const int64_t* indices_in_chunk,
//     const double* ts_chunks_selected,
//     int64_t* output_ts_indices,
//     int batch_size,
//     int chunk_size,
//     int k
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= batch_size) return;

//     int64_t current_chunk_id = chunk_ids[idx];
//     int64_t prev_chunk_id = previous_chunk_ids[idx];
//     int64_t current_idx = indices_in_chunk[idx];

//     int collected = 0;
//     bool still_have_chunk = (current_chunk_id != -1);
//     bool tried_prev = false;


//     printf("Hello from thread %d, idx=%d, chunk_id=%lld, previous chunk_id=%lld\n", 
//        idx, idx, current_chunk_id, prev_chunk_id);

//     while (collected < k) {
//         if (still_have_chunk && current_idx >= 0) {
//             output_ts_indices[idx * k + collected] = current_chunk_id * chunk_size + current_idx;
//             current_idx--;
//             collected++;
//         } 
//         else if (still_have_chunk && current_idx < 0 && prev_chunk_id != -1 && !tried_prev) {
//             // Move to previous chunk once
//             current_chunk_id = prev_chunk_id;
//             current_idx = chunk_size - 1;
//             prev_chunk_id = -1;
//             tried_prev = true;
//         } 
//         else {
//             // No more timestamps available, pad with -1
//             output_ts_indices[idx * k + collected] = -1;
//             collected++;
//         }
//     }
// }

__global__ void collect_prev_k_ts_kernel(
    const int64_t* chunk_ids,
    const int64_t* previous_chunk_ids,
    const int64_t* indices_in_chunk,
    const double* ts_chunks_selected,
    int64_t* output_ts_indices,
    int batch_size,
    int chunk_size,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int64_t current_chunk_id = chunk_ids[idx];
    int64_t prev_chunk_id = previous_chunk_ids[idx];
    int64_t current_idx = indices_in_chunk[idx];

    int collected = 0;
    bool switched_to_prev = false;

    // printf("Hello from thread %d, idx=%d, chunk_id=%lld, previous chunk_id=%lld and current_idx=%lld\n", 
    //     idx, idx, current_chunk_id, prev_chunk_id, current_idx);

    while (collected < k) {
        if (current_idx >= 0) {
            output_ts_indices[idx * k + collected] = current_chunk_id * chunk_size + current_idx;
            current_idx--;
            collected++;
        }
        else if (!switched_to_prev && prev_chunk_id != -1) {
            // switch to previous chunk ONCE
            current_chunk_id = prev_chunk_id;
            current_idx = chunk_size - 1;
            switched_to_prev = true;
        }
        else {
            // can't collect more
            output_ts_indices[idx * k + collected] = -1;
            collected++;
        }
    }
}


at::Tensor collect_prev_k_ts(
    at::Tensor chunk_ids,
    at::Tensor previous_chunk_ids,
    at::Tensor indices_in_chunk,
    at::Tensor ts_chunks_selected,
    int chunk_size,
    int k
) {
    const int batch_size = chunk_ids.size(0);

    auto output_ts_indices = torch::full({batch_size, k}, -1, torch::dtype(torch::kInt64).device(chunk_ids.device()));

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    // std::cout << "indices_in_chunk: ";
    // auto indices_in_chunk_cpu = indices_in_chunk.to(torch::kCPU);  // move to CPU
    // auto accessor = indices_in_chunk_cpu.accessor<int64_t, 1>();   // 1D tensor

    // for (int i = 0; i < indices_in_chunk_cpu.size(0); ++i) {
    //     std::cout << accessor[i] << " ";
    // }
    // std::cout << std::endl;

    indices_in_chunk = indices_in_chunk.contiguous();
    collect_prev_k_ts_kernel<<<blocks, threads>>>(
        chunk_ids.data_ptr<int64_t>(),
        previous_chunk_ids.data_ptr<int64_t>(),
        indices_in_chunk.data_ptr<int64_t>(),
        ts_chunks_selected.data_ptr<double>(),
        output_ts_indices.data_ptr<int64_t>(),
        batch_size,
        chunk_size,
        k
    );

    return output_ts_indices;
}








__global__ void fused_find_and_collect_kernel(
    const double* __restrict__ ts_chunks_selected,
    const int64_t* __restrict__ chunk_ids,
    const int64_t* __restrict__ previous_chunk_ids,
    const double* __restrict__ root_ts,
    int64_t* __restrict__ output_ts_indices,
    int batch_size,
    int chunk_size,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    double query_ts = root_ts[idx];
    int64_t cur_chunk_id = chunk_ids[idx];
    int64_t prev_chunk_id = previous_chunk_ids[idx];

    // Pointer to current chunk
    const double* chunk_ptr = ts_chunks_selected + cur_chunk_id * chunk_size;

    // Step 1: Binary search to find best_idx within current chunk
    int left = 0, right = chunk_size - 1;
    int64_t best_idx = -1;

    while (left <= right) {
        int mid = (left + right) / 2;
        double val = chunk_ptr[mid];

        if (val < query_ts) {
            best_idx = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    // Step 2: Collect k timestamps
    int collected = 0;
    bool switched_to_prev = false;

    while (collected < k) {
        if (best_idx >= 0) {
            output_ts_indices[idx * k + collected] = cur_chunk_id * chunk_size + best_idx;
            best_idx--;
            collected++;
        }
        else if (!switched_to_prev && prev_chunk_id != -1) {
            // Switch to previous chunk
            cur_chunk_id = prev_chunk_id;
            chunk_ptr = ts_chunks_selected + cur_chunk_id * chunk_size;
            best_idx = chunk_size - 1;
            switched_to_prev = true;
        }
        else {
            // No more timestamps available
            output_ts_indices[idx * k + collected] = -1;
            collected++;
        }
    }
}

at::Tensor fused_find_and_collect(
    at::Tensor ts_chunks_selected,
    at::Tensor chunk_ids,
    at::Tensor previous_chunk_ids,
    at::Tensor root_ts,
    int chunk_size,
    int k
) {
    const int batch_size = chunk_ids.size(0);

    auto output = torch::full({batch_size, k}, -1, torch::dtype(torch::kInt64).device(ts_chunks_selected.device()));

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    fused_find_and_collect_kernel<<<blocks, threads>>>(
        ts_chunks_selected.data_ptr<double>(),
        chunk_ids.data_ptr<int64_t>(),
        previous_chunk_ids.data_ptr<int64_t>(),
        root_ts.data_ptr<double>(),
        output.data_ptr<int64_t>(),
        batch_size,
        chunk_size,
        k
    );

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_chunk_from_last_ts", &find_chunk_from_last_ts, "Find chunk id using last timestamp");
    m.def("find_index_in_chunk", &find_index_in_chunk, "Find chunk id using last timestamp");
    m.def("collect_prev_k_ts", &collect_prev_k_ts, "Collect previous k timestamps across chunks");
    m.def("fused_find_and_collect", &fused_find_and_collect, "find index and then collect previous k events");

}
