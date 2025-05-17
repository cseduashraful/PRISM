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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mem_graph", &recent_index_cuda, "Recent index CUDA for mem update graph");
}
