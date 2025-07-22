#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void scatter_add_mapidx_kernel(
    const scalar_t* __restrict__ unique_vals,   // [M, D]
    const int64_t* __restrict__ map_idx,        // [N]
    const int64_t* __restrict__ index,          // [N]
    scalar_t* __restrict__ out,                 // [dim_size, D]
    int64_t N,
    int64_t D,
    int64_t dim_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int64_t src_idx = map_idx[i];
    int64_t dst_idx = index[i];

    for (int d = 0; d < D; d++) {
        atomicAdd(&out[dst_idx * D + d], unique_vals[src_idx * D + d]);
    }
}

void scatter_add_mapped(
    at::Tensor unique_vals,  // [M, D]
    at::Tensor map_idx,      // [N]
    at::Tensor index,        // [N]
    at::Tensor out           // [dim_size, D]
) {
    const int64_t N = map_idx.size(0);
    const int64_t D = unique_vals.size(1);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(unique_vals.scalar_type(), "scatter_add_mapidx_kernel", ([&] {
        scatter_add_mapidx_kernel<scalar_t><<<blocks, threads>>>(
            unique_vals.data_ptr<scalar_t>(),
            map_idx.data_ptr<int64_t>(),
            index.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            N, D, out.size(0)
        );
    }));

    cudaDeviceSynchronize();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_add_mapped", &scatter_add_mapped, "Scatter Add with MapIdx");
}