# Recent-K Sampler

Fast GPU-friendly sampler for temporal graph learning.

Features:
- CUDA-accelerated neighbor finding
- OpenMP C++ preprocessing
- Chunk-based efficient memory handling
- Chronologically optimized batch sampling


# Address Intra-batch Temporal Discontinuity using Aggregation

Multi-version memory management

Features:
- CUDA accelerated memory-update graph generation
- approximate dependency-aware multi-version intra-batch memory
- k-layer memory update module to push approximation k-hop away


To build the prepocessor, mem_graph generator and sampler

    python setup.py build_ext --inplace


We built the Sampler on cuda 11.8
