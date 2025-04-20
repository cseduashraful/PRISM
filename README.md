# Recent-K Sampler

Fast GPU-friendly sampler for temporal graph learning.

Features:
- CUDA-accelerated neighbor finding
- OpenMP C++ preprocessing
- Chunk-based efficient memory handling
- Chronologically optimized batch sampling

To build the prepocessor and sampler

    python setup.py build_ext --inplace


We built the Sampler on cuda 11.8