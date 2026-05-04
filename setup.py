import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# Build fat binaries once so compiled extensions run across common cluster GPUs:
# 7.5=RTX 2080 Ti, 8.0=A100, 8.6=A16/A40, 8.9=L4/L40S, 9.0+PTX=H100 forward-compatible PTX.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0+PTX")

setup(
    name="prism",
    version="0.1.0",
    description="PRISM: Parallel Refinement of Intra-Batch Staleness in MTGNN",
    packages=find_packages(include=["prism", "prism.*", "modules", "modules.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "requests",
        "beautifulsoup4",
        "gdown",
        "clint",
        "torch",
        "torch-geometric",
        "py-tgb",
        "tqdm",
    ],
    ext_modules=[
        CppExtension(
            "preprocessor",
            ["preprocess.cpp", "binding.cpp"],
            extra_compile_args=["-O3", "-fopenmp", "-std=c++17"],
            extra_link_args=["-fopenmp"],
        ),
        CUDAExtension(
            "sampler",
            ["sampler.cu"],
        ),
        CUDAExtension(
            "mem_update_graph",
            ["mem_update_graph.cu"],
        ),
        CUDAExtension(
            "mapped_scatter",
            ["mapped_scatter.cu"],
        ),
        CppExtension(
            "chunkio",
            ["chunk_streaming.cpp"],
            define_macros=[("CHUNKIO_WITH_TORCH", "1")],
            extra_compile_args=["-O3", "-fopenmp", "-std=c++17"],
            extra_link_args=["-fopenmp"],
        ),
    ],
    entry_points={
        "console_scripts": [
            "prism-negatives=prism.negatives:main",
        ]
    },
    cmdclass={"build_ext": BuildExtension},
)
