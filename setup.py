from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# # Optionally override architecture list if not set
# os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0")

setup(
    name='multi_extensions_project',
    ext_modules=[
        # Preprocessing C++ Extension
        CppExtension(
            'preprocessor',  # Python module name
            ['preprocess.cpp', 'binding.cpp'],
            extra_compile_args=["-fopenmp", "-std=c++17"],
            extra_link_args=["-fopenmp"]
        ),
        # Find chunk CUDA Extension
        CUDAExtension(
            'sampler',  # Python module name
            ['sampler.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
