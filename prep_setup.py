# # setup.py
# import sys
# import sysconfig
# import platform
# from pathlib import Path
# from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext

# # ---- Helpers to fetch pybind11 include path even if not installed yet ----
# class get_pybind_include(object):
#     """Defer importing pybind11 until it's actually installed."""
#     def __init__(self, user=False):
#         self.user = user
#     def __str__(self):
#         import pybind11
#         return pybind11.get_include(self.user)

# class BuildExt(build_ext):
#     """Add compiler-specific options (OpenMP, etc.)."""
#     c_opts = {
#         "msvc": ["/std:c++17", "/O2"],
#         "unix": ["-O3", "-std=c++17"],
#     }
#     l_opts = {
#         "msvc": [],
#         "unix": [],
#     }

#     def build_extensions(self):
#         ct = self.compiler.compiler_type
#         opts = self.c_opts.get(ct, [])
#         link_opts = self.l_opts.get(ct, [])

#         system = platform.system()

#         # --- OpenMP flags by platform ---
#         if ct == "msvc":
#             # Visual Studio
#             opts.append("/openmp")
#         else:
#             if system == "Darwin":
#                 # Apple Clang: needs libomp installed (e.g. `brew install libomp`)
#                 # and these flags to enable OpenMP.
#                 opts += ["-Xpreprocessor", "-fopenmp"]
#                 link_opts += ["-lomp"]
#                 # Slightly stricter warnings are fine to remove:
#                 # opts += ["-Wall", "-Wextra"]
#             else:
#                 # Linux / other Unix (GCC/Clang with libgomp)
#                 opts.append("-fopenmp")
#                 link_opts.append("-fopenmp")

#         # On some systems you may need to bump the macOS deployment target
#         if system == "Darwin":
#             mac_target = "10.15"
#             cur = sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
#             if not cur or tuple(map(int, cur.split("."))) < tuple(map(int, mac_target.split("."))):
#                 os_env = self.distribution.get_command_obj('build_ext')._get_export_symbols
#                 # Not strictly necessary; modern pip sets it for you. Skip explicit env edits.

#         for ext in self.extensions:
#             ext.extra_compile_args = ext.extra_compile_args + opts
#             ext.extra_link_args = ext.extra_link_args + link_opts
#         super().build_extensions()

# ext_modules = [
#     Extension(
#         "chunkio",
#         sources=["chunk_streaming.cpp"],
#         include_dirs=[
#             # pybind11 headers
#             get_pybind_include(),
#             get_pybind_include(user=True),
#         ],
#         language="c++",
#         extra_compile_args=[],
#         extra_link_args=[],
#     ),
# ]

# setup(
#     name="chunkio",
#     version="0.1.0",
#     description="Streaming chunk preprocessor with on-disk storage and Python access",
#     author="",
#     ext_modules=ext_modules,
#     cmdclass={"build_ext": BuildExt},
#     zip_safe=False,
#     install_requires=[
#         "pybind11>=2.10",
#         # numpy is optional at build time; used at runtime when returning arrays
#     ],
# )


from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="chunkio",
    ext_modules=[
        CppExtension(
            "chunkio",
            ["chunk_streaming.cpp"],
            define_macros=[("CHUNKIO_WITH_TORCH", "1")],
            extra_compile_args={"cxx": ["-O3", "-std=c++17", "-fopenmp"]},
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
