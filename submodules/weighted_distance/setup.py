from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == "nt":
    cxx_compiler_flags.append("/wd4624")

setup(
    name="weighted_distance",
    ext_modules=[
        CUDAExtension(
            name="weighted_distance._C",
            sources=["weighted_distance.cu", "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
