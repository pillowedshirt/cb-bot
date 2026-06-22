from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys


extra_compile_args = []

if sys.platform.startswith("win"):
    extra_compile_args = [
        "/O2",
        "/EHsc",
        "/bigobj",
        "/std:c++17",
    ]
else:
    extra_compile_args = [
        "-O3",
        "-std=c++17",
    ]


ext_modules = [
    Pybind11Extension(
        "fast_calibration_core",
        ["cpp/fast_calibration_core.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
    Pybind11Extension(
        "fast_institutional_core",
        ["fast_institutional_core.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]


setup(
    name="fast_calibration_core",
    version="0.1.0",
    description="Fast C++ calibration core for the Binance.US trading bot",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
