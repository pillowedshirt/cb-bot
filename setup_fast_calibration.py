from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fast_calibration_core",
        ["cpp/fast_calibration_core.cpp"],
        cxx_std=17,
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
