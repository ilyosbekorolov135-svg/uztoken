"""
C extension build script for uztoken._ctrie

Usage:
    python setup_ext.py build_ext --inplace
"""
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

ext = Extension(
    "uztoken._ctrie",
    sources=["uztoken/_ctrie.c"],
    extra_compile_args=["-O2"] if sys.platform != "win32" else ["/O2"],
)

class BuildExtOptional(build_ext):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception:
            print("WARNING: Failed to compile C extension. Falling back to pure Python.")

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtOptional},
)
