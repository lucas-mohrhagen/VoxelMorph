# module load gcc

from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
ext_modules = cythonize("triangle_hash.pyx"),
include_dirs=[numpy.get_include()]
)