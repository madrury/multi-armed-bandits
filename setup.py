from distutils.core import setup, Extension
from codecs import open
from os import path

import numpy as np
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='multi-armed-bandits',
    varsion='0.0.1',
    description="Tools for Multi Armed Bandit Simulations",
    long_description=long_description,
    url='https://github.com/madrury/multi-armed-bandits',
    author='Matthew Drury',
    author_email='matthew.drury.83@gmail.com',
    ext_modules=cythonize(
        Extension(
            "bandits",
            sources=["bandits.pyx"],
            include_dirs=[np.get_include()]),
        annotate=True),
    install_requires=['numpy', 'cython']
)
