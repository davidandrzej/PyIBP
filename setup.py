from distutils.core import Extension, setup
import os

from numpy.distutils.misc_util import *

py_mods = ['PyIBP']

setup(name = 'PyIBP',
      description = 'Implementation of accelerated sampling for Linear-Gaussian Indian Buffet Process (IBP) model using SciPy/NumPy',
      version = '0.0.0',
      author = 'David Andrzejewski',
      author_email = 'andrzeje@cs.wisc.edu',
      license = 'GNU General Public License (Version 3 or later)',
      url = 'http://pages.cs.wisc.edu/~andrzeje',
      py_modules = py_mods)
