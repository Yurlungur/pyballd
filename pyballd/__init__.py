#!/usr/bin/env python2

"""pyballd
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-13 16:09:32 (jmiller)>

pyballd is an elliptic solver for axisymmetric problems
implemented in Python. It uses spectral methods
for maximum efficiency.
"""
from .version import __version__
from elliptic import pde_solve_once
