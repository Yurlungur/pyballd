#!/usr/bin/env python2

"""elliptic.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-13 13:39:41 (jmiller)>

This is a module for pyballd. It contains the routines required for
solving elliptic systems.
"""

from enum import Enum
import numpy as np
import scipy as sp

DEFAULT_ORDER_X = 100
DEFAULT_ORDER_THETA = 24

CmpType = Enum('standard','BH')

def pde_solve_once(residual,
                   order_X = DEFAULT_ORDER_X,
                   order_theta = DEFAULT_ORDER_THETA,
                   bdry_X_inner=None,
                   bdry_theta_min=None,
                   bdry_theta_max=None,
                   compactification_type=CmpType.standard,
                   ):
    """Solves an elliptic PDE system.

    The PDE is defined via a RESIDUAL. A residual

    L[r,theta,u,
      (d/dr),(d/dtheta),
      (d^2/dr^2),(d^2/dtheta^2),(d^2/drdtheta)]

    acts on a state vector u and its first and second derivatives in
    theta. If

    L[u] = 0

    then u(r,theta) is a solution to the PDE system.

    An elliptic PDE is not well-posed without the addition of boundary
    conditions, which select fro the particular solution.
