#!/usr/bin/env python2

"""elliptic.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-13 14:06:43 (jmiller)>

This is a module for pyballd. It contains the routines required for
solving elliptic systems.
"""

from enum import Enum
import numpy as np
import scipy as sp

DEFAULT_ORDER_X = 100
DEFAULT_ORDER_THETA = 24
DEFAULT_R_H = 1.
DEFAULT_THETA_MIN = 0.
DEFAULT_THETA_MAX = np.pi

CmpType = Enum('standard','BH')

def pde_solve_once(residual,
                   r_h = DEFAULT_R_H
                   theta_min = DEFAULT_THETA_MIN,
                   theta_max = DEFAULT_THETA_MAX,
                   order_X = DEFAULT_ORDER_X,
                   order_theta = DEFAULT_ORDER_THETA,
                   compactification_type = CmpType.standard,
                   bdry_X_inner = None,
                   bdry_theta_min = None,
                   bdry_theta_max = None,
                   initial_guess = None
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
    conditions, which select for the particular solution. At infinity
    (X=1),
    """
