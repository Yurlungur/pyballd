#!/usr/bin/env python2

"""elliptic.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-13 14:58:49 (jmiller)>

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
                   cmp_type = CmpType.standard,
                   bdry_X_inner = None,
                   bdry_theta_min = None,
                   bdry_theta_max = None,
                   initial_guess = None
                   ):
    """Solves the elliptic pde system defined by residual.
    See the README for more details.

    Parameters
    ----------
    residual       -- The residual defining the PDE system.
                      should be a function with prototype
                      R(r,theta,u,d)
                      where d defines derivative operations.
                      It is a function with prototype
                      d(u,order_r,order_theta)
                      so that dudr = d(u,1,0)
                              dudtheta = d(u,0,1)
                              d2udr2 = d(u,2,0)
                              ...and so on
                   
    r_h            -- The "physical" radius of the x = 0 coordinate
                      defaults to 1.
                   
    theta_min      -- The minimum value for theta. Defaults to 0.
                   
    theta_max      -- The maximum value for theta. Defaults to pi.
                   
    order_X        -- The number of Legendre polynomials in the
                      r (or X) direction. Defaults to 100
                   
    order_theta    -- The number of Legende polynomials in the theta
                      direction. Default is 24.
                   
    cmp_type       -- The type of compactification used at the inner
                      boundary. The options are:
                      'standard' -- In this case, x = r -r_h
                      'BH'       -- In this case, x = sqrt(r^2-r_h^2)
                      Then the full compactification maps
                      X = x/(1+x)
                   
    bdry_x_inner   -- The boundary condition on the r=r_h boundary.
                      It is defined in a way similar to residual.
                      It should be a function with prototype
                      B(theta,u,d)
                      where d is the same derivate defined above.
                      The solver attempt to find u such that
                      B(theta,u,d) = 0
                      Note that if cmp_type = 'standard',
                      then bdry_x_inner does nothing
                      as von-neuman boundary conditions
                      are explicitly enforced as a regularity condition.
                      If bdry_x_inner is not set, then
                      Dirichlet conditions set the inner boundary to zero.

    bdry_theta_min -- As bdry_x_inner, but the residual is
                      B(r,u,d)
                      and it is for the minimum theta boundary.
                      Default is Von-Neumann boundary conditions
                      so that the derivative vanishes at theta_min

    bdry_theta_max -- As bdry_x_inner, but the residual is
                      B(r,u,d)
                      and it is for the maximum theta boundary.
                      Default is Von-Neumann boundary conditions
                      so that the derivative vanishes at theta_max

    initial_guess  -- Initial guess for the functional form of the solution
                      should be a function of the form
                      u(r,theta)
                      u(r=infinity,theta) will automatically be set to zero.
                      However, the initial guess should be compatible with
                      this square integrability condition.
    """
