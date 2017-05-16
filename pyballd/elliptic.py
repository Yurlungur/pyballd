#!/usr/bin/env python2

"""elliptic.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-15 22:20:52 (jmiller)>

This is a module for pyballd. It contains the routines required for
solving elliptic systems.
"""

from __future__ import print_function
import numpy as np
import scipy as sp
from scipy import optimize
import domain

DEFAULT_ORDER_X = 24
DEFAULT_ORDER_THETA = 8
DEFAULT_R_H = 1.
DEFAULT_THETA_MIN = 0.
DEFAULT_THETA_MAX = np.pi
VERBOSE=True

def DEFAULT_BDRY_X_INNER(theta,u,d):
    out = u
    return out

def DEFAULT_BDRY_THETA_MIN(r,u,d):
    ushape = u.shape
    num_vars = np.prod(ushape[:-2])
    uflat = u.reshape(tuple([num_vars])+ushape[-2:])
    out = np.empty_like(uflat)
    for i in range(out.shape[0]):
        out[i] = d(uflat[i],0,1)
    out.reshape(ushape)
    return out

def DEFAULT_BDRY_THETA_MAX(r,u,d):
    ushape = u.shape
    num_vars = np.prod(ushape[:-2])
    uflat = u.reshape(tuple([num_vars])+ushape[-2:])
    out = np.empty_like(uflat)
    for i in range(out.shape[0]):
        out[i] = d(uflat[i],0,1)
    out.reshape(ushape)
    return out

def DEFAULT_INITIAL_GUESS(r,theta):
    out = (0.*theta).reshape(tuple([1])+theta.shape)
    return out

def pde_solve_once(residual,
                   r_h            = DEFAULT_R_H,
                   theta_min      = DEFAULT_THETA_MIN,
                   theta_max      = DEFAULT_THETA_MAX,
                   order_X        = DEFAULT_ORDER_X,
                   order_theta    = DEFAULT_ORDER_THETA,
                   cmp_type       = 'standard',
                   bdry_X_inner   = DEFAULT_BDRY_X_INNER,
                   bdry_theta_min = DEFAULT_BDRY_THETA_MIN,
                   bdry_theta_max = DEFAULT_BDRY_THETA_MAX,
                   initial_guess  = DEFAULT_INITIAL_GUESS,
                   f_tol          = 1e-8,
                   method         = 'hybr'
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

    f_tol          -- The tolerance on the residual for the
                      nonlinear vector root finder.

    method         -- The solver method to use. (Exposes scipy functionality.)
                      The default is MINPACK's hybrid method 'hybr'.
                      Options are
                      -- 'hybr'
                      -- 'lm'
                      -- 'broyden1'
                      -- 'broyden2'
                      -- 'anderson'
                      -- 'linearmixing'
                      -- 'diagbroyden'
                      -- 'excitingmixing'
                      -- 'krylov'
                      -- 'df-sane'
                      -- 'newton-krylov'
                      For small problems, 'hybr' is probably ideal.
                      For large problems, probably 'newton-krylov'
                      See:
                      https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.root.html
                      and
                      https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.newton_krylov.html

    Returns
    -------
    soln  -- The solution
    d     -- The discretization object associated with the solution.
             It can be used to differentiate, interpolate, and more.
    """
    ok_methods = ['hybr',
                  'lm',
                  'broyden1',
                  'broyden2',
                  'anderson',
                  'linearmixing',
                  'diagbroyden',
                  'excitingmixing',
                  'krylov',
                  'df-sane',
                  'newton-krylov']
    if method not in ok_methods:
        raise ValueError("Method must be one of {}".format(ok_methods))

    if VERBOSE:
        print("Welcome to the Pyballd Elliptic Solver")
        
    if cmp_type is 'standard':
        Stencil = domain.PyballdStencil
    elif cmp_type is 'BH':
        Stencil = domain.PyballdStencilBH
    else:
        raise ValueError("Invalid compactification type. "
                         +"Valid options are: 'standard','bh'")
    
    # define domain
    if VERBOSE:
        print("Generating pseudospectral derivatives")
    s = Stencil(order_X,r_h,
                order_theta,
                theta_min,theta_max)
    R,THETA = s.get_coords_2d()
    X,THETA = s.get_x2d()

    # initial guess
    if VERBOSE:
        print("Constructing initial guess")
    u0 = initial_guess(R,THETA)

    # define numerical residual
    if VERBOSE:
        print("Defining Residuals")
    if cmp_type is 'standard':
        def f(u):
            u = u.reshape(u0.shape)
            d = s.differentiate_wrt_R
            out = residual(R,THETA,u,d)
            out[...,0,:] = bdry_X_inner(THETA,u,d)[...,0,:]
            out[...,:,0] = bdry_theta_min(R,u,d)[...,:,0]
            out[...,:,-1] = bdry_theta_max(R,u,d)[...,:,-1]
            out[...,-1,:] = 0.
            return out.flatten()
    elif cmp_type is 'BH':
        def f(u):
            u = u.reshape(u0.shape)
            d = s.differentiate_wrt_R
            out = residual(R,THETA,u,d)
            out[...,0,:] = s.differentiate(u,1,0)[...,0,:]
            out[...,:,0] = bdry_theta_min(R,u,d)[...,:,0]
            out[...,:,-1] = bdry_theta_max(R,u,d)[...,:,-1]
            out[...,-1,:] = 0.
            return out.flatten()
    else:
        raise ValueError("Invalid compactification type. "
                         +"Valid options are: 'standard','bh'")

    # solve!
    if VERBOSE:
        print("Beginning solve")
        print("using method: {}".format(method))
    if method == 'newton-krylov':
        soln = optimize.newton_krylov(f,u0,f_tol=f_tol,verbose=VERBOSE)
        soln.reshape(u0.shape)
    else:
        soln = optimize.root(f,u0.flatten(),tol=f_tol,method=method)
        print(soln.message)
        if soln.success:
            items = dir(soln)
            if VERBOSE:
                print("solve succesfull!")
                print("Solve status is: {}".format(soln.status))
                if 'nfev' in items:
                    print("Solved in {} function evaluations".format(soln.nfev))
                if 'nit' in items:
                    print("Solved in {} iterations".format(soln.nit))
            soln = soln.x.reshape(u0.shape)
        else:
            print("Solve failed!")
            print(soln.status)
            print(soln.message)
            raise ValueError(soln.message)

    if VERBOSE:
        print("Solve complete.")

    return soln,s
