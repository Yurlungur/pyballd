#!/usr/bin/env python2

"""poisson.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-17 21:18:36 (jmiller)>

This is an example script that solves the Poisson equation using
pyballd.
"""

from __future__ import print_function
import pyballd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

r_h = 1.0
k = 4
a = 2
order_X = 24
order_theta = 6
exclude_last=1
theta_max = np.pi/2
rmax = 1.5

def residual(r,theta,u,d):
    u = u[0]
    out = (2*np.sin(theta)*r*d(u,1,0)
           + r*r*np.sin(theta)*d(u,2,0)
           + np.cos(theta)*d(u,0,1)
           + np.sin(theta)*d(u,0,2))
    out = out.reshape(tuple([1]) + out.shape)
    return out

def bdry_X_inner(theta,u,d):
    u = u[0]
    out = u - a*np.cos(k*theta)
    out = out.reshape(tuple([1]) + out.shape)
    return out

def initial_guess(r,theta):
    out = 1./r
    out = out.reshape(tuple([1]) + out.shape)
    return out

if __name__ == "__main__":
    SOLN,s = pyballd.pde_solve_once(residual,
                                    r_h = r_h,
                                    order_X = order_X,
                                    order_theta = order_theta,
                                    bdry_X_inner = bdry_X_inner,
                                    initial_guess = initial_guess,
                                    theta_min = 0,
                                    theta_max = theta_max,
                                    method = 'hybr',
                                    f_tol=1e-10)
    

    SOLN = SOLN[0]
    r = np.linspace(r_h,2*rmax,200)
    theta = np.linspace(0,theta_max,200)
    R,THETA = np.meshgrid(r,theta,indexing='ij')
    mx,mz = R*np.sin(THETA), R*np.cos(THETA)
    s_interpolator = s.get_interpolator_of_r(SOLN)
    soln_interp = s_interpolator(R,THETA)
    plt.pcolor(mx,
               mz,
               soln_interp)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('z',fontsize=16)
    cb =plt.colorbar()
    cb.set_label(label='solution to Poisson Eqn',
                 fontsize=16)
    plt.axis('scaled')
    plt.xlim(0,rmax)
    plt.ylim(0,rmax)
    #plt.ylim(-rmax/,rmax/2)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/poisson_solution'+postfix,
                bbox_inches='tight')
    plt.clf()

    R,THETA = s.get_coords_2d()
    X,THETA = s.get_x2d()
    np.savetxt('data/poisson_solution.txt',SOLN)
    np.savetxt('data/poisson_X.txt',X)
    np.savetxt('data/poisson_R.txt',R)
    np.savetxt('data/poisson_THETA.txt',THETA)
