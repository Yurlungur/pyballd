#!/usr/bin/env python2

"""poisson.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-06-05 14:18:24 (jmiller)>

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
order_X = 36
order_theta = 12
exclude_last=1
theta_max = np.pi/2
rmax = 1.5

def residual(r,theta,u,d):
    out = np.empty_like(u)
    out[0] = (2*np.sin(theta)*r*d(u[0],1,0)
              + r*r*np.sin(theta)*d(u[0],2,0)
              + np.cos(theta)*d(u[0],0,1)
              + np.sin(theta)*d(u[1],0,2))
    out[1] = (2*np.sin(theta)*r*d(u[1],1,0)
              + r*r*np.sin(theta)*d(u[1],2,0)
              + np.cos(theta)*d(u[1],0,1)
              + np.sin(theta)*d(u[1],0,2))
    return out

def bdry_X_inner(theta,u,d):
    out = u - np.array([a*np.cos(k*theta),
                         a*np.cos((k)*theta)])
    return out

def initial_guess(r,theta):
    out = np.array([1./r,1./r])
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
                                    f_tol=1e-13)


    SOLN = SOLN[1]
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
        plt.savefig('figs/poisson_vec_solution'+postfix,
                    bbox_inches='tight')
        plt.clf()

    R,THETA = s.get_coords_2d()
    X,THETA = s.get_x2d()
    np.savetxt('data/poisson_vec_solution.txt',SOLN)
    np.savetxt('data/poisson_vec_X.txt',X)
    np.savetxt('data/poisson_vec_R.txt',R)
    np.savetxt('data/poisson_vec_THETA.txt',THETA)