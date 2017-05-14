#!/usr/bin/env python2

"""poisson.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-14 01:10:10 (jmiller)>

This is an example script that solves the Poisson equation using
pyballd.
"""

from __future__ import print_function
import pyballd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

r_h = 1.0
k = 2
a = 4
order_X = 70
order_theta = 50
exclude_last=20

def residual(r,theta,u,d):
    out = (2*r*d(u,1,0)
           + r*r*d(u,2,0)
           + np.cos(theta)*d(u,0,1)
           + np.sin(theta)*d(u,0,2))
    return out

def bdry_X_inner(theta,u,d):
    out = u - a*np.cos(k*theta)
    return out

def initial_guess(r,theta):
    out = a*np.cos(k*theta)/r
    return out

R,X,THETA,SOLN = pyballd.pde_solve_once(residual,
                                        r_h = r_h,
                                        order_X = order_X,
                                        order_theta = order_theta,
                                        bdry_X_inner = bdry_X_inner,
                                        initial_guess = initial_guess,
                                        f_tol=6e-6)

mx,mz = R*np.sin(THETA), R*np.cos(THETA)
plt.pcolor(mx[:-exclude_last,:],
           mz[:-exclude_last,:],
           SOLN[:-exclude_last,:])
plt.xlabel('x',fontsize=16)
plt.ylabel('z',fontsize=16)
cb =plt.colorbar()
cb.set_label(label='solution to Poisson Eqn',
             fontsize=16)
plt.xlim(0,2.5)
plt.ylim(-2.5,2.5)
for postfix in ['.png','.pdf']:
    plt.savefig('figs/poisson_solution'+postfix,
                bbox_inches='tight')
plt.clf()

np.savetxt('data/poisson_solution.txt',SOLN)
np.savetxt('data/poisson_X.txt',X)
np.savetxt('data/poisson_R.txt',R)
np.savetxt('data/poisson_THETA.txt',THETA)
