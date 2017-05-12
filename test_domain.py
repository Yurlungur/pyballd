#!/usr/bin/env python2

"""test_domain.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-12 18:29:13 (jmiller)>

Tests the domain module of pyballd.
"""

from __future__ import print_function
import numpy as np
from domain import PyballdStencil
import matplotlib as mpl
from matplotlib import pyplot as plt

r_h = 1.
THETA_ORDER = 20
THETA_MIN = 0
THETA_MAX = np.pi/2
f = lambda r,theta: np.cos(2*np.pi*r)*np.sin(theta)/r
dfdr = lambda r,theta: -np.sin(theta)*(np.cos(2*np.pi*r)/(r*r) + 2*np.pi*np.sin(2*np.pi*r)/r)

def test_func_and_derivative():
    X_order = 50
    s = PyballdStencil(X_order,r_h,
                       THETA_ORDER,
                       THETA_MIN,THETA_MAX)
    R,THETA = s.get_coords_2d()
    X,THETA = s.get_x2d()
    f_ana = f(R,THETA)
    dfdr_ana = dfdr(R,THETA)
    dfdr_ana[-1] = 0
    dfdr_num = s.differentiate_wrt_R(f_ana,1,0)
    dfdr_num[-1] = 0
    plt.plot(X[:,-1],s.dXdR[:,-1],lw=3)
    plt.xlabel(r'$(r-r_h)/(1+r-r_h)$',fontsize=16)
    plt.ylabel(r'$\partial_r X$',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/domain_dXdr'+postfix,
                    bbox_inches='tight')
    plt.clf()

    plt.plot(X[:,-1],f_ana[:,-1],'b-',lw=3)
    plt.xlabel(r'$(r-r_h)/(1+r-r_h)$',fontsize=16)
    plt.ylabel(r'$\sin(\theta)/r$',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/domain_test_function'+postfix,
                    bbox_inches='tight')
    plt.clf()

    plt.plot(X[:,-1],dfdr_ana[:,-1],'b-',lw=3,label='analytic')
    plt.plot(X[:,-1],dfdr_num[:,-1],'bo',lw=3,label='numerical')
    plt.xlabel(r'$(r-r_h)/(1+r-r_h)$',fontsize=16)
    plt.ylabel(r'$\partial_r \sin(\theta)/r$',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/deriv_domain_test_function'+postfix,
                    bbox_inches='tight')
    plt.clf()

def test_pointwise_errors():
    X_orders = [4*(i+1) for i in range(3)]
    for o in X_orders:
        s = PyballdStencil(o,r_h,
                           THETA_ORDER,
                           THETA_MIN,THETA_MAX)
        R,THETA = s.get_coords_2d()
        X,THETA = s.get_x2d()
        f_ana = f(R,THETA)
        dfdr_ana = dfdr(R,THETA)
        dfdr_ana[-1] = 0
        dfdr_num = s.differentiate_wrt_R(f_ana,1,0)
        dfdr_num[-1] = 0
        delta = dfdr_num - dfdr_ana
        plt.plot(X[:,-1],delta[:,-1],lw=3)
    plt.xlabel(r'$(r-r_h)/(1+r-r_h)$',fontsize=16)
    plt.ylabel('error',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/domain_pointwise_errors'+postfix,
                    bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    test_func_and_derivative()
    test_pointwise_errors()
