#!/usr/bin/env python2

"""test_domain.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-13 16:24:11 (jmiller)>

Tests the domain module of pyballd.
"""

from __future__ import print_function
import numpy as np
import pyballd
from pyballd.domain import PyballdStencil
import matplotlib as mpl
from matplotlib import pyplot as plt

r_h = 1.
THETA_MIN = 0
THETA_MAX = np.pi/2
def f(r,theta):
    out = np.sin(theta)*r*np.exp(-r/2.)
    out[-1] = 0
    return out
def dfdr(r,theta):
    out = np.sin(theta)*(np.exp(-r/2.) - (1./2.)*r*np.exp(-r/2.))
    out[-1] = 0
    return out

def test_func_and_derivative():
    THETA_ORDER = 50
    X_order = 100
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
    plt.ylabel(r'$r e^{-r/2}\sin(\theta)$',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/domain_test_function'+postfix,
                    bbox_inches='tight')
    plt.clf()

    mx,mz = np.sin(THETA)*R,np.cos(THETA)*R
    plt.pcolor(mx[:-20,:],mz[:-20,:],f_ana[:-20,:])
    #(r'$(r-r_h)/(1+r-r_h)$',fontsize=16)
    #plt.ylabel(r'$\theta$',fontsize=16)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    cb = plt.colorbar()
    cb.set_label(label=r'$r e^{-r/2}\sin(\theta)$',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/domain_test_function_2d'+postfix,
                    bbox_inches='tight')
    plt.clf()
    
    plt.plot(X[:,-1],dfdr_ana[:,-1],'b-',lw=3,label='analytic')
    plt.plot(X[:,-1],dfdr_num[:,-1],'bo',lw=3,label='numerical')
    plt.xlabel(r'$(r-r_h)/(1+r-r_h)$',fontsize=16)
    plt.ylabel(r'$\partial_r [r e^{-r/2}\sin(\theta)]$',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/deriv_domain_test_function'+postfix,
                    bbox_inches='tight')
    plt.clf()

    plt.pcolor(mx[:-20,:],mz[:-20,:],dfdr_ana[:-20,:])
    # plt.xlabel(r'$(r-r_h)/(1+r-r_h)$',fontsize=16)
    # plt.ylabel(r'$\theta$',fontsize=16)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    cb = plt.colorbar()
    cb.set_label(label=r'$\partial_r [r e^{-r/2}\sin(\theta)]$',
                 fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/deriv_domain_test_function_2d'+postfix,
                    bbox_inches='tight')
    plt.clf()

def test_errors():
    THETA_ORDER=20
    X_orders = [2+(i) for i in range(117)]
    l1_errors = [None for o in X_orders]
    print("iteration, order")
    for i,o in enumerate(X_orders):
        #print(i,o)
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
        #plt.plot(X[:,-1],delta[:,-1],lw=3)
        l1_errors[i] = np.max(delta[1:])
    # plt.xlabel(r'$(r-r_h)/(1+r-r_h)$',fontsize=16)
    # plt.ylabel('error',fontsize=16)
    # for postfix in ['.png','.pdf']:
    #     plt.savefig('figs/domain_pointwise_errors'+postfix,
    #                 bbox_inches='tight')
    # plt.clf()

    plt.semilogy(X_orders,l1_errors,'b-',lw=3,ms=12)
    plt.xlabel('order',fontsize=16)
    plt.ylabel('max(error)',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/domain_l1_errors'+postfix,
                    bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    test_func_and_derivative()
    test_errors()
