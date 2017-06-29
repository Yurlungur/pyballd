#!/usr/bin/env python2

"""test_domain.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-06-28 20:27:13 (jmiller)>

Tests the domain module of pyballd.
"""

from __future__ import print_function
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import pyballd
from pyballd.domain import PyballdDiscretization

r_h = 1.
THETA_MIN = 0
THETA_MAX = np.pi/2
K=1
TEST_FUNC_NAME=r'$\sin(\theta)r e^{-r/2}$'
DR_TEST_FUNC_NAME=r'$\partial_r [\sin(\theta)re^{-r/2}]$'
DR_DTHETA_TEST_FUNC_NAME=r'$\partial_r\partial_\theta [\sin(\theta)r e^{-r/2}]$'
USE_FIGS_DIR=False

def f(r,theta):
    out = np.sin(theta)*r*np.exp(-r/2.)
    out[-1] = 0
    return out
def dfdr(r,theta):
    out = np.sin(theta)*(np.exp(-r/2.) - (1./2.)*r*np.exp(-r/2.))
    out[-1] = 0
    return out

def dfdrdtheta(r,theta):
    out = np.cos(theta)*(np.exp(-r/2.) - (1./2.)*r*np.exp(-r/2.))
    out[-1] = 0
    return out

def test_func_and_derivative():
    THETA_ORDER = 50
    X_order = 100
    s = PyballdDiscretization(X_order,r_h,
                              THETA_ORDER,
                              THETA_MIN,THETA_MAX,
                              L = 3*r_h)
    R,THETA = s.get_coords_2d()
    X,THETA = s.get_x2d()
    f_ana = f(R,THETA)
    dfdr_ana = dfdr(R,THETA)
    dfdr_ana[-1] = 0
    dfdrdtheta_ana = dfdrdtheta(R,THETA)
    dfdrdtheta_ana[-1] = 0
    dfdr_num = s.differentiate_wrt_R(f_ana,1,0)
    dfdrdtheta_num = s.differentiate_wrt_R(f_ana,1,1)

    plt.plot(X[:,-1],s.dXdR[:,-1],lw=3)
    plt.xlabel(r'$X$',fontsize=16)
    plt.ylabel(r'$\partial_r X$',fontsize=16)
    for postfix in ['.png','.pdf']:
        name = 'domain_dXdr'+postfix
        if USE_FIGS_DIR:
            name = 'figs/' + name
        plt.savefig(name,
                    bbox_inches='tight')
    plt.clf()

    plt.plot(X[:,-1],f_ana[:,-1],'b-',lw=3)
    plt.xlabel(r'$X$',fontsize=16)
    plt.ylabel(TEST_FUNC_NAME,
               fontsize=16)
    for postfix in ['.png','.pdf']:
        name = 'domain_test_function_exp'+postfix
        if USE_FIGS_DIR:
            name = 'figs/' + name
        plt.savefig(name,
                    bbox_inches='tight')
    plt.clf()

    mx,mz = np.sin(THETA)*R,np.cos(THETA)*R
    plt.pcolor(mx[:-1,:],mz[:-1,:],f_ana[:-1,:])
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.xlim(0,5)
    plt.ylim(0,5)
    cb = plt.colorbar()
    cb.set_label(label=TEST_FUNC_NAME,fontsize=16)
    for postfix in ['.png','.pdf']:
        name = 'domain_test_function_exp_2d'+postfix
        if USE_FIGS_DIR:
            name = 'figs/' + name
        plt.savefig(name,
                    bbox_inches='tight')
    plt.clf()
    
    plt.plot(X[:,-1],dfdr_ana[:,-1],'b-',lw=3,label='analytic')
    plt.plot(X[:,-1],dfdr_num[:,-1],'ro',lw=3,label='numerical')
    plt.xlabel(r'$X$',fontsize=16)
    plt.ylabel(DR_TEST_FUNC_NAME,fontsize=16)
    plt.legend()
    for postfix in ['.png','.pdf']:
        name = 'deriv_domain_test_function_exp'+postfix
        if USE_FIGS_DIR:
            name = 'figs/' + name
        plt.savefig(name,
                    bbox_inches='tight')
    plt.clf()

    plt.pcolor(mx[:-1,:],mz[:-1,:],dfdrdtheta_ana[:-1,:])
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    cb = plt.colorbar()
    plt.xlim(0,5)
    plt.ylim(0,5)
    cb.set_label(label=DR_DTHETA_TEST_FUNC_NAME,
                 fontsize=16)
    for postfix in ['.png','.pdf']:
        name = 'deriv_domain_test_function_exp_2d'+postfix
        if USE_FIGS_DIR:
            name = 'figs/' + name
        plt.savefig(name,
                    bbox_inches='tight')
    plt.clf()

def test_errors():
    orders = [4*(i+1) for i in range(10)]
    l1_errors = [None for o in orders]
    l2_errors = [None for o in orders]
    for i,o in enumerate(orders):
        s = PyballdDiscretization(o,r_h,
                                  o,
                                  THETA_MIN,
                                  THETA_MAX,
                                  L=25*r_h)
        R,THETA = s.get_coords_2d()
        X,THETA = s.get_x2d()
        f_ana = f(R,THETA)
        dfdr_ana = dfdr(R,THETA)
        dfdr_ana[-1] = 0
        dfdr_num = s.differentiate_wrt_R(f_ana,1,0)
        dfdr_num[-1] = 0
        delta = dfdr_num - dfdr_ana
        if 2 <= i <= 4:
            plt.plot(X[:,-1],delta[:,-1],lw=3,
                     label="order = {}".format(o))
        l1_errors[i] = np.max(np.abs(delta[1:]))
        l2_errors[i] = s.l2_norm_to_infty(delta)
    plt.xlabel(r'$X$',fontsize=16)
    plt.ylabel('error',fontsize=16)
    plt.legend()
    for postfix in ['.png','.pdf']:
        name = 'domain_pointwise_errors_exp'+postfix
        if USE_FIGS_DIR:
            name = 'figs/' + name
        plt.savefig(name,
                    bbox_inches='tight')
    plt.clf()

    plt.semilogy(orders,l1_errors,'bo--',lw=3,ms=12)
    plt.xlabel('order',fontsize=16)
    plt.ylabel(r'|error|$_\infty$',fontsize=16)
    for postfix in ['.png','.pdf']:
        name = 'domain_l1_errors_exp'+postfix
        if USE_FIGS_DIR:
            name = 'figs/' + name
        plt.savefig(name,
                    bbox_inches='tight')
    plt.clf()

    plt.loglog(orders,l1_errors,'bo--',lw=3,ms=12)
    plt.xlabel('order',fontsize=16)
    plt.ylabel(r'|error|$_\infty$',fontsize=16)
    for postfix in ['.png','.pdf']:
        name = 'domain_l1_errors_loglog_exp'+postfix
        if USE_FIGS_DIR:
            name = 'figs/' + name
        plt.savefig(name,
                    bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    test_func_and_derivative()
    test_errors()
