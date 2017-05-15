#!/usr/bin/env python2

"""
test_orthopoly.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-15 19:35:25 (jmiller)>

Tests the orthopoly module.
"""

from __future__ import print_function
import pyballd
from pyballd.orthopoly import PseudoSpectralStencil2D
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
mpl.rcParams.update({'font.size':12})

XMIN,XMAX = -np.pi/2.,np.pi/2.
YMIN,YMAX = 0,2*np.pi
KX = 1
KY = 2
X_OVER_Y = 2

f = lambda x,y: np.cos(KX*x)*np.sin(KY*y)
dfdx = lambda x,y: -KX*np.sin(KX*x)*np.sin(KY*y)
dfdx2 = lambda x,y: -KX*KX*np.cos(KX*x)*np.sin(KY*y)
dfdy = lambda x,y: KY*np.cos(KX*x)*np.cos(KY*y)
dfdy2 = lambda x,y: -KY*KY*np.cos(KX*x)*np.sin(KY*y)
dfdxdy = lambda x,y: -KX*KY*np.sin(KX*x)*np.cos(KY*y)
g = lambda x,y: dfdx2(x,y) + dfdy2(x,y) + dfdxdy(x,y)

def test_derivatives_at_order(ordery):
    orderx = X_OVER_Y*ordery
    s = PseudoSpectralStencil2D(orderx,XMIN,XMAX,
                                ordery,YMIN,YMAX)
    X,Y = s.get_x2d()
    f_ana = f(X,Y)
    g_ana = g(X,Y)
    g_num = (s.differentiate(f_ana,2,0)
             + s.differentiate(f_ana,0,2)
             + s.differentiate(f_ana,1,1))
    delta_g = g_num - g_ana
    norm2dg = s.norm2(delta_g)
    return norm2dg

def plot_test_function(orderx,ordery):
    s = PseudoSpectralStencil2D(orderx,XMIN,XMAX,
                                ordery,YMIN,YMAX)
    X,Y = s.get_x2d()
    f_ana = f(X,Y)
    plt.pcolor(X,Y,f_ana)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.xlim(XMIN,XMAX)
    plt.ylim(YMIN,YMAX)
    cb = plt.colorbar()
    cb.set_label(label=r'$\cos(x)\sin(2 y)$',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/test_function'+postfix,
                    bbox_inches='tight')
    plt.clf()

def test_derivatives():
    orders = [4+(2*i) for i in range(12)]
    errors = [test_derivatives_at_order(o) for o in orders]
    plt.semilogy(orders,errors,'bo-',lw=2,ms=12)
    plt.xlabel('order in y-direction',fontsize=16)
    plt.ylabel(r'$|E|_2$',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/orthopoly_errors'+postfix,
                    bbox_inches='tight')
    plt.clf()

def test_interp_at_order(ordery):
    orderx = X_OVER_Y*ordery
    s = PseudoSpectralStencil2D(orderx,XMIN,XMAX,
                                ordery,YMIN,YMAX)
    Xc,Yc = s.get_x2d()
    x = np.linspace(XMIN,XMAX,100)
    y = np.linspace(YMIN,YMAX,100)
    Xf,Yf = np.meshgrid(x,y,indexing='ij')
    f_coarse = f(Xc,Yc)
    f_fine = f(Xf,Yf)
    f_interpolator = s.to_continuum(f_coarse)
    f_num = f_interpolator(Xf,Yf)
    delta = f_num - f_fine
    return np.max(np.abs(delta))

def plot_interpolation(orderx,ordery):
    s = PseudoSpectralStencil2D(orderx,XMIN,XMAX,
                                ordery,YMIN,YMAX)
    Xc,Yc = s.get_x2d()
    x = np.linspace(XMIN,XMAX,100)
    y = np.linspace(YMIN,YMAX,100)
    Xf,Yf = np.meshgrid(x,y,indexing='ij')
    f_coarse = f(Xc,Yc)
    f_interpolator = s.to_continuum(f_coarse)
    f_num = f_interpolator(Xf,Yf)
    plt.pcolor(Xf,Yf,f_num)
    cb = plt.colorbar()
    cb.set_label('interpolated function',fontsize=16)
    plt.xlabel('x')
    plt.ylabel('y')
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/orthopoly_interpolated_function'+postfix,
                    bbox_inches='tight')
    plt.clf()

def test_interpolation():
    xfine = np.linspace(XMIN,XMAX,100)
    yfine = np.linspace(YMIN,YMAX,100)
    orders = [4+(2*i) for i in range(12)]
    errors = [test_interp_at_order(o) for o in orders]
    plt.semilogy(orders,errors,'bo-',lw=2,ms=12)
    plt.xlabel('order in y-direction',fontsize=16)
    plt.ylabel('max(interpolation error)',fontsize=16)
    for postfix in ['.png','.pdf']:
        plt.savefig('figs/orthopoly_interp_errors'+postfix,
                    bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    plot_test_function(80,160)
    test_derivatives()
    plot_interpolation(10,20)
    test_interpolation()

