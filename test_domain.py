#!/usr/bin/env python2

"""test_domain.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-12 08:41:08 (jmiller)>

Tests the domain module of pyballd.
"""

from __future__ import print_function
import numpy as np
from domain import PyballdStencil

r_h = 1.
THETA_ORDER = 20
THETA_MIN = 0
THETA_MAX = np.pi/2
f = lambda r,theta: np.sin(theta)/r
dfdr = lambda r,theta: -np.sin(theta)/(r*r)

def test_pointwis_error():
    X_orders = [8*(i+1) for i in range(3)]
    for o in orders:
        s = PyballdStencil(o,r_h,
                           theta_order,
                           THETA_MIN,THETA_MAX)
        R,THETA = s.get_coords_2d()
        f_ana = f(R,THETA)
        
    
