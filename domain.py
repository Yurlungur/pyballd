#!/usr/bin/env python2

"""domain.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-11 19:19:14 (jmiller)>

A component of the pyballd library. This module defines the domain and
coordinate system used for pyballd:

The core domain one wants for axisymmetry is:
(r,theta) in [r_h,infinity) x [0,pi]
for some r_h > 0, which is a minimum radius. (It can be zero.)

However we compactify our domain via the following definitions:
x = sqrt(r^2 - r_h^2)
X = X/(1+X)

Then our PDE system acquires a few Jacobian terms of the form
(d/dr) = (dX/dr) (d/dX)
where
(dX/dr) = 1/(dr/dX)
which we calculate numerically.
"""

from __future__ import print_function
import numpy as np
from orthopoly import PseudoSpectralStencil2D

class PyballdStencil(PseudoSpectralStencil2D):
    """A derivative stencil explicitly designed for axisymmetric
    domains. Automatically performs appropriate compactification.
    """
    X_min = 0
    X_max = 1

    def __init__(self,
                 order_X,r_h,
                 order_theta,theta_min,theta_max):
        """Constructor.

        Parameters
        ----------
        order_X     -- polynomial order in X direction
        r_h         -- physical minimum radius (uncompactified coordinates)
        order_theta -- polynomial order in theta direction
        theta_min   -- minimum longitudinal value. Should be no less than 0.
        theta_max   -- maximum longitudinal value. Should be no greater than pi.
        """
        self.order_X = order_X
        self.order_theta = order_theta
        self.r_h = r_h
        self.theta_min = theta_min
        self.theta_max = theta_max
        super(PyballdStencil,self).__init__(order_X,
                                            self.X_min,self.X_max,
                                            order_theta,
                                            theta_min,theta_max)
        self.r = self.get_r_from_X(self.x)
        self.R = self.get_r_from_X(self.X)
        self.dRdX = self.differentiate(self.R,1,0)
        self.drdX = self.dRdx[:,0]
        self.dXdR = 1./self.dRdX
        self.dXdr = self.dXdR[:,0]
        self.theta = self.y
        self.THETA = self.Y

    def differentiate_wrt_X(self,grid_func,order_X,order_theta):
        "Differentiate a grid function on our compactified domain"

        assert type(order_X) is int
        assert type(order_theta) is int
        assert order_X >= 0
        assert order_theta >= 0

        if order_X > 0:
            df = np.empty_like(grid_func)
            for j in range(df.shape[1]):
                df[:,j] = self.stencil_x.differentiate(grid_func[:,j],order_X)
            df *= self.dXdR
            return self.differentiate(df,0,ordery)
        
        return self.differentiate(grid_func,0,order_theta)

    def inner_product_on_sphere(self,gf1,gf2):
        "Inner product using spherical coordinates"
        factors = [s.get_scale_factor() for s in self.stencils]
        factor = np.prod(factors)
        # exclude last point because it diverges
        integrand = (gf1*self.weights2D*gf2*(self.R**2)*np.sin(self.THETA))[:-1,0]
        integral_unit_cell = np.sum(integrand)
        integral_physical = integral_unit_cell*factor*(2*np.pi)
        return integral_physical

    def get_x_from_r(self,r):
        x = np.sqrt(r**2 - self.r_h**2)
        return x

    def get_X_from_x(self,x):
        X = x/(1.0+x)
        return X

    def get_X_from_r(self,r):
        x = self.get_x_from_r(r)
        X = self.get_X_from_x(self,x)
        return X

    def get_x_from_X(self,X):
        x = (1.0 - X)**(-1) - 1
        return x

    def get_r_from_x(self,x):
        r = np.sqrt(x**2 + r_h**2)
        return r

    def get_r_from_X(self,X):
        x = self.get_x_from_X(X)
        r = self.get_r_from_x(x)
        return r

    def get_coords_1d(self,axis=0):
        if axis == 0:
            return self.r
        if axis == 1:
            return self.theta
        raise ValueError("Invalid axis")

    def get_coords_2d(self):
        return self.R,self.THETA

    
