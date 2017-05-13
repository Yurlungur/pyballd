#!/usr/bin/env python2

"""domain.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-13 10:59:09 (jmiller)>

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
EPSILON_SML=1e-10
EPSILON_BIG=1e10

    
class PyballdStencil(PseudoSpectralStencil2D):
    """A derivative stencil explicitly designed for axisymmetric
    domains. Automatically performs appropriate compactification.

    This version uses
    x = r - r_h,
    thus shifting the domain of x to [0,infinity)
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
        self.dRdX = self.get_drdX(self.X)
        self.drdX = self.get_drdX(self.x)
        self.dXdR = self.get_dXdr(self.X)
        self.dXdr = self.get_dXdr(self.x)
        self.d2XdR2 = self.get_d2Xdr2(self.X)
        self.d2Xdr2 = self.get_d2Xdr2(self.x)
        self.d2RdX2 = self.get_d2rdX2(self.X)
        self.d2rdX2 = self.get_d2rdX2(self.x)
        self.theta = self.y
        self.THETA = self.Y

    def differentiate_wrt_R(self,grid_func,order_X,order_theta):
        "Differentiate a grid function on our compactified domain"

        assert type(order_X) is int
        assert type(order_theta) is int
        assert order_X >= 0
        assert order_theta >= 0

        if order_X > 2:
            raise NotImplementedError("Derivatives in X higher than order 2"
                                      +" not implemented.")
        if order_X == 1:
            df = self.differentiate(grid_func,1,order_theta)
            df *= self.dXdR
            return df
        if order_X == 2:
            d2fdX2 = self.differentiate(grid_func,2,order_theta)
            dfdX = self.differentiate(grid_func,1,order_theta)
            d2fdR2 = (self.dXdR**2)*d2fdX2 + self.d2XdR2*dfdX
            return d2fdR2
        # else (if order_X == 0)
        return self.differentiate(grid_func,0,order_theta)

    def inner_product_on_sphere(self,gf1,gf2):
        "Inner product using spherical coordinates"
        factors = [s.get_scale_factor() for s in self.stencils]
        factor = np.prod(factors)
        # exclude last point because it diverges
        integrand = (gf1*self.weights2D*gf2
                     *self.dRdX
                     *(self.R**2)*np.sin(self.THETA))[:-1,0]
        integral_unit_cell = np.sum(integrand)
        integral_physical = integral_unit_cell*factor*(2*np.pi)
        return integral_physical

    def get_drdX(self,X):
        dXdr = self.get_dXdr(X)
        with np.errstate(invalid='ignore'):
            drdX = 1./dXdr
        return drdX

    def get_dXdr(self,X):
        dXdr = (X-1)**2
        return dXdr

    def get_d2rdX2(self,X):
        d2Xdr2 = self.get_d2Xdr2(X)
        with np.errstate(invalid='ignore'):
            d2rdX2 = 1./d2Xdr2
        return d2rdX2

    def get_d2Xdr2(self,X):
        d2Xdr2 = 2*(X-1)**3
        return d2Xdr2

    def get_x_from_r(self,r):
        x = r - self.r_h
        return x

    def get_X_from_x(self,x):
        X = x/(1.0+x)
        return X

    def get_X_from_r(self,r):
        x = self.get_x_from_r(r)
        X = self.get_X_from_x(self,x)
        return X

    def get_x_from_X(self,X):
        with np.errstate(invalid='ignore'):
            x = (1.0 - X)**(-1) - 1
        return x

    def get_r_from_x(self,x):
        r = x + self.r_h
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

    
class PyballdStencilBH(PyballdStencil):
    """A derivative stencil explicitly designed for axisymmetric
    black holes. Automatically performs appropriate compactification.

    This version uses
    x = sqrt(r^2 - r_h^2),
    thus shifting the domain of x to [0,infinity)
    """

    def get_dXdr(self,X):
        num = ((X-1)**2)*np.sqrt(X**2+(self.r_h**2)*((X-1)**2))
        denom = X
        dXdr = num/denom
        return dXdr

    def get_d2Xdr2(self,X):
        num = ((X-1)**3)*(2*(X**3)+(self.r_h**2)*((X-1)**2)*(1+2*X))
        denom = X**3
        d2Xdr2 = num/denom
        return d2Xdr2

    def get_x_from_r(self,r):
        x = np.sqrt(r**2 - self.r_h**2)
        return x

    def get_r_from_x(self,x):
        r = np.sqrt(x**2 + self.r_h**2)
        return r
