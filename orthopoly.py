#!/usr/bin/env python2

"""
orthopoly.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-11 10:37:26 (jmiller)>

A module for orthogonal polynomials for pseudospectral methods in Python
"""



# ======================================================================
# imports
# ======================================================================
import numpy as np
from numpy import polynomial
from numpy import linalg
# ======================================================================



# ======================================================================
# Global constants
# ======================================================================
LOCAL_XMIN = -1 # Maximum and min values of reference cell
LOCAL_XMAX = 1
poly = polynomial.legendre.Legendre  # A class for orthogonal polynomials
# method for generating 2D Vandermonde matrices
pvander2d = poynomial.legendre.legvander2d
# a method for evaluating a 2D poynomial 
pgrid2d = poynomial.legendre.leggrid2d
# ======================================================================



# ======================================================================
# Nodal and Modal Details
# ======================================================================
def get_quadrature_points(order):
    """
    Returns the quadrature points for Gauss-Lobatto quadrature
    as a function of the order of the polynomial we want to
    represent.
    See: https://en.wikipedia.org/wiki/Gaussian_quadrature
    """
    return np.sort(np.concatenate((np.array([-1,1]),
                                   poly.basis(order).deriv().roots())))

def get_integration_weights(order,nodes=None):
    """
    Returns the integration weights for Gauss-Lobatto quadrature
    as a function of the order of the polynomial we want to
    represent.
    See: https://en.wikipedia.org/wiki/Gaussian_quadrature
    """
    if nodes == None:
        nodes=get_quadrature_points(order)
    interior_weights = 2/((order+1)*order*poly.basis(order)(nodes[1:-1])**2)
    boundary_weights = np.array([1-0.5*np.sum(interior_weights)])
    weights = np.concatenate((boundary_weights,
                              interior_weights,
                              boundary_weights))
    return weights

def get_vandermonde_matrices(order,nodes=None):
    """
    Returns the Vandermonde fast-Fourier transform matrices s2c and c2s,
    which convert spectral coefficients to configuration space coefficients
    and vice-versa respectively. Requires the order of the element/method
    as input.
    """
    if nodes == None:
        nodes = get_quadrature_points(order)
    s2c = np.zeros((order+1,order+1),dtype=float)
    for i in range(order+1):
        for j in range(order+1):
            s2c[i,j] = poly.basis(j)(nodes[i])
    c2s = linalg.inv(s2c)
    return s2c,c2s

def get_modal_differentiation_matrix(order):
    """
    Returns the differentiation matrix for the first derivative in the
    modal basis.
    """
    out = np.zeros((order+1,order+1))
    for i in range(order+1):
        out[:i,i] = poly.basis(i).deriv().coef
    return out

def get_nodal_differentiation_matrix(order,
                                     s2c=None,c2s=None,
                                     Dmodal=None):
    """
    Returns the differentiation matrix for the first derivative
    in the nodal basis

    It goes without saying that this differentiation matrix is for the
    reference cell.
    """
    if Dmodal == None:
        Dmodal = get_modal_differentiation_matrix(order)
    if s2c == None or c2s == None:
        s2c,c2s = get_vandermonde_matrices(order)
    return np.dot(s2c,np.dot(Dmodal,c2s))
# ======================================================================



# Operators Outside Reference Cell
# ======================================================================
def get_colocation_points(order,xmin=LOCAL_XMIN,xmax=LOCAL_XMAX,quad_points=None):
    """
    Generates order+1 colocation points on the domain [xmin,xmax]
    """
    if quad_points == None:
        quad_points = get_quadrature_points(order)
    scale_factor = (xmax-float(xmin))/(LOCAL_XMAX-float(LOCAL_XMIN))
    shift_factor = xmin-float(LOCAL_XMIN)
    return scale_factor*(shift_factor + quad_points)

def get_global_differentiation_matrix(order,
                                      xmin=LOCAL_XMIN,
                                      xmax=LOCAL_XMAX,
                                      s2c=None,
                                      c2s=None,
                                      Dmodal=None):
    """
    Returns the differentiation matrix in the nodal basis
    for the global coordinates (outside the reference cell)

    Takes the Jacobian into effect.
    """
    scale_factor = (xmax-float(xmin))/(LOCAL_XMAX-float(LOCAL_XMIN))
    LD = get_nodal_differentiation_matrix(order,s2c,c2s,Dmodal)
    PD = LD/scale_factor
    return PD

# ======================================================================



# ======================================================================
# Reconstruct Global Solution
# ======================================================================
def get_continuous_object(grid_func,
                          xmin=LOCAL_XMIN,xmax=LOCAL_XMAX,
                          c2s=None):
    """
    Maps the grid function grid_func, which is any field defined
    on the colocation points to a continuous function that can
    be evaluated.

    Parameters
    ----------
    xmin -- the minimum value of the domain
    xmax -- the maximum value of the domain
    c2s  -- The Vandermonde matrix that maps the colocation representation
            to the spectral representation

    Returns
    -------
    An numpy polynomial object which can be called to be evaluated
    """
    order = len(grid_func)-1
    if c2s == None:
        s2c,c2s = get_vandermonde_matrices(order)
    spec_func = np.dot(c2s,grid_func)
    my_interp = poly(spec_func,domain=[xmin,xmax])
    return my_interp
# ======================================================================



# ======================================================================
# A convenience class that generates everything and can be called
# ======================================================================
class PseudoSpectralStencil1D:
    """Given an order, and a domain [xmin,xmax]
    defines internally all structures and methods the user needs
    to calculate spectral derivatives in 1D
    """
    def __init__(self,order,xmin,xmax):
        "Constructor. Needs the order of the method and the domain [xmin,xmax]."
        self.order = order
        self.xmin = xmin
        self.xmax = xmax
        self.quads = get_quadrature_points(self.order)
        self.weights = get_integration_weights(self.order,self.quads)
        self.s2c,self.c2s = get_vandermonde_matrices(self.order,self.quads)
        self.Dmodal = get_modal_differentiation_matrix(self.order)
        self.Dnodal = get_nodal_differentiation_matrix(self.order,
                                                       self.s2c,self.c2s,
                                                       self.Dmodal)
        self.colocation_points = get_colocation_points(self.order,
                                                       self.xmin,self.xmax,
                                                       self.quads)
        self.PD = get_global_differentiation_matrix(self.order,
                                                    self.xmin,self.xmax,
                                                    self.s2c,self.c2s,
                                                    self.Dmodal)

    def get_x(self):
        """
        Returns the colocation points
        """
        return self.colocation_points

    def differentiate(self,grid_func,order=1):
        """
        Given a grid function defined on the colocation points,
        returns its derivative of the appropriate order
        """
        assert type(order) == int
        assert order >= 0
        if order == 0:
            return grid_func
        else:
            return self.differentiate(np.dot(self.PD,grid_func),order-1)

    def to_continuum(self,grid_func):
        """
        Given a grid function defined on the colocation points, returns a
        numpy polynomial object that can be evaluated.
        """
        return get_continuous_object(grid_func,self.xmin,self.xmax,self.c2s)
# ======================================================================


# ======================================================================
# Higher dimensions
# ======================================================================
class PseudoSpectralStencil2D:
    """Given an order in x and y and a domain
    [xmin,xmax]x[ymin,ymax],
    defines a psuedospectral stencil in two dimensions
    """
    def __init__(self,
                 orderx,xmin,xmax,
                 ordery,ymin,ymax):
        "Constructor. Needs order and domain in x and y"
        self.orderx,self.ordery = orderx,ordery
        self.stencils = [PseudoSpectralStencil1D(orderx,xmin,xmax),
                         PseudoSpectralStencil1D(ordery,ymin,ymax)]
        self.stencil_x,self.stencil_y = self.stencils
        self.quads = [s.quads for s in self.stencils]
        self.x,self.y = self.quads
        self.quads2D = np.meshgrid(*self.squads,indexing='ij')
        self.X,self.Y = self.quads2D
        self.s2c2d = pvander2d(self.x,self.y,[orderx,ordery])
        self.c2s2d = linalg.inv(c2s2d)

    def get_x1d(self,axis=0):
        "Returns the colocation points on a given axis"
        return self.quads[axis]

    def get_x2d(self):
        "Returns the 2d grid"
        return self.quads2D

    def differentiate(self,grid_func,orderx,ordery):
        """Given a grid function defined on the colocation points,
        differentiate it up to the appropriate order in each direction.
        """
        assert type(orderx) is int
        assert type(ordery) is int
        assert orderx >= 0
        assert ordery >= 0
        
        if orderx > 0:
            df = np.empty_like(grid_func)
            for j in range(df.shape[1]):
                df[:,j] = self.stencil_x.differentiate(grid_func[:,j],orderx)
            return self.differentiate(df,orderx-1,ordery)

        if ordery > 0:
            df = np.empty_like(grid_func)
            for i in range(df.shape[0]):
                df[i,:] = self.stencil_y.differentiate(grid_func[i,:],ordery)
            return self.differentiate(df,orderx,ordery-1)

        #if orderx == 0 and ordery == 0:
        return grid_func

    def to_continuum(self,grid_func):
        """Given a grid function defined on the colocation points,
        returns a function f(x,y), which can be evaluated anywhere
        in the continuum
        """
        coeffs = np.dot(self.c2s2d,
                        np.flatten(grid_func)).reshape(self.orderx,
                                                       self.ordery)
        f = lambda x,y: pgrid2d(x,y,coeffs)
        return f

    def __call__(self,grid_func,x,y):
        "Evaluates grid_function at points x,y, may interpolate."
        f = self.to_continuum(grid_func)
        return f(x,y)
        
# ======================================================================



# Warning not to run this program on the command line
if __name__ == "__main__":
    raise ImportError("Warning. This is a library. It contains no main function.")
