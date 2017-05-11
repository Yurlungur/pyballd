#!/usr/bin/env python2

"""
orthopoly.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-11 17:03:47 (jmiller)>

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
pval2d = polynomial.legendre.legval2d
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
    if nodes is None:
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
    if nodes is None:
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
    if Dmodal is None:
        Dmodal = get_modal_differentiation_matrix(order)
    if s2c is None or c2s is None:
        s2c,c2s = get_vandermonde_matrices(order)
    return np.dot(s2c,np.dot(Dmodal,c2s))
# ======================================================================



# Operators Outside Reference Cell
# ======================================================================
def get_colocation_points(order,xmin=LOCAL_XMIN,xmax=LOCAL_XMAX,quad_points=None):
    """
    Generates order+1 colocation points on the domain [xmin,xmax]
    """
    if quad_points is None:
        quad_points = get_quadrature_points(order)
    local_width=float(LOCAL_XMAX-LOCAL_XMIN)
    physical_width = float(xmax-xmin)
    x = xmin + (LOCAL_XMAX+quad_points)*physical_width/local_width
    return x

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

    def get_scale_factor(self):
        "Jacobian between local and global"
        return (self.xmax-float(self.xmin))/(LOCAL_XMAX-float(LOCAL_XMIN))

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
        self.colocs = [s.colocation_points for s in self.stencils]
        self.x,self.y = self.colocs
        self.colocs2d = np.meshgrid(*self.colocs,indexing='ij')
        self.X,self.Y = self.colocs2d
        self.weights = [s.weights for s in self.stencils]
        self.weights_x,self.weights_y = self.weights
        self.weights2D = np.tensordot(*self.weights,axes=0)

    def get_x1d(self,axis=0):
        "Returns the colocation points on a given axis"
        return self.colocs[axis]

    def get_x2d(self):
        "Returns the 2d grid"
        return self.colocs2d

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
            return self.differentiate(df,0,ordery)

        if ordery > 0:
            df = np.empty_like(grid_func)
            for i in range(df.shape[0]):
                df[i,:] = self.stencil_y.differentiate(grid_func[i,:],ordery)
            return self.differentiate(df,orderx,0)

        #if orderx == 0 and ordery == 0:
        return grid_func

    def inner_product(self,gf1,gf2):
        """Calculates the 2D inner product between grid functions
        gf1 and gf2 using the appropriate quadrature rule
        """
        factors = [s.get_scale_factor() for s in self.stencils]
        factor = np.prod(factors)
        integrand = gf1*self.weights2D*gf2
        integral_unit_cell = np.sum(integrand)
        integral_physical = integral_unit_cell*factor
        return integral_physical

    def norm2(self,grid_func):
        """Calculates the 2norm of grid_func"""
        factor = np.prod([(s.xmax-s.xmin) for s in self.stencils])
        integral = self.inner_product(grid_func,grid_func) / factor
        norm2 = np.sqrt(integral)
        return norm2

    def to_continuum(grid_func):
        coeffs_x = np.empty_like(grid_func)
        for j in range(coeffs_x.shape[1]):
            coeffs_x[:,j] = np.dot(self.stencil_x.c2s,grid_func[:,j])
        coeffs_xy = np.empty_like(grid_func)
        for i in range(coeffs_xy.shape[0]):
            coeffs_xy[i,:] = np.dot(self.stencil_y.c2s,coeffs_x[i,:])
        f = lambda x,y: pval2d(x,y,coeffs_xy)
        return f

    def __call__(self,grid_func,x,y):
        "Evaluates grid_function at points x,y, may interpolate."
        f = self.to_continuum(grid_func)
        return f(x,y)
        
# ======================================================================



# Warning not to run this program on the command line
if __name__ == "__main__":
    raise ImportError("Warning. This is a library. It contains no main function.")
