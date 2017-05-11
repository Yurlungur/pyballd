pyballd
=======

A Pseudospectral Elliptic Solver for Axisymmetric Problems Implemented
in Python

# Domain

The appropriate domain for an axisymmetric problem is

![axisymmetric domain base](eqns/domain_base.gif)

where *r<sub>h</sub>* is some minimum radius. Infinite domains are
difficult to handle. Therefore, following the work of Herdeiro and
Radu [1], we define

![definition of x](eqns/def_x.gif)

and

![definition of X](eqns/def_X.gif)

so that *X* is defined on the domain *[0,1]*. We perform our
differentiation on *X*, which has no effect on the original PDE system
except the introduction of Jacobian terms of the form

![jacobian terms](eqns/X_Jacobian.gif)

in a few places. Since one may want to assume additional (or
different!) symmetry in the longitudinal direction, we do not impose
any restriction there.

# Pseudospectral Derivatives

Pyballd uses Legendre pseudospectral derivatives to attain very high
accuracy with fairly low resolution. For example, if we numerically
take second-order derivatives of this function:

![analytic function](figs/test_function.png)

and vary the number of points (or alternatively the maximum order of
Legendre polynomial used for differentiation), we find that our error
decays exponentially with the number of points. This is called
"spectral" or "evanescent" convergence:

![evanescent convergence](figs/orthopoly_errors.png)

# References

[1] Herderio, Radu, Runarrson. Kerr black holes with Proca
hair. *Classical and Quantum Gravity* **33-15** (2016).
