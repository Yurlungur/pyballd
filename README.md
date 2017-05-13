pyballd
=======

Author: Jonah Miller (jonah.maxwell.miller@gmail.com)

A Pseudospectral Elliptic Solver for Axisymmetric Problems Implemented
in Python

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

# Domain

The appropriate domain for an axisymmetric problem is

![axisymmetric domain base](eqns/domain_base.gif)

where *r<sub>h</sub>* is some minimum radius. Infinite domains are
difficult to handle. Therefore, following the work of Herdeiro and
Radu [1], we define either

![definition of x for most boundaries](eqns/def_x_dirichlet.gif)

for most boundary situations or 

![definition of x](eqns/def_x_bh.gif)

when *r<sub>h</sub>* is a black hole event horizon. We then define

![definition of X](eqns/def_X.gif)

so that *X* is defined on the domain *[0,1]*. We perform our
differentiation on *X*, which has no effect on the original PDE system
except the introduction of Jacobian terms of the form

![jacobian terms](eqns/X_Jacobian.gif)

in a few places. Since one may want to assume additional (or
different!) symmetry in the longitudinal direction, we do not impose
any restriction there.

## Jacobian for the Compactified Domain

When *x* is defined as 

![definition of x for most boundaries](eqns/def_x_dirichlet.gif)

the Jacobian for the coordinate transformation looks like

![Jacobian for the coordinate transformation](figs/domain_dXdr.png)

The primary advantage is that *1/r<sup>n</sup>* falloffs are linear in
this coordinate system and so a low-order spectral method will
represent the solution exactly.

## Convergence on the Compactified Domain

For more complex functions, such as this one:

![test function on compact domain](figs/domain_test_function_2d.png)

which has this derivative

![derivative of test function on compact domain](figs/deriv_domain_test_function_2d.png)

On the compact domain (on the equator), this function becomes

![test function on equator](figs/domain_test_function.png)

with derivative

![derivative of test function on equator](figs/deriv_domain_test_function.png)

In this setup, our convergence becomes substantially slower. However,
we still retain spectral convergence as this plot of the maximum of
the errors in the derivative of the above function on the
compactified domain shows:

![errors on compactified domain](figs/domain_l1_errors.png)

# Solving an Elliptic PDE

In Pyballd, an elliptic system is defined via a *residual.* A residual

![residual](eqns/residual.gif)

acts on a state vector *u* and its first and second derivatives in (in
our case, axisymmetry) *r* and &#952;. If

![residual vanishes](eqns/residual_vanishes.gif)

then *u(r,&#952;)* is a solution to the PDE system.

An elliptic PDE is not well-posed without the addition of boundary
conditions, which select for the particular solution. At infinity
(*X=1*), we automatically demand that the solution must vanish. (In
other words, we demand that all solutions are square-integrable.)

Dirichlet, Von-Neumann, or Robin boundary conditions can be imposed on
the inner radius (*r_h*), the position of minimum &#952; (often the
axis of symmetry), and the position of maximum &#952; (often the axis
of symmetry or the equator).

In the case of black-hole like compactifications, where

![definition of x](eqns/def_x_bh.gif)

we automatically impose Von-Neumann boundary conditions at the inner
boundary such that

![von neumann regularity condition](eqns/von-neumann-regularity-condition.gif)

which is a regularity condition we need to impose on solutions in
these coordinates.

# References

[1] Herderio, Radu, Runarrson. Kerr black holes with Proca
hair. *Classical and Quantum Gravity* **33-15** (2016).
