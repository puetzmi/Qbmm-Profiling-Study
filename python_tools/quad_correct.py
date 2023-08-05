"""!
@file quad_correct.py
@author M. Puetz
@brief This Python module contains functions to correct a Gaussian quadrature to precisely match a given set of moments.

The correction of moments up to a long double precision occurs using the so-called good Broyden method (see @cite Broyden1965,@cite Broyden2000). A very good initial guess is required to achieve convergence.

"""

import numpy as np
import mpmath


def _linsolve(A, b): 
    dtype = A.dtype
    precision = np.finfo(dtype).precision
    mpmath.mp.dps = precision
    mat = mpmath.matrix(A)
    rhs = mpmath.matrix(b)
    return np.array(mpmath.lu_solve(mat, rhs), dtype=dtype)
 
 
def newton(f, x0, rhs, jac, atol, rtol, maxiter=1000, args=()):
    """
    @brief Find the solution to a system of non-linear equations using Newton's method.
    @param x array: Vector of independent variables / initial guess.
    @param f callable: Target function taking a vector as parameter and returning a vector of same length.
    @param rhs array: Right-hand-side vector such that the algorithm finds the roots of `f(x) - rhs`.
    @param jac callable: Callable object returning the Jacobian of shape `(len(x),len(x))` given `x` as paramter.
    @param atol float: Absolute tolerance with respect to the maximum error in the dependent variable.
    @param rtol float: Relative tolerance with respect to the maximum error in the dependent variable.
    @param max_iter int: Maximal allowed number of iterations (optional, default is 1000).
    @param args tuple: Additional arguments passed to target function and jacobian function (optional).

    @return it int: Number of iterations
    @return x array: Approximate solution.

    """
    x = x0.copy()
    _f = lambda x: f(x) - rhs
    for it in range(maxiter+1):
        fx = _f(x, *args)
        if np.all(np.abs(fx) < atol + rtol*np.abs(rhs)):
            return it, x
        delta = _linsolve(jac(x, *args), -fx)
        x += delta

    msg = "Newton's method did not converge after {0:d} iterations.".format(maxiter)
    raise RuntimeError(msg)


def good_broyden(x, func, Jfunc, rhs=None, atol=None, rtol=0., max_iter=50, args=()):
    """!
    @brief Find the solution to a system of non-linear equations using the good Broyden method @cite Broyden1965,@cite Broyden2000.

    @param x array: Vector of independent variables / initial guess.
    @param func callable: Target function taking a vector as parameter and returning a vector of same length.
    @param Jfunc callable: Callable object returning the Jacobian of shape `(len(x),len(x))` given `x` as paramter.
    @param rhs array: Right-hand-side vector such that the algorithm finds the roots of `func(x) - rhs` (optional, default is a zero-vector).
    @param atol float: Absolute tolerance with respect to the maximum error in the dependent variable (optional, default is machine epsilon of the data type of `x`).
    @param rtol float: Relative tolerance with respect to the maximum error in the dependent variable (optional, default is 0, i.e. no consideration of relative tolerance).
    @param max_iter int: Maximal allowed number of iterations (optional, default is 50).
    @param args tuple: Additional arguments passed to target function and jacobian function (optional).

    @return it int: Number of iterations
    @return x array: Approximate solution.

    """
    if rhs is None:
        rhs = np.zeros_like(x)

    if atol is None:
        atol = np.finfo(x.dtype).eps

    _func = lambda x, *args: func(x, *args) - rhs
    f = _func(x, *args)
    J = Jfunc(x, *args)
    s = np.empty_like(x)

    for it in range(max_iter):

        #s = la.solve(J,-1*f)
        s[:] = mpmath.lu_solve(J, -f)
        x += s
        f_new = _func(x, *args)
        y = f_new - f
        J += np.outer(y - J@s, s) / (s@s)
        f = f_new

        if np.all(np.abs(f) <= atol + rtol*np.abs(rhs)):
            return it, x

    msg = "Broyden's method did not converge after {0:d} iterations.".format(max_iter)
    raise RuntimeError(msg)


def compute_moments(x, w):
    """!
    @brief Compute 2n moments given a set of n quadrature nodes and weights

    @param x array: Array of length `n` containing the quadrature nodes.
    @param w array: Array of length `n` containing the quadrature weights.

    @return moments array: Array of length `2*n` with the computed moments.

    """
    n = len(x)
    f = np.array( \
            [w@x**k for k in range(2*n)], \
            dtype=x.dtype)
    return f


def compute_jacobian(x, w):
    """!
    @brief Compute Jacobian of the moment function @f$ \nabla \mathbf{m}(\mathbf{z}) @f$,
    where @f$ \mathbf{m} : \mathbb{R}^{2n} \rightarrow \mathbb{R}^{2n}, \; z \in \mathbb{R}^{2n} @f$ represents
    the @f$ n @f$ Gaussian quadrature nodes and weights, i.e. @f$ \mathbf{z} = (x_1,\dots,x_n,w_1,\dots,w_n)^T @f$.

    @param x array: The quadrature nodes @f$ x_1,\dots,x_n @f$.
    @param w array: The quadrature weights @f$ w_1,\dots,w_n @f$.

    @return J array: The Jacobian matrix @f$ \nabla \mathbf{m}(\mathbf{z}) \in \mathbb{R}^{2n \times 2n} @f$.

    """
    n = len(x)
    jac = np.empty((2*n,2*n), dtype=x.dtype)
    for k in range(2*n):
        for j in range(n):
            factor = np.ones(n, dtype=x.dtype)
            exponent = k*factor
            factor[j] = k
            exponent[j] -= 1
            jac[k,j] = factor*w@(x**exponent)
            jac[k,j+n] = x[j]**k
    return jac


def correct_quadrature(nodes, weights, moments, dtype=np.longdouble, rtol=None):
    """!
    Compute corrected/refined Gaussian quadrature given a very good initial guess and a set of moments to match with high precision.

    Given a very good initial guess, the refinement of the quadrature is done using a root finding algorithm, th so-called
    good Broyden method, see @cite Broyden1965,@cite Broyden2000.

    @param nodes array: Array of length `n` containing the intial Gaussian quadrature nodes.
    @param weights array: Array of length `n` containing the intial Gaussian quadrature weights.
    @param moments array: Array of length `2*n` with the moments that must be matched by the corrected quadrature.
    @param dtype array: Data type of the corrected quadrature nodes and weights, determines the default precision (optional, default is numpy.longdouble).
    @param rtol float: Relative tolerance for moment correction (oprional, default is the machine epsilon of `dtype`).

    @return nodes_corr array: Corrected quadrature nodes matching the given moments.
    @return weights_corr array: Corrected quadrature weights matching the given moments.

    """
    if rtol is None:
        rtol = 2*np.finfo(dtype).eps

    n = len(nodes)
    x = nodes.astype(dtype)
    w = weights.astype(dtype)
    z = np.empty(2*n, dtype=dtype)
    z[:n] = x
    z[n:] = w
    f = lambda z: compute_moments(z[:n], z[n:])
    J = lambda z: compute_jacobian(z[:n], z[n:])
    atol = max(10*np.finfo(dtype).eps, 0.1*np.finfo(np.float64).eps)
    rtol = max(10*np.finfo(dtype).eps, 0.1*np.finfo(np.float64).eps)
    #_, z_corr = good_broyden(z, f, J, rhs=moments, rtol=rtol, max_iter=1000)
    _, z_corr = newton(f, x0=z, rhs=moments, jac=J, rtol=rtol, atol=atol)
    nodes_corr, weights_corr = z_corr[:n], z_corr[n:]
    return nodes_corr, weights_corr