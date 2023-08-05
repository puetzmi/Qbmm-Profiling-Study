#!/usr/bin/env python3
"""!
@file generate_random_moments.py
@author M. Puetz
@brief
This Python script generates pseudo-random moment sequences based on given command line parameters in the form `KEY=VALUE` (for details see further below) and creates output with additional quantities characterizing the generated moment sequences.

@param nmom Number of moments per moment set.
@param nsets Number of moment sets to be generated.
@param delta Distribution parameter(s) @f$ \delta_j @f$, see @cite Dette_2012,@cite Dette_2016. VALUE may either be a list of length nmom-1 in the form `delta=[..,..,..]` or a single number, e.g. `delta=1.5`. In case of the latter, the distribution parameters will be a uniform list with the given value.
@param gamma Distribution parameter(s) @f$ \gamma_j @f$, see @cite Dette_2012,@cite Dette_2016. VALUE may either be a list of length nmom/2-1 in the form `gamma=[..,..,..]` or a single number, e.g. `gamma=1`. In case of the latter, the vector of distribution parameters will be calculated such that the shape parameter of the Gamma-distribution in each dimension equals the given value.
@param random-seed Seed for random number generator.
@param outfile-prefix Prefix of output files (optional, default is an empty string).
@param outfile-suffix Suffix, e.g. file extension, of output files (optional, default is an empty string).
@param output-dir Output directory (optional, default is the current working directory).

@par Output
This script generates output files containing
    - the recurrence coefficients of orthogonal polynomials (sampled in double precision and converted to long double),
    - the moments (computed in long double precision),
    - the nodes and weights of the corresponding quadrature,
    - properties of the Hankel moment matrix, in particular
        - the determinant,
        - the smallest singular value,
        - the radius of nonsingularity (see @cite Poljak1993,@cite Hartman2018),
        - the Forbenius norm
    - the "normalized distance" of the (2n-2)th moment from the same moment on the boundary of the moment space.


@par Examples
The following command generates 1000 sequences of @f$ m = 10 @f$ moments with the distribution parameters @f$ \delta_j = 1 \; \forall j \in \{1,2,\dots,9 \} @f$ and @f$ \gamma_j = 3/2 + 2j - m \; \forall j \in \{1,2,3,4\} @f$ (follows from the statements above and e.g. @cite Dette_2016 Eq. (2.11) and Lemma 2.3), and creates output files of the form 'my_random_moms_{OUTPUT_QUANTITY}.dat':
@code{.sh}
python3 generate_random_moments.py nmom=10 nsets=1000 delta=1 gamma=1.5 outfile-prefix=my_random_moms outfile-suffix=.dat
@endcode
The same is achieved by
@code{.sh}
python3 generate_random_moments.py nmom=10 nsets=1000 delta=[1,1,1,1,1,1,1,1,1] gamma=[-6.5,-4.5,-2.5,-0.5] outfile-prefix=my_random_moms outfile-suffix=.dat
@endcode

@par Additional information
The random moments are generated using the module `moments.randmom` of the Python package `quadmompy`, For additional information, see the inline comments in this file and the documentation's main page.

"""
import numpy as np
import sys
import time
from quadmompy.moments.randmom import RandomHamburgerMoments
from quadmompy.core.hankel import HankelMatrix
from quadmompy.moments.transform import rc2mom
import os
from scipy.linalg import eigh_tridiagonal
from quad_correct import correct_quadrature
import mpmath


def binary_table(n_digits):
    """!
    @brief Make binary table, i.e.\ an array containing all binary numbers with given number of digits.

    @param n_digits int: Number of digits.

    @return array: NumPy array of shape `(2**n_digits,n_digits)` with all possible binary numbers in ascending order.

    """
    return np.array([ \
                        np.array(list(np.binary_repr(num).zfill(n_digits))).astype(np.int8) \
                        for num in range(2**n_digits) \
                    ])


def main():
    """!
    @brief Main function to generate random moments based on command line parameters (see file description).

    """

    # Use longdouble precision in eigenvalue computation
    mpmath.mp.dps = np.finfo(np.longdouble).precision

    # List of valid parameters (see file description)
    valid_params = ["nmom", "nsets", "delta", "gamma", "outfile-prefix", "outfile-suffix", "random-seed", "output-dir"]

    # Name of this script
    this_file = os.path.basename(__file__)

    # Timestamp in output files
    #now = datetime.now()
    #timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    now = time.localtime()
    timestamp = "{0:d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d} {6:s}".format( \
            now.tm_year, now.tm_mon, now.tm_mday, \
            now.tm_hour, now.tm_min, now.tm_sec, time.tzname[now.tm_isdst] \
            )


    # Read required parameters from command line arguments of the form `KEY=VALUE`
    args = dict([arg.split("=") for arg in sys.argv[1:]])

    # Check for invalid parameters
    for key in args.keys():
        if key not in valid_params:
            msg = "Invalid parameter key '{0:s}'. Allowed keys are: {1!s}".format(key, valid_params)
            raise KeyError(msg)

    # Number of moments
    nmom = int(args["nmom"])
    n = nmom//2

    # Number of moment sets to be sampled
    nsets = int(args["nsets"])

    # Seed for random number generator
    random_seed = int(args["random-seed"])

    # Distribution parameters, see documentation of the `quadmompy.moments.randmom`-module and docstring of this script
    gamma = eval(args["gamma"])
    try:
        if len(gamma) != nmom//2 - 1:   # Check if gamma has valid length
            msg = "The distribution parameter `gamma` must be a single number or a list of length `nmom//2 - 1`."
            raise ValueError(msg)
        gamma = np.array(gamma, dtype=float)
    except TypeError:
        gamma = gamma - 2*(n - np.arange(1, n, dtype=float))

    delta = eval(args["delta"])
    try:
        if len(delta) != nmom - 1:   # Check if delta has valid length
            msg = "The distribution parameter `delta` must be a single number or a list of length `nmom - 1`."
            raise ValueError(msg)
        delta = np.array(delta, dtype=float)
    except TypeError:
        delta = delta*np.ones(2*n-1)


    # Optional parameters
    try:
        outfile_prefix = args["outfile-prefix"]
    except KeyError:
        outfile_prefix = ""
    try:
        outfile_suffix = args["outfile-suffix"]
    except KeyError:
        outfile_suffix = ""
    try:
        output_dir = args["output-dir"]
    except KeyError:
        output_dir = os.getcwd()

    # Generate random moments
    rng = np.random.default_rng(random_seed)
    mom_gen = RandomHamburgerMoments(nmom=nmom, delta=delta, gamma=gamma, rng=rng, dtype=np.longdouble)

    # Assign/compute output quantities; the quantities that are used as input for computations
    # have long double precision
    mom = np.empty((nsets, nmom), dtype=np.longdouble)          # moments
    alpha = np.empty((nsets, nmom//2), dtype=np.longdouble)     # first set of recurrence coefficients of orthogonal polynomials
    beta = np.empty((nsets, nmom//2), dtype=np.longdouble)      # second set of recurrence coefficients of orthogonal polynomials
    sqrt_beta = np.empty((nsets, nmom//2), dtype=np.longdouble) # sqrt(beta), the sub-/superdiagonal of the Jacobi matrix
    hdet = np.empty(nsets)                                      # determinant of Hankel moment matrix
    h_fro_norm = np.empty(nsets)                                # Frobenius norm of the Hankel matrix
    rreg = np.empty(nsets)                                      # regularity radius
    sigma_min = np.empty(nsets)                                 # smallest singular value
    mom_bound_dist = np.empty(nsets)                            # distance from the (2n-2)th moment on boundary
    quad_nodes = np.empty((nsets, nmom//2), dtype=np.longdouble)# Quadrature nodes
    quad_weights = np.empty((nsets,nmom//2),dtype=np.longdouble)# Quadrature weights
    size = nmom//2
    Z = -1 + 2*binary_table(size).T.astype(np.float64)          # Matrix with all elements +-1 for computation of regularity radius
    for i in range(nsets):
        mom[i] = mom_gen()
        alpha[i] = mom_gen.alpha
        beta[i] = mom_gen.beta
        beta_boundary = beta[i].copy()
        beta_boundary[-1] = 0.
        mom_boundary = rc2mom(alpha[i], beta_boundary)
        assert(np.allclose(mom[i][:-2], mom_boundary[:-2]))
        mom_bound_dist[i] = (mom[i][-2] - mom_boundary[-2])**(1/(2*n-2))
        assert(mom_bound_dist[i] >= 0)
        sqrt_beta[i] = beta[i]**0.5
        jacobi_matrix = np.diag(sqrt_beta[i,1:], -1)
        jacobi_matrix += jacobi_matrix.T
        jacobi_matrix += np.diag(alpha[i])
        x, w = mpmath.eigsy(mpmath.matrix(jacobi_matrix))
        quad_nodes[i,:] = x
        idx = np.argsort(x)
        quad_nodes[i,:] = quad_nodes[i,idx]
        quad_weights[i,:] = np.array(w, dtype=np.longdouble).reshape(jacobi_matrix.shape)[0,idx]**2

        # This may increase precision in terms of the reconstructed moments. It is,
        # however, questionable if it works without major changes in the entire procedure.
        # As the quadrature nodes and weights are currently only used as input for the isolated
        # linear solver comparison, increasing the precision is deemed unnecessary.
        #
        # quad_nodes[i], quad_weights[i] \
        #         = correct_quadrature( \
        #         quad_nodes[i], quad_weights[i], mom[i],
        #         rtol=0.1*np.finfo(np.float64).eps)          # High-precision quadrature correction
        H = HankelMatrix.from_moments(mom[i])               # Hankel moment matrix
        U, sigma, Vt = np.linalg.svd(H.matrix())            # Singular value decomposition
        hdet[i] = np.prod(sigma)                            # Hankel determinant (H is known to be symmetric positive defininte)
        h_fro_norm[i] = np.linalg.norm(sigma, 2)            # Frobenius norm from singular values
        Hinv = (Vt.T/sigma)@U.T                             # Inverse from SVD
        rreg[i] = 1./max(np.linalg.norm(Hinv@Z, 1, axis=0))
        sigma_min[i] = sigma[-1]


    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    elif not os.path.isdir(output_dir):
        msg = "The specified output directory name '{0:s}' exists but is not a directory.".format(output_dir)
        raise OSError(msg)

    # write moments to file
    header_timestamp = " ({0:s}).\n".format(timestamp)
    header_params = "Distribution parameters:\n" \
                    + "gamma = {0!s}\n".format(gamma) \
                    + "delta = {0!s}\n".format(delta)

    outfile = "{0:s}moments{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Random moments generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, mom, header=header)

    # write alpha recurrence coefficients to file
    outfile = "{0:s}alpha-coeffs{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Recurrence cofficients `alpha` generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, alpha, header=header)

    # write beta recurrence coefficients to file
    outfile = "{0:s}beta-coeffs{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Recurrence cofficients `beta` generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, beta, header=header)

    # write gamma (first diagonal of Jacobi matrix) to file
    outfile = "{0:s}gamma-coeffs{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Sub-/superdiagonal elements of the Jacobi matrix generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, sqrt_beta, header=header)

    # write Hankel determinants
    outfile = "{0:s}hankel-determinant{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Hankel determinants corresponding to random moments generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, hdet, header=header)

    # write radius of regularity
    outfile = "{0:s}regularity-radius{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Nonsingularity radius of the Hankel matrix corresponding to random moments generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, rreg, header=header)

    # write smallest singular value
    outfile = "{0:s}sigma-min{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Smallest singular value of the Hankel matrix corresponding to random moments generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, sigma_min, header=header)

    # write Frobenius norm of Hankel matrix
    outfile = "{0:s}hankel-frobenius-norm{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Frobenius norm of the Hankel matrix corresponding to random moments generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, h_fro_norm, header=header)

    # write distance of m_{2n-2} from moment space boundary
    outfile = "{0:s}mom2nm2-boundary-dist{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "The relative distance of the (2n-2)th moment from the boundary".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, mom_bound_dist, header=header)

    # write quadrature nodes
    outfile = "{0:s}quad-nodes{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Quadrature nodes corresponding to random moments generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, quad_nodes, header=header)

    # write quadrature weights
    outfile = "{0:s}quad-weights{1:s}".format(outfile_prefix, outfile_suffix)
    outfile = os.path.join(output_dir, outfile)
    header = "Quadrature nodes corresponding to random moments generated by '{0:s}'".format(this_file) \
            + header_timestamp \
            + header_params
    np.savetxt(outfile, quad_weights, header=header)


if __name__ == "__main__":
    main()
