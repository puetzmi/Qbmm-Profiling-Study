"""
Parameters for postprocessing script.

"""

# Pattern (before the number of moments) occurring in all data files
data_file_pattern = "data_nmom"

# Source directories and corresponding labels
source_dirs = [
                "intel_np1",
                "intel_np6",
                "gnu_np6",
            ]
labels = [
            "Intel (1 core)",
            "Intel (6 cores)",
            "GNU (1 core)",
        ]

# Main directory for complete analysis
main_dir = "intel_np1"

# Directory containing the input data for the benchmark application
data_dir = "../../data"

# Potential common prefix of all data files before `data_file_pattern`
infile_prefix = ""

# Common suffix of all data files after the number of moments
infile_suffix = ".out"

# Plot histograms of errors
plot_histograms = True

# Dictionary that maps configuration names used in data files to labels used for plots
config_to_label_map = { \
        "EigenTridiagonalSymmEigenlibQR+None": \
            "QR (Eigen)", \
        "EigenTridiagonalSymmEigenlibQR+LinearVandermondeSolver":
            "QR (Eigen) + Vandermonde solver", \
        "EigenTridiagonalSymmLapackQR+None": \
            "QR (LAPACK)", \
        "EigenTridiagonalSymmLapackQR+LinearVandermondeSolver": \
            "QR (LAPACK) + Vandermonde solver", \
        "EigenTridiagonalSymmLapackRRR+None": \
            "RRR (LAPACK)", \
        "EigenTridiagonalSymmLapackRRR+LinearVandermondeSolver": \
            "RRR (LAPACK) + Vandermonde solver"
                     }

# Dictionary that maps configuration names used in data files to labels used for plots
_errors = [
            ]
error_to_label_map = { \
                        e: e for e in _errors
                      }

# Dictionary that maps configuration names used in data files to labels used for plots
error_to_label_map = \
    {
            "MomentsRelError2Norm": \
                "Rel. moment error (euclidean norm)",
            "MomentsRelErrorInfNorm": \
                "Max. rel. moment error",
            "ComputingTime": \
                "CPU time",
            "RelativeSeparationCoefficient": \
                "Rel. separation coeff."
    }

# Quantities used to measure distance to moment space boundary
# (as in input filenames)
boundary_dist_quantities = ["regularity-radius"]
  #["sigma-min", "regularity-radius", "hankel-determinant", "beta-coeffs", "mom2nm2-boundary-dist"]

# Names / labels corresponding to `quantities`
boundary_dist_quantity_names = [r"$\mathrm{r_{reg}(\mathbf{M}_{2n-2})}$"]
"""
  [r"$\mathrm{\sigma_{min}}$",
   r"$\mathrm{r_{reg}}$",
   r"det($\mathrm{\mathbf{M}_{2n-2}})$",
   r"$\mathrm{\beta_{min}}$",
   r"$\mathrm{d_{2n-2}}$"]
"""

# functions to apply to input data
funcs = {quantity: lambda x, _: x for quantity in boundary_dist_quantities}
#funcs["hankel-determinant"] = lambda x, nmom: x**(2/nmom)
#funcs["beta-coeffs"] = lambda x, _: np.min(x[:,1:], axis=1)

# Target directory for output of figures
target_dir = "fig"

# Output format
output_format = ".pdf"
