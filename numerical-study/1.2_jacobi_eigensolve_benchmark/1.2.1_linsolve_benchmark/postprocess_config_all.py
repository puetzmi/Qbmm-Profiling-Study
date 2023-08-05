"""
Parameters for postprocessing script.

"""

# Pattern (before the number of moments) occurring in all data files
data_file_pattern = "data_nmom"

# Source directories and corresponding labels
source_dirs = ["gnu", "intel"]
labels = ["GNU", "Intel"]

# Main directory for complete analysis
main_dir = "intel"

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
                        "LinearLapackGesvSolver": "LU decomposition (LAPACK)", \
                        "LinearEigenlibPartialPivLuSolver": "LU decomposition (Eigen3)", \
                        "LinearVandermondeSolver": "Vandermonde solver"
                     }

# Dictionary that maps configuration names used in data files to labels used for plots
error_to_label_map = { \
                        "MomentsRelError2Norm": r"Rel. error in moments $||\mathbf{m}_{rerr}||_2$", \
                        "MomentsRelErrorInfNorm": r"Rel. error in moments $||\mathbf{m}_{rerr}||_{\infty}$", \
                        "WeightsRelError2Norm": r"Rel. error in weight vector",
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
