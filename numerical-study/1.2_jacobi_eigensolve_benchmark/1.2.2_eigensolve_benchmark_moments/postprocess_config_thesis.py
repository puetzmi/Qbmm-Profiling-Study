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

# Dictionary that maps configuration names used in data files to labels used for plots
config_to_label_map = { \
        "EigenTridiagonalSymmEigenlibQR+None": \
            "QR (Eigen3)", \
        "EigenTridiagonalSymmEigenlibQR+LinearVandermondeSolver":
            "QR (Eigen3) + Vandermonde solver", \
        "EigenTridiagonalSymmLapackQR+None": \
            "QR (LAPACK)", \
        "EigenTridiagonalSymmLapackQR+LinearVandermondeSolver": \
            "QR (LAPACK) + Vandermonde solver", \
        "EigenTridiagonalSymmLapackRRR+None": \
            "RRR (LAPACK)",
        "EigenTridiagonalSymmLapackRRR+LinearVandermondeSolver": \
            "RRR (LAPACK) + Vandermonde solver"
         }

# Dictionary that maps configuration names used in data files to labels used for plots
output_qty_to_label_map = \
    {
            "MomentsRelError2Norm": \
                "Rel. moment error (Euclidean norm)",
            "MomentsRelErrorInfNorm": \
                "Max. rel. moment error",
            "ComputingTime": \
                "Computation time",
            "RelativeSeparationCoefficient": \
                "Rel. separation coeff."
    }

# Quantities used to measure distance to moment space boundary
# (as in input filenames)
boundary_dist_quantities = ["regularity-radius"]
  #["sigma-min", "regularity-radius", "hankel-determinant", "beta-coeffs", "mom2nm2-boundary-dist"]

# Names / labels corresponding to `quantities`
boundary_dist_quantity_names = [r"$\mathrm{r_{reg}(\mathbf{M}_{n})}$"]
"""
  [r"$\mathrm{\sigma_{min}}$",
   r"$\mathrm{r_{reg}}$",
   r"det($\mathrm{\mathbf{M}_{n}})$",
   r"$\mathrm{\beta_{min}}$",
   r"$\mathrm{d_{2n-2}}$"]
"""

# functions to apply to input data
funcs = {quantity: lambda x, _: x for quantity in boundary_dist_quantities}
#funcs["hankel-determinant"] = lambda x, nmom: x**(2/nmom)
#funcs["beta-coeffs"] = lambda x, _: np.min(x[:,1:], axis=1)

# Target directory for output of figures
target_dir = "fig/doctoral_thesis_puetz"

# Output format
output_format = ".pdf"

# Plot histograms
plot_histograms = True

# Number of columns in figures (only to adjust figure size)
n_fig_columns = 1

# Specify pairs of mean errors plotted side by side
# in terms of the number of moments
#mean_nmom_pairs = [(6,16)]
mean_nmom_pairs = [(6,16)]

# Specify pairs of inversion types, of which histograms shall be plotted as examples in 2-by-2 figures
#mean_comptype_pairs = [("LqmdAlgorithm", "GolubWelschAlgorithmPlainCxx")]
mean_comptype_pairs = [("EigenTridiagonalSymmLapackQR+None",
        "EigenTridiagonalSymmLapackQR+LinearVandermondeSolver")]
