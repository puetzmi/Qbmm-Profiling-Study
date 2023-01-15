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

# Target directory for output of figures
target_dir = "fig"

# Output format
output_format = ".pdf"
