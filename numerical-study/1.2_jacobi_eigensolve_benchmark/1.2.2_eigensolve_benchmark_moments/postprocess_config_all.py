"""
Parameters for postprocessing script.

"""

# Pattern (before the number of moments) occurring in all data files
data_file_pattern = "data_nmom"

# Source directories and corresponding labels
source_dirs = ["gnu_np1", "gnu_np6", "intel_np1", "intel_np6"]
labels = ["GNU (1 core)", "GNU (6 cores)", "Intel (1 core)", "Intel (6 cores)"]

# Main directory for complete analysis
main_dir = "intel_np1"

# Directory containing the input data for the benchmark application
data_dir = "../data"

# Potential common prefix of all data files before `data_file_pattern`
infile_prefix = ""

# Common suffix of all data files after the number of moments
infile_suffix = ".out"

# Dictionary that maps configuration names used in data files to labels used for plots
config_to_label_map = { \
                        "LqmdAlgorithm": "LQMDA", \
                        "GolubWelschAlgorithmLapackPotrf": "GWA (MKL/LAPACK-POTRF)", \
                        "GolubWelschAlgorithmLapackPotrf2": "GWA (MKL/LAPACK-POTRF2)", \
                        "GolubWelschAlgorithmEigenlib": "GWA (Eigen3)", \
                        "GolubWelschAlgorithmPlainCxx": "GWA (plain C++)"
                     }

# Dictionary that maps configuration names used in data files to labels used for plots
error_to_label_map = { \
                        "JacobiMatrixRelErrorFrobeniusNorm": r"Error in Jacobi matrix (Frobenius norm)", \
                      }

# Target directory for output of figures
target_dir = "fig"

# Output format
output_format = ".pdf"
