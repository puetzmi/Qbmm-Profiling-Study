"""
Parameters for postprocessing script.

"""

# Pattern (before the number of moments) occurring in all data files
data_file_pattern = "data_nmom"

# Source directories and corresponding labels
source_dirs = ["intel_np1"]
labels = ["Intel"]

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
        "EigenTridiagonalSymmEigenlibQR None": \
            "QR algorithm (Eigen)", \
        "EigenTridiagonalSymmEigenlibQR LinearVandermondeSolver":
            "QR algorithm (Eigen) + Vandermonde solver", \
        "EigenTridiagonalSymmLapackQR None": \
            "QR (LAPACK)", \
        "EigenTridiagonalSymmLapackQR LinearVandermondeSolver": \
            "QR algorithm (LAPACK) + Vandermonde solver", \
        "EigenTridiagonalSymmLapackRRR None": \
            "RRR algorithm (LAPACK)", \
        "EigenTridiagonalSymmLapackRRR LinearVandermondeSolver": \
            "RRR algorithm (LAPACK) + Vandermonde solver"
                     }

# Dictionary that maps configuration names used in data files to labels used for plots
_errors = [
            "MomentsRelError2Norm", \
            "MomentsRelErrorInfNorm", \
            "QuadratureNodesRelError2Norm", \
            "QuadratureWeightsRelError2Norm"
            ]
error_to_label_map = { \
                        e: e for e in _errors
                      }

# Target directory for output of figures
target_dir = "fig"

# Output format
output_format = ".pdf"
