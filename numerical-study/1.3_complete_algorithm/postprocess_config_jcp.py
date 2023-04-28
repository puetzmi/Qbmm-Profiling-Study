"""
Parameters for postprocessing script.

"""

# Pattern (before the number of moments) occurring in all data files
data_file_pattern = "data_nmom"

# Source directories and corresponding labels
source_dirs = ["intel_np6"]
labels = ["Intel (6 cores)"]

# Main directory for complete analysis
main_dir = "intel_np6"

# Directory containing the input data for the benchmark application
data_dir = "../data"

# Potential common prefix of all data files before `data_file_pattern`
infile_prefix = ""

# Common suffix of all data files after the number of moments
infile_suffix = ".out"

# Specify types of QMOM, inversion, eigensolver and linear solver
qmom_type = "QmomStd"
inv_type = "LqmdAlgorithm"
eigensolver_type = "EigenTridiagonalSymmLapackQR"
linearsolver_type = "None"

# Map physical configurations to labels
config_to_label_map = {
        "FokkerPlanckConstantCd":
            "FPE (constant)",
        "FokkerPlanckVelocityDependentCd":
            "FPE (velocity-dependent)",
        "HardSphereCollision1D":
            "Hard-sphere collision"
}

# Map moment closure subroutines to labels
subroutine_to_label_map = {
        "CoreInversion":
            r"Subroutine (i): computation of Jacobi matrix",
        "Quadrature":
            r"Subroutine (ii): solution of eigenvalue problem",
        "Integrate":
            r"Subroutine (iii): closure of moment equations",
}

# Target directory for output of figures
target_dir = "fig/jcp"

# Output format
output_format = ".pdf"
