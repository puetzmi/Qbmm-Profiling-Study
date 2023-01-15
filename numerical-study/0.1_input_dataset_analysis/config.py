"""
Configuration file for input data analysis, read by `analyze.py`.

"""
import numpy as np

# Fields to read from data directory.
#
# The key is the filename prefix and the value is a tuple
# containing 
# 1. the field name fo the plots' legends,
# 2. a filter function that is applied to the field
#   before processing,
# 3. prefix of the output file name.
fields = {
            "beta-coeffs":
                (r"$\mathrm{\beta_{min}}$",
                 lambda x: np.min(x, axis=-1),
                 "beta-min"),
            "hankel-determinant":
                (r"det$(\mathrm{\mathbf{M}_{2n-2}})$",
                 lambda x: x.flatten(),
                 "det-hankel"),
            "sigma-min":
                (r"$\mathrm{\sigma_{min}}$",
                 lambda x: x.flatten(),
                 "sigma-min"),
            "regularity-radius":
                (r"$\mathrm{r_{reg}}$",
                 lambda x: x.flatten(),
                 "r-reg"),
            "mom2nm2-boundary-dist":
                (r"$\mathrm{d_{2n-2}}$",
                 lambda x: x.flatten(),
                 "d-2nm2"),
}


