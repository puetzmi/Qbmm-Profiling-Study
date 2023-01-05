import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import plot_tools

data_dir = sys.argv[1]

fields = {}
ext = ".dat"
s = "nmom"
for f in os.listdir(data_dir):
    # Only allow files with specified file extension
    if f[-len(ext):] == ext:
        f1 = f.replace(ext, "")
        idx = f1.find(s)
        nmom = int(f1[idx+len(s):])
        try:
            fields[nmom]
        except KeyError:
            fields[nmom] = {}

        fpath = os.path.join(data_dir, f)

        # only analyze 1D-fields (of course, this takes a bit longer but is
        # less tedious than defining the fields of interest manually)
        df = pd.read_csv(fpath, comment='#', delim_whitespace=True, nrows=1)
        if df.shape[1] == 1:
            df = pd.read_csv(fpath, comment='#', delim_whitespace=True, header=None)
            fields[nmom][f1[:idx-1]] = df

n_moments = list(set(fields.keys()))
field_names = list(fields[n_moments[0]].keys())

for field_name in field_names:
    print(field_name)
    fig, ax = plot_tools.figure()
    for nmom in n_moments:
        y = fields[nmom][field_name].values
        y[y==0] = np.min(y[y!=0])
        hist, bins = np.histogram(np.log10(y), bins='auto')
        x = 0.5*(bins[:-1] + bins[1:])
        plt.plot(x, hist, label=str(nmom))
    plt.legend()
    plt.show()
