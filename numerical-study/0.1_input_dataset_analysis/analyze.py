import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import plot_tools
import config
from scipy.integrate import simps

# directory containing the input data files, default: ../data
try:
    data_dir = sys.argv[1]
except IndexError:
    data_dir = "../data"

figure_output_dir = "./fig"

# extension of input data files
ext = ".dat"

# string indicating the number of moments in filenames
nmom_str = "nmom"

# Read all fields
fields_dict = config.fields
fields = {}
output_file_prefixes = {}
for f in os.listdir(data_dir):

    in_fields_dict = False
    for key in fields_dict.keys():
        in_fields_dict = in_fields_dict or f[:len(key)]==key
        if in_fields_dict:
            field_name = fields_dict[key][0]
            filter_function = fields_dict[key][1]
            output_file_prefixes[field_name] = \
                fields_dict[key][-1]
            break

    extension_matches = f[-len(ext):] == ext

    if not (extension_matches and in_fields_dict):
        continue

    f1 = f.replace(ext, "")
    idx = f1.find(nmom_str)
    nmom = int(f1[idx+len(nmom_str):])
    try:
        fields[nmom]
    except KeyError:
        fields[nmom] = {}

    fpath = os.path.join(data_dir, f)
    df = pd.read_csv(fpath, comment='#', delim_whitespace=True)

    fields[nmom][field_name] = filter_function(df.values)


n_moments = list(set(fields.keys()))
field_names = list(fields[n_moments[0]].keys())

color_cycle = plot_tools.get_colorcycle()
ls_cycle = plot_tools.get_lscycle()
# Compute histograms of all readquantities
for field_name in field_names:
    fig, ax = plot_tools.figure()
    arrow_begin = [1e300,-1e300]
    arrow_end = [-1e300,-1e300]
    for nmom in n_moments:
        y = fields[nmom][field_name]
        y[y==0] = np.min(y[y!=0])
        hist, bins = np.histogram(np.log10(y), 
                                  bins='auto', density=True)
        x = 0.5*(bins[:-1] + bins[1:])
        ax.plot(x, hist, label=f"{nmom}",
                c=next(color_cycle))
    ax.grid(which='both')
    ax.set_title(f"Distribution of {field_name}")
    ax.set_xlabel(f"$\mathrm{{log_{{10}}}}$({field_name})")
    ax.set_ylabel("Norm. number density")
    legend = ax.legend(title="Number of moments",
                       ncol=2)

    # "Vertical tight layout" using figure legend and
    # dummy axis
    ax_dummy = fig.add_subplot(111)
    ax_dummy.axis('off')
    plot_tools.figure_legend(fig, ax_dummy, ncol=1,
                             adjust_axes=True)
    
    try:    
        fig.savefig(f"{figure_output_dir}/hist_" \
            f"{output_file_prefixes[field_name]}.pdf")
    except FileNotFoundError:
        os.mkdir(figure_output_dir)
    plt.close(fig)


# Compute correlation coefficients for each combination
# of quantities
for i in range(len(field_names)):
    fig, ax = plot_tools.figure()
    for j in range(len(field_names)):
        if i==j:
            continue
        corrcoef =  [
                        np.corrcoef(
                            fields[nmom][field_names[i]].flatten(),
                            fields[nmom][field_names[j]].flatten()
                        )[0,1]
                        for nmom in n_moments
                    ]
        ax.plot(n_moments, corrcoef, label=field_names[j])

    ax.grid(which='both')
    ax.set_title(f"Correlation with {field_names[i]}")
    ax.set_xlabel("Number of moments")
    ax.set_ylabel("Correlation coefficient")
    legend = plot_tools.figure_legend(fig, ax, adjust_axes=True, fontsize=20)

    fig.savefig(f"{figure_output_dir}/corrcoef_" \
        f"{output_file_prefixes[field_names[i]]}.pdf")
    plt.close(fig)