#!/usr/bin/env python3
"""!
@file postprocess.py
@author M. Puetz
@brief This script generates plots of core inversion benchmark results given one or multiple source directories and a target directory.

@param source-dir The source directory or a list of source directories in Python syntax

@par Examples
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import plot_tools
import pandas as pd


def postprocess():
    """!
    @brief Main function.

    """
    try:
        import postprocess_config as config
    except ModuleNotFoundError as err:
        err.msg = "A 'postprocess_config.py' must be provided to run postprocessing script, see this script's documentation for more information."
        raise err

    ## READ PARAMETERS ##
    # Source directories
    source_dirs = config.source_dirs

    # Main directory containing the data for full analysis (correlations, errors, etc.)
    try:
        main_dir = config.main_dir
    except AttributeError as err:
        if len(source_dirs) == 1:
            main_dir = source_dirs[0]
        else:
            msg = "The configuration file 'postprocess_config.py' must specify the main source directory " \
                "for analysis as the attribute `main_dir`, which is not defined."
            raise AttributeError(msg)
    if main_dir not in source_dirs:
        raise ValueError("The main directory `main_dir='{0:s}` must be in `source_dirs` but it is not.".format(main_dir))

    # Labels corresponging to directories
    try:
        labels = config.labels
    except AttributeError:  # if not defined use directory names
        labels = [sd for sd in source_dirs]

    if len(labels) != len(source_dirs):
        msg = "The `labels` parameter must have the same length as `source_dirs`."
        raise ValueError(msg)

    # Directory containing the original input data for benchmark
    data_dir = config.data_dir

    # Components of input file names
    data_file_pattern = config.data_file_pattern
    try:
        infile_prefix = config.infile_prefix
    except AttributeError:
        infile_prefix = ""
    try:
        infile_suffix = config.infile_suffix
    except AttributeError:
        infile_suffix = ""

    # Target directory, default is current working directory if none is given
    try:
        target_dir = config.target_dir
    except AttributeError:
        target_dir = os.getcwd()

    # Optional dictionary that maps the column headings in input data files representing
    # the configuration to the labels used in plots
    try:
        config_to_label_map = config.config_to_label_map
    except AttributeError:
        config_to_label_map = None

    # Optional dictionary that maps the error keys in the data files representing to strings used as labels in plots
    try:
        error_to_label_map = config.error_to_label_map
    except AttributeError:
        error_to_label_map = None

    # Output format of figures (default: png)
    try:
        output_format = config.output_format
    except AttributeError:
        output_format = ".png"

    # Number of bins in 2D-histogram
    try:
        n_hist_bins = config.n_hist_bins
    except AttributeError:
        n_hist_bins = (20, 20)

    # Number of levels in contour plots
    try:
        n_contour_levels = config.n_contour_levels
    except AttributeError:
        n_contour_levels = 20

    # Color map
    try:
        color_map = config.color_map
    except AttributeError:
        color_map = "coolwarm"

    # Indicate whether or not to plot histograms of errors (may take a long time)
    try:
        plot_histograms = config.plot_histograms
    except AttributeError:
        plot_histograms = False


    ## LOAD DATA ##
    df_all = {}
    summary = {}
    for isrc, source_dir in enumerate(source_dirs):
        # List all files in the source directory and determine numbers of moments based on that based on that
        data_files = {}
        summary_files = {}
        for f in os.listdir(source_dir):
            is_data_file = f.find(infile_prefix + data_file_pattern) == 0 \
                and f[-len(infile_suffix):] == infile_suffix \
                and os.path.isfile(os.path.join(source_dir, f))
            if is_data_file:
                n_moments = int(f[len(infile_prefix + data_file_pattern):f.find(infile_suffix)])
                data_files[n_moments] = f
                summary_files[n_moments] = f.replace("data", "summary")
        n_moments = list(set(data_files.keys()))

        df = []
        # Read all data and summary files in all source directories
        for i,n_mom in enumerate(n_moments):

            data_file = os.path.join(source_dir, data_files[n_mom])
            summary_file = os.path.join(source_dir, summary_files[n_mom])
            print("Reading input file {0:s}".format(data_file))

            # Store column headings and corresponging indices
            df.append(pd.read_csv(data_file, comment='#', delim_whitespace=True))
            df[-1]["nMoments"] = n_mom*np.ones(len(df[-1]), dtype=int)
            df[-1]["ConfigNo"] = df[-1]["CaseNo"].astype(int)

            # Read summary file
            with open(summary_file, 'r') as fi:
                lines = [line.split() for line in fi.readlines()]
                lines = [line for line in lines[1:] if line and line[0] != '#']
                s = {int(line[0]): line[1] for line in lines[1:]}
            if not summary:
                summary = s.copy()
            # Make sure the summary dictionary does not differ from the one in the previous iteration (something is wrong if it does)
            if summary != s:
                msg = "The summary of configurations must be the same for all input files, but " \
                    "the setup in '{0:s}' differs from that in {1:s}".format(summary_files[i], summary_files[i-1])
                raise RuntimeError(msg)

        df = pd.concat(df, axis=0, ignore_index=True)
        df_all[labels[isrc]] = df.copy()


    ## PLOT AVERAGE CPU TIMES VS. NUMBER OF MOMENTS ##
    ls_cycle = plot_tools.get_lscycle()
    linestyles = {label: next(ls_cycle) for label in labels}
    color_cycle = plot_tools.get_colorcycle()
    colors = {i: next(color_cycle) for i in np.unique(df["ConfigNo"])}
    linewidth = 1.

    fig, ax = plot_tools.figure(shrink_axes=0.2)

    # Use column headings for configuration labels if no map is provided
    if not config_to_label_map:
        config_to_label_map = {x: x for x in summary.values()}

    add_label = True
    for label in labels:
        # Compute mean CPU time
        cpu_times = df_all[label].groupby(["nMoments", "ConfigNo"])["ComputingTime"]
        cpu_times_mean = cpu_times.mean()

        # Plot CPU times vs. number of moments
        for idx,name in summary.items():
            lbl = config_to_label_map[name] if add_label else None
            ls = linestyles[label] if len(labels) > 1 else next(ls_cycle)
            cpu_times_mean[:idx]
            colors[idx]
            ax.semilogy(n_moments, cpu_times_mean[:,idx], label=lbl, c=colors[idx], ls=ls, lw=linewidth)

        # Add labels only during the first iteration since the configurations are repeated
        add_label = False

    if len(source_dirs) > 1:
        plot_tools.linestyle_legend(ax, linestyles=linestyles.values(), labels=labels, \
            lw=linewidth, layout='vertical', ncol=2)

    ax.grid(which='both')
    ax.set_xlabel("Number of moments")
    ax.set_ylabel("CPU time [s]")
    plot_tools.figure_legend(fig, ax, adjust_axes=True, linewidth=linewidth, vspace=4, rel_width=0.9)

    # Create target directory if it does not exist
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    elif not os.path.isdir(target_dir):
        msg = "The specified target directory name '{0:s}' exists but is not a directory.".format(target_dir)
        raise OSError(msg)

    fig.savefig(os.path.join(target_dir, "cpu_times-n_mom{0:s}".format(output_format)))
    plt.close(fig)


    ## LOAD DATA FROM DATA DIRECTORY ##
    # quantities characterizing distance from moment space boundary
    quantities = ["sigma-min", "regularity-radius", "hankel-determinant", "beta-coeffs", "mom2nm2-boundary-dist"]

    # Names of quantities (used as labels)
    quantity_names = [r"$\sigma_{min}$", r"$r_{reg}$", r"$|H|$", r"$\beta_{min}$", "my new quantity"]
    quantity_names = {quantities[i]: qn for i,qn in enumerate(quantity_names)}

    # functions to apply to input data
    funcs = {quantity: lambda x, _: x for quantity in quantities}
    funcs["hankel-determinant"] = lambda x, nmom: x**(2/nmom)
    funcs["beta-coeffs"] = lambda x, _: np.min(x[:,1:], axis=1)
    assert(len(quantities) == len(funcs))   # make sure no additional entries have been created accidentally

    # Read all original input data files
    main_key = labels[source_dirs.index(main_dir)]
    df_main = df_all[main_key]  # now consider only the specified main directory
    data_dir_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    df_orig = []            # original input data in `data_dir`
    for quantity in quantities:
        df = []
        for n_mom in n_moments:
            pattern = "{0:s}_nmom{1:d}".format(quantity, n_mom)
            infile = [f for f in data_dir_files if f.find(pattern) > -1]
            if len(infile) > 1:
                msg = "Provided input data is ambiguous: There are multiple files in " \
                    "'{0:s}' containing the patterns '{1:s}' and 'nmom{2:d}'".format(data_dir, quantity, n_mom)
                raise RuntimeError(msg)
            try:
                infile = os.path.join(data_dir, infile[0])
            except IndexError:
                msg = "The directory '{0:s}' does not contain any files matching the " \
                    "patterns '{1:s}' and 'nmom{2:d}'".format(data_dir, quantity, n_mom)
                raise RuntimeError(msg)

            tmp = pd.read_csv(infile, comment='#', delim_whitespace=True, header=None)
            df.append(pd.DataFrame(funcs[quantity](tmp.values, n_mom), columns=[quantity]))

        df_orig.append(pd.concat(df, axis=0))

    df_orig = pd.concat(df_orig, axis=1)
    df_orig.reset_index(drop=True, inplace=True)


    ## PLOT HISTOGRAMS OF ERRORS ##
    # Plot all columns whose keys contain 'error'
    error_keys = [col for col in df_main.columns if col.lower().find("error") > -1]
    ls_cycle = plot_tools.get_lscycle()
    color_cycle = plot_tools.get_colorcycle()
    gmean = {}
    gstd = {}
    for idx,name in summary.items():
        df = pd.concat([df_main[df_main["ConfigNo"] == idx].reset_index(drop=True), df_orig], axis=1)
        assert(np.all(np.isfinite(df.values)))
        gmean[name] = {}
        for n_mom in n_moments:
            df_ = df[df["nMoments"]==n_mom][error_keys + quantities]
            gmean[name][n_mom] = {}
            for error_key in error_keys:
                gmean[name][n_mom][error_key] = {}
                y = np.maximum(df_[error_key], np.finfo(df_[error_key].dtype).eps)
                for quantity in quantities:
                    x = df_[quantity]
                    x[x==0] = np.min(x[x!=0])   # this case should be extremely rare and thus not make any visible difference

                    if plot_histograms:
                        fig, ax = plt.subplots()
                        z_func = lambda z: z + 0.1*np.min(z[z!=0])*(z==0).astype(int)   # replace zeros before taking logarithm for histogram plot
                        contour, hist, x_bins, y_bins = plot_tools.contour_hist2d(ax, x, y, \
                            x_scale='log', y_scale='log', z_scale='log', cmap=color_map, \
                            levels=n_contour_levels, bins=n_hist_bins, z_func=z_func, \
                            return_hist_data=True, normalize=True)
                        hist /= len(x)  # normalize
                    else:
                        x_bins = 2**np.linspace(
                            np.log2(np.min(x)), np.log2(np.max(x)), n_hist_bins[0] + 1
                            )

                    x_data = 0.5*(x_bins[1:] + x_bins[:-1]) 
                    x_bins[0] = 0.
                    x_bins[-1] *= 1 + np.finfo(x_bins.dtype).eps
                    y_gmean = np.zeros_like(x_data) 
                    for i in range(len(x_data)):
                        idx = (x >= x_bins[i]) & (x < x_bins[i+1])
                        y_gmean[i] = 2**np.mean(np.log2(y[idx]))
                    gmean[name][n_mom][error_key][quantity] = (x_data, np.array(y_gmean))
                    
                    if plot_histograms:
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        not_nan = ~np.isnan(y_gmean) # may happen in empty bins
                        ax.loglog(x_data[not_nan], y_gmean[not_nan], color='k', 
                            marker='o', label="Geometric mean")
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)
                        ax.set_xlabel(quantity_names[quantity])
                        ax.set_ylabel(error_to_label_map[error_key])
                        ax.set_title("nmom = {0:d}".format(n_mom))
                        ax.legend(loc='lower left')
                        left = fig.subplotpars.left
                        right = fig.subplotpars.right
                        fig.tight_layout()
                        fig.subplots_adjust(right=right, left=left)
                        fig.colorbar(contour)
                        plt.show()
                    

    for error_key in error_keys:
        for quantity in quantities:
            for n_mom in n_moments:
                ls_cycle = plot_tools.get_lscycle()
                color_cycle = plot_tools.get_colorcycle()
                fig, ax = plot_tools.figure(shrink_axes=0.2)
                for idx, name in summary.items():
                    x_data, err_data = gmean[name][n_mom][error_key][quantity]
                    idx = ~np.isnan(err_data)
                    ax.errorbar(x_data[idx], err_data[idx], lw=linewidth, ls=next(ls_cycle), \
                        c=next(color_cycle), label=config_to_label_map[name])
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(which='both')
                ax.set_xlabel(quantity_names[quantity])
                ax.set_ylabel(error_to_label_map[error_key])
                ax.set_title("nmom = {0:d}".format(n_mom))
                ax.legend()
                ax_corners = ax.get_tightbbox(fig.canvas.get_renderer()).corners()
                fig_height = fig.canvas.get_width_height()[1]
                bottom = ax_corners[0,1]/fig_height
                fig.subplots_adjust(bottom=0.2)
                plt.show()
                #fig.tight_layout()
                #plot_tools.figure_legend(fig, ax, adjust_axes=True, linewidth=linewidth, vspace=4, rel_width=0.9)
                #plot_tools.figure_legend(fig, ax, adjust_axes=True, linewidth=linewidth, vspace=4, rel_width=0.9)


if __name__ == "__main__":
    postprocess()