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

    # Number of bins in 2D-histogram (currently unused)
    try:
        n_hist_bins = config.n_hist_bins
    except AttributeError:
        n_hist_bins = (20, 20)

    # Number of levels in contour plots (currently unused)
    try:
        n_contour_levels = config.n_contour_levels
    except AttributeError:
        n_contour_levels = 20

    # Color map
    try:
        color_map = config.color_map
    except AttributeError:
        color_map = "coolwarm"

    # Indicate whether or not to plot histograms of errors, which may take some time
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
    print("Plotting average CPU times...")
    # Create target directory if it does not exist
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    elif not os.path.isdir(target_dir):
        msg = "The specified target directory name '{0:s}' exists but is not a directory.".format(target_dir)
        raise OSError(msg)
    # Plot twice, once for all source directories (if > 1)
    # and once for specified main source directory
    source_dirs_all = source_dirs.copy()
    labels_all = labels.copy()
    end = 1 if len(source_dirs) == 1 else 2
    for _ in range(end):
        ls_cycle = plot_tools.get_lscycle()
        linestyles = {label: next(ls_cycle) for label in labels}
        ls_cycle = plot_tools.get_lscycle() # reset
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
            n_moment_sets = cpu_times.size().values[0]  # equal in all groups
            cpu_times_sum = cpu_times.sum()
            inversions_per_second = n_moment_sets/cpu_times_sum

            # Plot CPU times vs. number of moments
            for idx,name in summary.items():
                lbl = config_to_label_map[name] if add_label else None
                ls = linestyles[label] if len(labels) > 1 else next(ls_cycle)
                ax.semilogy(n_moments, inversions_per_second[:,idx],
                        label=lbl, c=colors[idx], ls=ls, lw=linewidth)

            # Add labels only during the first iteration since the configurations are repeated
            add_label = False

        if len(source_dirs) > 1:
            plot_tools.linestyle_legend(ax, linestyles=linestyles.values(), labels=labels, \
                lw=linewidth, layout='vertical', ncol=2)

        ax.grid(which='both')
        ax.set_xlabel("Number of moments")
        ax.set_ylabel("Inversions per second")
        plot_tools.figure_legend(fig, ax, adjust_axes=True, linewidth=linewidth, vspace=4, rel_width=0.9)

        target_filename = os.path.join(target_dir, "cpu-times_nmom{0:s}".format(output_format))
        if len(source_dirs) > 1:
            target_filename = \
                target_filename.replace(output_format, "__all{0:s}".format(output_format))
        print("\t{0:s}".format(target_filename))
        fig.savefig(target_filename)
        plt.close(fig)
        labels = [labels[source_dirs.index(main_dir)]]
        source_dirs = [main_dir]

    labels = labels_all.copy()
    source_dirs = source_dirs_all.copy()


    ## LOAD DATA FROM DATA DIRECTORY ##
    # quantities characterizing distance from moment space boundary
    boundary_dist_quantities = config.boundary_dist_quantities
    # Names of quantities (used as labels)
    boundary_dist_quantity_names = {boundary_dist_quantities[i]:
        qn for i,qn in enumerate(config.boundary_dist_quantity_names)}

    # functions to apply to input data
    funcs = config.funcs
    assert(len(boundary_dist_quantities) == len(funcs))   # make sure no additional entries have been created accidentally

    # Read all original input data files
    main_key = labels[source_dirs.index(main_dir)]
    df_main = df_all[main_key]  # now consider only the specified main directory
    data_dir_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    df_orig = []            # original input data in `data_dir`
    for quantity in boundary_dist_quantities:
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
    print("Computing {0:s}histograms...".format("and plotting " if plot_histograms else ""))
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
            df_ = df[df["nMoments"]==n_mom][error_keys + boundary_dist_quantities]
            gmean[name][n_mom] = {}
            for error_key in error_keys:
                gmean[name][n_mom][error_key] = {}
                y = np.maximum(df_[error_key], np.finfo(df_[error_key].dtype).eps)
                for quantity in boundary_dist_quantities:
                    x = df_[quantity]
                    x[x==0] = np.min(x[x!=0])   # this case should be extremely rare and thus not make any visible difference

                    if plot_histograms:
                        fig, ax = plt.subplots()
                        z_func = lambda z: z + 0.1*np.min(z[z!=0])*(z==0).astype(int)   # replace zeros before taking logarithm for histogram plot
                        """
                        contour, hist, x_bins, y_bins = plot_tools.contour_hist2d(ax, x, y, \
                            x_scale='log', y_scale='log', z_scale='log', cmap=color_map, \
                            levels=n_contour_levels, bins=n_hist_bins, z_func=z_func, \
                            return_hist_data=True, normalize=True)
                        """
                        log10_x = np.log10(x)
                        x_bins = np.logspace(np.min(log10_x), np.max(log10_x),
                                n_hist_bins[0] + 1)
                        log10_y = np.log10(y)
                        y_bins = np.logspace(np.min(log10_y), np.max(log10_y),
                                n_hist_bins[1] + 1)
                        hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
                        hist /= len(x)  # normalize

                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        contour = ax.hexbin(x, y, xscale='log', yscale='log',
                                bins='log', cmap=color_map)
                        #plt.show()
                        #plt.close(fig1)
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
                            marker='o', markersize=4, label="Conditional geometric mean")
                        ax.set_xlim(x_data[not_nan][0], x_data[not_nan][-1])
                        ax.set_ylim(ylim)
                        ax.set_xlabel(boundary_dist_quantity_names[quantity])
                        ax.set_ylabel(error_to_label_map[error_key])
                        ax.legend(loc='lower left')
                        ax.grid(which='both')
                        left = fig.subplotpars.left
                        right = fig.subplotpars.right
                        fig.tight_layout()
                        fig.subplots_adjust(right=right, left=left)
                        fig.colorbar(contour)
                        target_filename = os.path.join(target_dir, \
                                "hist__{0:s}_{1:s}__nmom{2:d}__{3:s}{4:s}".format( \
                                quantity, \
                                error_key, \
                                n_mom, \
                                name, \
                                output_format))
                        print("\n\t{0:s}".format(target_filename))
                        fig.savefig(target_filename)
                        plt.close(fig)

    print("Plotting mean errors...")
    for error_key in error_keys:
        for quantity in boundary_dist_quantities:
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
                ax.set_xlabel(boundary_dist_quantity_names[quantity])
                ax.set_ylabel(error_to_label_map[error_key])
                ax.legend()
                ax_corners = ax.get_tightbbox(fig.canvas.get_renderer()).corners()
                fig_height = fig.canvas.get_width_height()[1]
                bottom = ax_corners[0,1]/fig_height
                fig.subplots_adjust(bottom=0.2)
                target_filename = os.path.join(target_dir, \
                        "mean-errors__{0:s}_{1:s}__nmom{2:d}{3:s}".format( \
                        quantity, \
                        error_key, \
                        n_mom, \
                        output_format))
                print("\n\t{0:s}".format(target_filename))
                fig.savefig(target_filename)
                plt.close(fig)
                #plt.show()
                #fig.tight_layout()
                #plot_tools.figure_legend(fig, ax, adjust_axes=True, linewidth=linewidth, vspace=4, rel_width=0.9)
                #plot_tools.figure_legend(fig, ax, adjust_axes=True, linewidth=linewidth, vspace=4, rel_width=0.9)


if __name__ == "__main__":
    postprocess()
