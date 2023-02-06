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
import sys
import plot_tools
import pandas as pd


def postprocess_siam(config_module):
    """!
    @brief Main function.

    """
    config = config_module

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


    # Output format of figures (default: png)
    try:
        output_format = config.output_format
    except AttributeError:
        output_format = ".png"


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
                s = {int(line[0]): line[1:] for line in lines[1:]}
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
        os.makedirs(target_dir, exist_ok=True)
    elif not os.path.isdir(target_dir):
        msg = "The specified target directory name '{0:s}' exists but is not a directory.".format(target_dir)
        raise OSError(msg)

    # Get indices of relevant cases
    inv_type = config.inv_type
    qmom_type = config.qmom_type
    eigensolver_type = config.eigensolver_type
    linearsolver_type = config.linearsolver_type
    physical_setups = {}
    for idx, case_setup in summary.items():
        if inv_type in case_setup \
                and qmom_type in case_setup \
                and eigensolver_type in case_setup \
                and linearsolver_type in case_setup:
            physical_setups[idx] = case_setup[-2]


    # Plot
    fig = plt.figure()
    fig_size = fig.get_size_inches()
    fig_size[1] *= 1.2
    fig.set_size_inches(fig_size)
    axs = fig.subplots(ncols=len(physical_setups), sharey=True)
    axs = {idx: axs[i] for i,idx in enumerate(physical_setups)}
    plt.rcParams['lines.markerfacecolor'] = 'none'
    plt.rcParams['lines.markersize'] = 4.5
    plt.rcParams['lines.linewidth'] = 0.8

    df_group = df.groupby(["nMoments", "ConfigNo"])

    subroutine_to_label_map = config.subroutine_to_label_map
    subroutines = subroutine_to_label_map.keys()

    for setup_no, physical_setup in physical_setups.items():
        marker_cycle = plot_tools.get_markercycle()
        color_cycle = plot_tools.get_colorcycle()
        total_cpu_time = 0.
        for subroutine in subroutines:
            cpu_times = df_group[subroutine]
            n_moment_sets = cpu_times.size().values[0]
            mean_cpu_times = cpu_times.sum()/n_moment_sets
            total_cpu_time += mean_cpu_times[:,setup_no]
            axs[setup_no].semilogy(n_moments, mean_cpu_times[:,setup_no],
                    label=subroutine_to_label_map[subroutine],
                    c=next(color_cycle), marker=next(marker_cycle))

        axs[setup_no].set_title(config_to_label_map[physical_setup])
        axs[setup_no].semilogy(n_moments, total_cpu_time, label="Total",
                               c=next(color_cycle), marker=next(marker_cycle))
        axs[setup_no].set_xticks(n_moments[::2])
        axs[setup_no].set_xticks(n_moments[1::2], minor=True)
        axs[setup_no].grid(which='both')

    keys = list(axs.keys())
    axs[keys[0]].set_ylabel("Computation time [s]")
    axs[keys[1]].set_xlabel("Number of moments")
    fig.subplots_adjust(wspace=0, right=0.95)
    plot_tools.figure_legend(fig, axs[keys[0]], adjust_axes=True)

    target_filename = os.path.join(target_dir, \
                          f"mean_cpu_times__nmom{output_format}"
                    )
    fig.savefig(target_filename)
    plt.close(fig)


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config_module = importlib.import_module(config_file.replace('.py',''))
    except IndexError:
        try:
            import postprocess_config_siamjscicomp as config_module
        except ModuleNotFoundError as err:
            err.msg = "A configuration file must be provided to run postprocessing script."
            raise err
    df = postprocess_siam(config_module)
