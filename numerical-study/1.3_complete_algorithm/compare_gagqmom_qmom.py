"""!
@author Michele Puetz
@brief Script for visual comparison of CPU times of GaG-QMOM and standard QMOM.

"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_tools


def main(config_module):
    """!
    @brief Main function.

    """
    plt.rcParams["figure.figsize"] = (4.2, 2.6)
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["axes.formatter.limits"] = [-2,3]
    plt.rcParams["axes.titlesize"] = 10

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


    df_group = df.groupby(["nMoments", "ConfigNo"])

    subroutine_to_label_map = {
            "CoreInversion":
                r"Moments $\rightarrow$ Jacobi matrix",
            "Quadrature":
                r"Jacobi matrix $\rightarrow$ quadrature",
            "Integrate":
                r"Moment closure",
    }
    subroutines = subroutine_to_label_map.keys()

    marker_cycle = plot_tools.get_markercycle()

    physical_setup = "FokkerPlanckConstantCd"
    inv_algorithm = "LqmdAlgorithm"
    qmom_types = ["QmomStd", "QmomGaG"]
    qmom_linestyles = ['--', '-']
    qmom_labels = ["Standard QMOM", "GaG-QMOM"]

    mean_cpu_times = {}

    for subroutine in subroutines:
        mean_cpu_times[subroutine] = {}
        for qmom_type in qmom_types:
            idx = [i for i in summary.keys() if qmom_type in summary[i]
                and physical_setup in summary[i]
                and inv_algorithm in summary[i]][0]

            cpu_times = df_group[subroutine]
            n_moment_sets = cpu_times.size().values[0]
            mean_cpu_times[subroutine][qmom_type] = (cpu_times.sum()/n_moment_sets)[:,idx].values


    fig, axs= plot_tools.figure_2by1(ax_keys=[0,1])
    ax = axs[0]
    color_cycle = plot_tools.get_colorcycle()
    marker_cycle = plot_tools.get_markercycle()
    total_time = {key: np.zeros(len(n_moments)) for key in qmom_types}
    for subroutine in subroutines:
        color = next(color_cycle)
        marker = next(marker_cycle)
        for i in range(len(qmom_types)):
            qmom_type = qmom_types[i]
            total_time[qmom_type] += mean_cpu_times[subroutine][qmom_type]
            ax.semilogy(n_moments, mean_cpu_times[subroutine][qmom_type],
                    label=subroutine_to_label_map[subroutine],
                    c=color, ls=qmom_linestyles[i], marker=marker)

    ax = axs[1]
    subroutine_to_label_map["Total"] = "Total"
    subroutines = subroutine_to_label_map.keys()
    mean_cpu_times["Total"] = {qmom_type: total_time[qmom_type] for qmom_type in qmom_types}
    color_cycle = plot_tools.get_colorcycle()
    marker_cycle = plot_tools.get_markercycle()
    for subroutine in subroutines:
        color = next(color_cycle)
        marker = next(marker_cycle)
        overhead = mean_cpu_times[subroutine]["QmomGaG"] \
                / mean_cpu_times[subroutine]["QmomStd"]
        ax.plot(n_moments, overhead,
                label=subroutine_to_label_map[subroutine],
                c=color, marker=marker)

    for ax in axs:
        axs[ax].grid(which='both')

    axs[1].set_xlabel("Number of moments")
    axs[0].set_ylabel("Computation time [s]")
    axs[1].set_ylabel("Rel. overhead")

    fig.subplots_adjust(left=0.15, right=0.97)
    fig.align_ylabels()
    plot_tools.figure_legend(fig, axs[1], adjust_axes=True, rel_width=0.98)
    plot_tools.linestyle_legend(axs[0], qmom_linestyles[::-1], qmom_labels[::-1], ncol=1)

    target_filename = os.path.join(target_dir, \
                          f"cpu_times_gagqmom_qmom__nmom{output_format}"
                    )
    fig.savefig(target_filename)


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config_module = importlib.import_module(config_file.replace('.py',''))
    except IndexError:
        try:
            import postprocess_config_thesis as config_module
        except ModuleNotFoundError as err:
            err.msg = "A configuration file must be provided to run postprocessing script."
            raise err
    main(config_module)
