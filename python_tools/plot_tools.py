"""!
@file plot_tools.py
@author M. Puetz
@brief This Python module loads and applies frequently reused plot parameters and provides some convenience functions.

"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.transforms import Bbox
from itertools import chain as itchain
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm


# Set figure, font sizes etc.
mpl_params = {'legend.fontsize': 8,
          'figure.figsize': (8, 3.2),
         'axes.labelsize': 10,
         'axes.titlesize': 12,
         'xtick.labelsize': 8,
         'ytick.labelsize': 8,
         'grid.linestyle': ':',
         'grid.linewidth': 0.5,
         'axes.formatter.limits': [-2,2],
         'axes.formatter.use_mathtext': True,
         'lines.markerfacecolor': 'none',
         'lines.markersize': 4.5,
         'lines.linewidth': 1.,
 }
plt.rcParams.update(mpl_params)

# Line property cycles
_default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
get_colorcycle = lambda: cycle(_default_colors)
_basic_linestyles = ["-","--","-.",":", (0, (3, 1, 1, 1))]
get_lscycle = lambda: cycle(_basic_linestyles)
_basic_markers = ['o', 's', 'v', '^', 'D', 'h']
get_markercycle = lambda: cycle(_basic_markers)


def figure(shrink_axes=None):
    """!
    @brief Convenience function to create figure with single coordinate system.

    @param shrink_axes float: If provided the coordinate system is shrunk horizontally by the given relative length with respect to the figure width.

    @return fig Figure: The created `Figure` object.
    @return ax AxesSubplot: The created axes object.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if shrink_axes:
        fig.subplots_adjust(left=fig.subplotpars.left + 0.5*shrink_axes)
        fig.subplots_adjust(right=fig.subplotpars.right - 0.5*shrink_axes)

    return fig, ax


def figure_1by2(ax_keys=['l','r']):
    """!
    @brief Convenience function to create 1x2 figure with initialized side-by-side axes.

    @param ax_keys iterable: Iterable object containing the keys to address the axes (optional default: ['l', 'r']).

    @return fig Figure: The created `Figure` object.
    @return ax AxesSubplot: Dictionary containing the created `AxesSubplot` objects.

    """
    fig = plt.figure()
    ax = {key: fig.add_subplot(121 + i) for i,key in enumerate(ax_keys)}
    return fig, ax


def figure_2by2(ax_keys=['ul','ur','ll','lr']):
    """!
    @brief Convenience function to create 1x2 figure with initialized side-by-side axes.

    @param ax_keys iterable: Iterable object containing the keys to address the axes (optional default: ['l', 'r']).

    @return fig Figure: The created `Figure` object.
    @return ax AxesSubplot: Dictionary containing the created `AxesSubplot` objects.

    """
    fig = plt.figure()
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height*2)
    ax = {key: fig.add_subplot(221 + i) for i,key in enumerate(ax_keys)}
    return fig, ax


def figure_legend(fig, ax, loc='bottom', ncol='auto', frameon=False, adjust_axes=False, \
        layout='horizontal', linewidth=None, rel_width=None, vspace=2, **kwargs):
    """!
    @brief Create a legend above or below the coordinate system(s) with the legend handles of the given axis object.

    @param fig Figure: The matplotlib `Figure` object where the legend is to be created
    @param ax AxesSubplot: An `AxesSubplot` object with the legend handles and labels.
    @param loc str: Location of the legend, must be one of 'top' or 'bottom' (option, default: 'bottom').
    @param ncol int or 'auto': Number of columns in the legend, must be a positive integer or 'auto',
        see the comments for information on the determination of `ncol` in case of the latter (optional, default: 'auto').
    @param frameon bool: Show legend frame (optional, default: `False`).
    @param adjust_axes bool: If `True` a potential overlap of the legend and axes is corrected by moving the coordinate system(s), (optional, default: `False`).
    @param layout str: Indicates if the legend's elements are ordered horizontally or vertically (optional, default: 'horizontal').
    @param linewidth float: Uniform linewidth if desired (optional).
    @param rel_width float: Optional parameter that represents the relative width of the legend with respect to the figure width (only used is if `ncol==auto`, if not provided the width of the coordinate system is taken).
    @param vspace int: Vertical space between legend and coordinate system in pixels, which is only used if `adjust_axes==True` (optional, default: 2).
    @param kwargs: Additional parameters passed to matplotlib's `legend` method.

    @return legend Legend The created matplotlib `Legend` object.

    """
    # Check if `ncol` is valid
    ncol_int = False
    ncol_auto = False
    try:
        ncol_int = int(ncol) > 0
        ncol_auto = False
    except ValueError:
        ncol_int = False
        ncol_auto = ncol == 'auto'
    if not (ncol_int or ncol_auto):
        msg = "The number of columns `ncol` must be a positive integer or 'auto'."
        raise ValueError(msg)

    handles, labels = ax.get_legend_handles_labels()
    renderer = fig.canvas.get_renderer()
    fig_width, fig_height = fig.canvas.get_width_height()

    # Execute this if `ncol` is an integer; even if `ncol=='auto'` this is always the case in the recursive calls.
    if ncol_int:

        # Check if `layout` is valid and rearrange legend handles and labels accordingly
        if layout == 'horizontal':
            flip = lambda items, ncols: itchain(*[items[i::ncol] for i in range(ncols)])
            handles = flip(handles, ncol)
            labels = flip(labels, ncol)
        elif layout != 'vertical':
            msg = "The parameter `layout` must be either 'horizontal' or 'vertical'."
            raise ValueError(msg)

        # Create legend at specified position
        if loc == 'top':
            legend = fig.legend(handles, labels, loc='upper center', ncol=ncol, frameon=True, **kwargs)
        elif loc == 'bottom':
            legend = fig.legend(handles, labels, loc='lower center', ncol=ncol, frameon=True, **kwargs)
        else:
            msg = "The parameter `loc` must be either 'bottom' or 'top'."

        # Adjust linewidth to uniform value if specified
        if linewidth is not None:
            for legobj in legend.legendHandles:
                legobj.set_linewidth(linewidth)

        # If `adjust_axes==True` detect overlap between coordinate systems and legend and
        # move the coordinate systems accordingly
        if adjust_axes:
            axes_bbox = Bbox.union([a.get_tightbbox(renderer) for a in fig.get_axes()])
            legend_bbox = legend.get_tightbbox(renderer)
            if Bbox.intersection(axes_bbox, legend_bbox):
                if loc == 'bottom':
                    vertical_shift = legend_bbox.max[1] - axes_bbox.min[1] + vspace
                    vertical_shift /= fig_height
                    bottom = fig.subplotpars.bottom + vertical_shift
                    fig.subplots_adjust(bottom=bottom)
                else:
                    vertical_shift = legend_bbox.min[1] - axes_bbox.max[1] - vspace
                    vertical_shift /= fig_height
                    top = fig.subplotpars.bottom + vertical_shift
                    fig.subplots_adjust(top=top)

        # Done
        legend.set_frame_on(frameon)
        return legend


    # If `ncol=='auto'`, set the number of columns such that the legend has maximal
    # horizontal extent without exceeding the width of the coordinate systems (excluding
    # ticklabels). This is done by calling this function recursively with increasing
    # number of columns until the maximum number is found.
    axes_bbox = Bbox.union([a.get_window_extent() for a in fig.get_axes()])
    target_width = axes_bbox.width if rel_width is None else rel_width*fig_width
    ncol = 1
    larger = False
    while not larger and ncol <= len(handles):
        ncol += 1
        #legend = figure_legend(fig=fig, ax=ax, loc=loc, ncol=ncol, frameon=True, adjust_axes=False)
        legend = figure_legend(fig=fig, ax=ax, loc=loc, ncol=ncol, frameon=True, adjust_axes=False, \
            layout=layout, linewidth=linewidth, rel_width=rel_width, vspace=vspace)
        extent = legend.get_window_extent(renderer)
        larger = extent.width > target_width
        legend.remove()
    ncol -= 1

    return figure_legend(fig=fig, ax=ax, loc=loc, ncol=ncol, frameon=frameon, adjust_axes=adjust_axes, \
        layout=layout, linewidth=linewidth, rel_width=rel_width, vspace=vspace)


def linestyle_legend(ax, linestyles, labels, loc='best', lw=1., color=[0.4]*3, layout='vertical', ncol=None, **kwargs):
    """!
    @brief Create custom legend of lines with different linestyles and uniform line color in given coordinate system.

    @param ax AxesSubplot: An `AxesSubplot` object where the legend is to be created.
    @param linestyles iterable: List of any other iterable object containing linestyles, e.g. `['-', '--']`.
    @param loc str or int: Location of the legend, see e.g. the documentation of the `matplotlib.pyplot.plot` function for additional information (optional, default: 'best').
    @param lw float: Uniform line width (optional, default: 1).
    @param color : Any object that can be interpreted as a color by matplotlib (known colow string, color object, RGB values, etc.).
    @param layout str: Indicates horizontal or vertical layout (optional, default: 'vertical).
    @param ncol int: Number of columns (optional, default is 1 or the length of `labels` depending on `layout`).
    @param kwargs: Additional parameters passed to matplotlib's `legend` method.

    @return legend Legend The created matplotlib `Legend` object.

    """
    lines = [Line2D([0], [0], color=color, lw=lw, ls=ls) for ls in linestyles]

    if layout == 'horizontal':
        if ncol is None:
            ncol = len(labels)
        flip = lambda items, ncols: itchain(*[items[i::ncol] for i in range(ncols)])
        lines = flip(lines, ncol)
        labels = flip(labels, ncol)
    elif layout != 'vertical':
        if ncol is None:
            ncol = 1
        msg = "The parameter `layout` must be either 'horizontal' or 'vertical'."
        raise ValueError(msg)

    return ax.legend(lines, labels, loc=loc, ncol=ncol, **kwargs)


def contour_hist2d(ax, x, y, bins=(10,10), x_scale='linear', y_scale='linear', z_scale='linear', z_func=lambda z: z, normalize=False, levels=10, filled=True, return_hist_data=False, **kwargs):
    """!
    @brief Create contour plot

    @param ax AxesSubplot: An `AxesSubplot` object to create the contour plot.
    @param x array_like: Array of x-data.
    @param y array_like: Array of y-data.
    @param bins array_like: Array-like object with size 2 containing the number of bins in x- and y-direction, respectively.
    @param x_scale str: Scale of the x-axis, must be either 'linear' or 'log' (optional, default: 'linear').
    @param y_scale str: Scale of the y-axis, must be either 'linear' or 'log' (optional, default: 'linear').
    @param z_scale str: Scale of the z-azis, must be either 'linear' or 'log' (optional, default: 'linear').
    @param z_func callable: Callable object to apply histogram(z)-data after computation, e.g. to replace zeros when `z_scale=True` (optional).
    @param normalize bool: If `normalize==True` the histogram data is normalized by division by the number of samples (optional, default: False).
    @param levels int or array_like: The contour level boundary values in increasing order or the number of contour levels (optional, default: `levels=10`).
    @param filled bool: If `filled==True` the space between contour lines are filled (optional, default: `True`).
    @param return_hist_data bool: If `return_hist_data==True` the histogramm data, i.e. histogram values and bin edges, are also returned from the function (optional, default: `False`).
    @param kwargs: Additional keyword arguments passed to the method called to generate contour plot.

    @return contour QuadContourSet: The object generated by Matplotlib.
    @return z array: Only if `return_hist_data==True`: 2D-array containing the histogram values.
    @return x_bin_edges array: Only if `return_hist_data==True`: Bin edges of the x-data.
    @return y_bin_edges array: Only if `return_hist_data==True`: Bin edges of the y-data.

    """
    # Raise error if scale is neither 'log' nor 'linear'
    def _invalid_scale_error(val, varname):
        msg = "Got invalid parameter '{0:s}' for parameter `{1:s}_scale`, which must be either 'linear' or 'log'.".format(val, varname)
        raise ValueError(msg)

    # Check scale for validity / scale x-/y-data
    xy_scale = {'x': x_scale, 'y': y_scale}
    xy_data = {'x': x, 'y': y}
    for key, scale in xy_scale.items():
        if scale == 'log':
            xy_data[key] = np.log10(xy_data[key])
        elif scale != 'linear':
            _invalid_scale_error(scale, key)

    # Compute histogram data z
    z, x_bin_edges, y_bin_edges = np.histogram2d(xy_data['x'], xy_data['y'], bins=bins)

    # Check scale for validity / scale z-data
    z = z_func(z)
    if normalize:
        z /= len(x)

    # Compute bin centers and create contour plot
    bin_centers = lambda edges: 0.5*(edges[1:] + edges[:-1])
    contour_func = ax.contourf if filled else ax.contour
    if x_scale == 'log':
        x_bin_edges = 10**x_bin_edges
    if y_scale == 'log':
        y_bin_edges = 10**y_bin_edges
    x_grid, y_grid = np.meshgrid(bin_centers(x_bin_edges), bin_centers(y_bin_edges), indexing='ij')
    if z_scale == 'log':
        try:
            len(levels)
        except TypeError:
            min_ = np.log10(np.min(z))
            max_ = np.log10(np.max(z))
            levels_ = np.logspace(min_, max_, levels)
            contour = contour_func(x_grid, y_grid, z, norm=LogNorm(), levels=levels_, **kwargs)
    else:
        contour = contour_func(x_grid, y_grid, z, levels=levels, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    if not return_hist_data:
        return contour
    return contour, z, x_bin_edges, y_bin_edges


if __name__ == "__main__":
    # A simple test of default parameters and methods
    import numpy as np
    n = 16
    x = np.linspace(-1.2, 1.2, 1000)
    fig, ax = figure_1by2()
    ls_cycle = get_lscycle()
    c_cycle = get_colorcycle()
    ls = [next(ls_cycle) for _ in range(2)]
    for i in range(n):
        c = next(c_cycle)
        ax['l'].plot(x, x**i, c=c, label="$x^{{{0:d}}}$".format(i), ls=ls[i%2])
        ax['r'].semilogy(x, x**i, ls=ls[i%2], c=c)
    ax['l'].set_xlabel('x')
    figure_legend(fig, ax['l'], ncol='auto', adjust_axes=True)
    linestyle_legend(ax['r'], ls, ["{0:s} exponent".format(s) for s in ['even', 'odd']])
    for a in ax.values():
        a.grid(which='both')
    plt.show()
