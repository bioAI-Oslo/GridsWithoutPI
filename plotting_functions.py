import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def multiimshow(zz, figsize=(1,1), normalize=True, add_colorbar=True, rect=(0,0,1,0.87), axes_pad=0.05, fig=None, **kwargs):
    # prepare figure
    ncols = int(np.ceil(np.sqrt(zz.shape[0])))
    nrows = int(round(np.sqrt(zz.shape[0])))
    fig = plt.figure(figsize=figsize) if fig is None else fig
    if add_colorbar and normalize:
        grid = ImageGrid(fig, rect=rect, nrows_ncols=(nrows, ncols), axes_pad=axes_pad, cbar_mode='single', cbar_location='right', cbar_pad=0.1, cbar_size='5%')
    else:
        grid = ImageGrid(fig, rect=rect, nrows_ncols=(nrows, ncols), axes_pad=axes_pad)
    vmin, vmax = (np.nanmin(zz), np.nanmax(zz)) if normalize else (None, None)
    kwargs['vmin'] = vmin if 'vmin' not in kwargs else kwargs['vmin']
    kwargs['vmax'] = vmax if 'vmax' not in kwargs else kwargs['vmax']
    # plot response maps using imshow
    for ax, data in zip(grid, zz):
        im = ax.imshow(data, **kwargs)
    [ax.axis('off') for ax in grid]
    fig.colorbar(im, cax=grid.cbar_axes[0]) if (normalize and add_colorbar) else None
    return fig, grid.axes_all


def set_size(width=345.0, fraction=1, mode="wide"):
    """Set figure dimensions to avoid scaling in LaTeX.
    Taken from:
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    To get the width of a latex document, print it with:
    \the\textwidth
    (https://tex.stackexchange.com/questions/39383/determine-text-width)
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    mode: str
            Whether figure should be scaled by the golden ratio in height
            or width
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width  # * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if mode == "wide":
        fig_height_in = fig_width_in * golden_ratio
    elif mode == "tall":
        fig_height_in = fig_width_in / golden_ratio
    elif mode == "square":
        fig_height_in = fig_width_in
    fig_height_max = 550.0 / 72.27
    if mode == "max" or fig_height_in > fig_height_max:
        # standard max-height of latex document
        fig_height_in = fig_height_max
    fig_dim = (fig_width_in, fig_height_in)
    if isinstance(fraction, (int, float)):
        fraction = (fraction, fraction)
    fig_dim = (fig_width_in * fraction[0], fig_height_in * fraction[1])
    return fig_dim