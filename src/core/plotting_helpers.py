import numpy as np
import uuid

import matplotlib.scale as mscale
from matplotlib.transforms import Transform
from matplotlib.ticker import AutoLocator

from core.helpers import set_rwth_color

# DEFINE FUNCTIONS

def plot_custom_violin(ax, data, pos, color, facehatch = '', s=50):

    # Compute statistics
    mean_val = np.mean(data)
    p25, p75 = np.percentile(data, [25, 75])
    p5, p95 = np.percentile(data, [5, 95])

    filtered_data =  data[(data >= p5) & (data <= p95)]
    # filtered_data =  data[(data <= p95)]
    outliers = data[(data < p5) | (data > p95)]

    # Draw vertical "violin-like" shape manually
    violins = ax.violinplot(filtered_data, positions=[pos], widths=0.9, showmeans=False, vert=False)
    for v in violins['bodies']:
        v.set_alpha(1)
        v.set_facecolor(set_rwth_color(color, 50))
        v.set_edgecolor(set_rwth_color(color, 100))
        v.set_hatch(facehatch)

    violins['cmins'].set_alpha(0)
    violins['cmaxes'].set_alpha(0)

    # Draw quartile box
    ax.plot([p25, p75], [pos, pos], color='black', lw=4, alpha=1, zorder=2)

    # Draw whiskers (5th to 95th)
    ax.plot([p5, p95], [pos, pos], color='black', lw=2, zorder=3)

    # Mean marker
    ax.scatter(mean_val, pos, marker='o', color='white', edgecolor='black', s=s, zorder=5)

    # Median marker
    ax.scatter(np.median(data), pos,  marker='^', color='white', edgecolor='black', s=s, zorder=5)

    # Outliers
    ax.scatter(outliers, [pos] * len(outliers),  marker='o', color='grey', s=1, zorder=1.9,
               rasterized=True) # Scatter becomes an image, prevents large pdfs

    return p95

def get_grouped_positions(n_groups:int, n_members_per_group:int, spacing:float):
    pos = [1]* n_members_per_group * n_groups

    for i in range(n_groups-1):
        pos[ (i+1) * n_members_per_group] += spacing

    return np.cumsum(pos)

def get_center_positions(n_groups:int, n_members_per_group:int, spacing:float):
    centers = [float(n_members_per_group+1)/2] + [spacing + n_members_per_group]* (n_groups - 1)

    return np.cumsum(centers)

def row_major_for_legend(handles, ncol):
    """
    Intended use for custom legends: get column-by-column legend entries instead of row-by-row
    ax_leg.legend(
        handles=row_major(legend_elements, ncol),
        loc="upper center",
        ncol=ncol,
        frameon=True
    )

    """
    n = len(handles)
    nrow = int(np.ceil(n / ncol))
    grid = [[None]*ncol for _ in range(nrow)]

    for i, h in enumerate(handles):
        r = i // ncol
        c = i % ncol
        grid[r][c] = h

    # convert to column-major order expected by matplotlib
    return [grid[r][c]
            for c in range(ncol)
            for r in range(nrow)
            if grid[r][c] is not None]

def get_latex_textwidth():
    return 6.842519441

def get_latex_columnwidth():
    return 3.296726719

def binarize_lost_load(y, abs_tol:float=1e-13) -> np.ndarray:
    """ 1 = feasible (lostload <= 0), 0 = infeasible (lostload > 0)

    Args:
        y (array-like): lost load values to binarize
        abs_tol (float, optional): Absolute tolerance to account for numerical precision issues when
            comparing lost load to zero. Defaults to 1e-13.

    Returns:
        np.ndarray: Binary array where 1 indicates feasible (lost load <= 0) and 0 indicates
            infeasible (lost load > 0).
    """
    return (np.array(y) <= 0 + abs_tol).astype(int)

def binarize_overcapacity(y, oc_min=0, abs_tol:float=1e-13) -> np.ndarray:
    """ 1 = feasible (overcapacity >= 0), 0 = infeasible (overcapacity < 0)

    Args:
        y (array-like): overcapacity values to binarize
        oc_min (int, optional): Minimum overcapacity threshold to consider as feasible. The higher
            this threshold, the more conservative the classifiction. Defaults to 0.
        abs_tol (float, optional): Absolute tolerance to account for numerical precision issues when
            comparing overcapacity to the threshold. Defaults to 1e-13.

    Returns:
        np.ndarray: Binary array where 1 indicates feasible (overcapacity >= oc_min) and 0 indicates
            infeasible (overcapacity < oc_min).
    """
    return (np.array(y) >= oc_min - abs_tol).astype(int)

def create_piecewise_linear_scale(vmin, vsplit, vmax, split_pos=0.5):
    """
    Returns a unique scale name that you can pass to ax.set_xscale(...)

    Example usage:

    scale_name :str = create_piecewise_linear_scale(
        vmin=30,
        vsplit=95,
        vmax=100,
        split_pos=0.3   # 30% of axis for first segment
    )

    x = np.linspace(30, 100, 500)
    y = np.sin(x / 5)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set_xscale(scale_name)
    ax.set_xlim(30, 100)
    ax.grid()
    plt.show()

    """

    if not (vmin < vsplit < vmax):
        raise ValueError("Require vmin < vsplit < vmax.")
    if not (0.0 < split_pos < 1.0):
        raise ValueError("Require 0 < split_pos < 1.")

    class _Forward(Transform):
        input_dims = output_dims = 1
        is_separable = True

        def transform_non_affine(self, x):
            x = np.asarray(x, dtype=float)
            out = np.empty_like(x)

            m1 = x <= vsplit
            out[m1] = split_pos * (x[m1] - vmin) / (vsplit - vmin)

            m2 = ~m1
            out[m2] = split_pos + (1 - split_pos) * (x[m2] - vsplit) / (vmax - vsplit)

            return out

        def inverted(self):
            return _Inverse()

    class _Inverse(Transform):
        input_dims = output_dims = 1
        is_separable = True

        def transform_non_affine(self, u):
            u = np.asarray(u, dtype=float)
            out = np.empty_like(u)

            m1 = u <= split_pos
            out[m1] = vmin + (u[m1] / split_pos) * (vsplit - vmin)

            m2 = ~m1
            out[m2] = vsplit + ((u[m2] - split_pos) / (1 - split_pos)) * (vmax - vsplit)

            return out

        def inverted(self):
            return _Forward()

    class PiecewiseScale(mscale.ScaleBase):
        def get_transform(self):
            return _Forward()

        def set_default_locators_and_formatters(self, axis):
            axis.set_major_locator(AutoLocator())

        def limit_range_for_scale(self, vmin_in, vmax_in, minpos):
            return max(vmin_in, vmin), min(vmax_in, vmax)

    # Assign unique name AFTER class creation
    scale_name = f"pw_{uuid.uuid4().hex}"
    PiecewiseScale.name = scale_name

    mscale.register_scale(PiecewiseScale)

    return scale_name
