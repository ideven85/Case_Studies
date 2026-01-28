import base64
from io import BytesIO, StringIO
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects
import plotly.tools
import torch

from lib import cv


def tidy_plot(
    xmin, xmax, ymin, ymax, center=False, title=None, xlabel=None, ylabel=None
):
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")
        # deprecated, and function with lines removed seems to plot the same
        # ax.spines['left'].set_smart_bounds(True)
        # ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    eps = 0.05
    plt.xlim(xmin - eps, xmax + eps)
    plt.ylim(ymin - eps, ymax + eps)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def plot_data(
    data, labels, ax=None, clear=False, xmin=None, xmax=None, ymin=None, ymax=None
):
    if ax is None:
        if xmin == None:
            xmin = np.min(data[0, :]) - 0.5
        if xmax == None:
            xmax = np.max(data[0, :]) + 0.5
        if ymin == None:
            ymin = np.min(data[1, :]) - 0.5
        if ymax == None:
            ymax = np.max(data[1, :]) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin
        y_range = ymax - ymin
        if 0.1 < x_range / y_range < 10:
            ax.set_aspect("equal")
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    colors = np.choose(labels > 0, cv(["r", "g"]))[0]
    ax.scatter(data[0, :], data[1, :], c=colors, marker="o", s=50, edgecolors="none")
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, which="both")
    # ax.axhline(y=0, color='k')
    # ax.axvline(x=0, color='k')
    return ax


def plot_nonlin_sep(
    predictor, ax=None, xmin=None, xmax=None, ymin=None, ymax=None, res=30
):
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    cmap = colors.ListedColormap(["black", "white"])
    bounds = [-2, 0, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ima = np.array(
        [
            [predictor(x1i, x2i) for x1i in np.linspace(xmin, xmax, res)]
            for x2i in np.linspace(ymin, ymax, res)
        ]
    )
    im = ax.imshow(
        np.flipud(ima),
        interpolation="none",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        norm=norm,
    )


def savefig():
    b = BytesIO()
    plt.savefig(b)
    return (
        '<img style="width:600px" src="data:image/png;base64,%s" />'
        % base64.b64encode(b.getvalue()).decode()
    )


def plotly2html(fig):
    '''
    Return HTML for plotly figure.
    Note that the plotly js library is NOT included; this must be added, e.g. in preload.py with

    cs_scripts += """<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>"""
    cs_scripts += """<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>"""

    '''
    with StringIO() as ofp:
        fig.write_html(ofp, auto_open=False, include_plotlyjs=False, full_html=False)
        ofp.seek(0)
        fig_output = ofp.read()
    return fig_output


def savefig_plotly(legend=False, width=720, height=400):
    """
    Attempt to generate a plotly figure from a matplotlib figure.
    Doesn't always work!
    """

    def is_frame_like(self, *args, **kwargs):
        return False

    matplotlib.spines.Spine.is_frame_like = is_frame_like
    fig = plotly.tools.mpl_to_plotly(plt.gcf())
    if legend:
        fig["layout"]["showlegend"] = True
    fig.update_layout(width=width, height=height)
    return plotly2html(fig)


def plot_fun(ax, f, xmin, xmax, nPts=100, label=None):
    x = np.linspace(xmin, xmax, nPts)
    y = cv([float(f(xi)) for xi in x])
    ax.plot(x, y, label=label)
    return savefig()


def plot_points(
    x,
    y,
    ax=None,
    clear=False,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    style="or-",
    equal=False,
    mark_initial=False,
    mark_final=False,
):
    padup = lambda v: v + 0.05 * abs(v)
    paddown = lambda v: v - 0.05 * abs(v)
    if ax is None:
        if xmin == None:
            xmin = paddown(np.min(x))
        if xmax == None:
            xmax = padup(np.max(x))
        if ymin == None:
            ymin = paddown(np.min(y))
        if ymax == None:
            ymax = padup(np.max(y))
        ax = tidy_plot(xmin, xmax, ymin, ymax)
        x_range = xmax - xmin
        y_range = ymax - ymin
        if equal and 0.1 < x_range / y_range < 10:
            # ax.set_aspect('equal')
            plt.axis("equal")
            if x_range > y_range:
                ax.set_xlim((xmin, xmax))
            else:
                ax.set_ylim((ymin, ymax))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x, y, style, markeredgewidth=0.0, linewidth=3.0)
    if mark_initial:
        ax.plot(x[0], y[0], "xb", markersize=12)
    if mark_final:
        ax.plot(x[-1], y[-1], "Db", markersize=6)
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, which="both")
    return ax


def plot_heat(X, y, model, res=200):
    eps = 0.1
    X, y = X.numpy(), y.numpy()
    xmin = np.min(X[:, 0]) - eps
    xmax = np.max(X[:, 0]) + eps
    ymin = np.min(X[:, 1]) - eps
    ymax = np.max(X[:, 1]) + eps
    ax = tidy_plot(xmin, xmax, ymin, ymax, xlabel="x", ylabel="y")
    xl = np.linspace(xmin, xmax, res)
    yl = np.linspace(ymin, ymax, res)
    xx, yy = np.meshgrid(xl, yl, sparse=False)
    data = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    zz = np.argmax(model(data).detach().numpy(), axis=1)
    im = ax.imshow(
        np.flipud(zz.reshape((res, res))),
        interpolation="none",
        extent=[xmin, xmax, ymin, ymax],
        cmap="viridis",
    )
    plt.colorbar(im)
    for yi in set([int(_y) for _y in set(y)]):
        color = ["r", "g", "b"][yi]
        marker = ["X", "o", "v"][yi]
        cl = np.where(y == yi)
        ax.scatter(X[cl, 0], X[cl, 1], c=color, marker=marker, s=80, edgecolors="none")
    return ax
