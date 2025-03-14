import os
import typing as t
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def mix(*color_value_list, alpha=True):
    if alpha:
        colorvec = lambda c: np.array(mpl.colors.to_rgba(c))
    else:
        colorvec = lambda c: np.array(mpl.colors.to_rgba(c))[:3]
        # colorvec = lambda c: np.array(mpl.colors.to_rgb(c))
    assert len(color_value_list) > 0
    c1 = colorvec(color_value_list[0])
    if len(color_value_list) == 1:
        return c1
    v = color_value_list[1] / 100
    assert isinstance(v, float)
    if len(color_value_list) > 2:
        c2 = colorvec(color_value_list[2])
    else:
        c2 = 1
    c3 = v * c1 + (1 - v) * c2
    return mix(c3, *color_value_list[3:])


bimosblack = "#23373B"
bimosred = "#A60000"  # (.65,0,0)
bimosyellow = "#F9F7F7"

# \setbeamercolor{normal text}{fg=mDarkTeal, bg=bimosyellow!50!bimosred!8}
normal_text_fg = mix(bimosblack, 95, bimosyellow)
normal_text_bg = bimosyellow
# \setbeamercolor{alerted text}{fg=bimosred!80!bimosyellow}
alerted_text_fg = mix(bimosred, 80, bimosyellow)
# \setbeamercolor{example text}{fg=bimosred!50!bimosyellow!80}
example_text_fg = mix(bimosred, 50, bimosyellow, 80)
# \setbeamercolor{block title}{fg=normal text.fg, bg=normal text.bg!80!fg}
block_title_bg = mix(normal_text_bg, 80, bimosblack)
# \setbeamercolor{block body}{bg=block title.bg!50!normal text.bg}
block_body_bg = mix(block_title_bg, 50, normal_text_bg)
# \setbeamercolor{frametitle}{fg=bimosred!75!normal text.fg, bg=normal text.bg}
frametitle_fg = mix(bimosred, 75, normal_text_fg)


class PlotStyle(t.Protocol):
    @property
    def geometry(self) -> dict[str, float]:
        pass

    @property
    def fg(self) -> str:
        pass

    @property
    def bg(self) -> str:
        pass

    @property
    def font(self) -> dict[str, t.Any]:
        pass

    def set(self) -> None:
        pass


class DarkMode(object):
    def __init__(self) -> None:
        self.geometry = {
            "top": 1,
            "bottom": 0,
            "left": 0,
            "right": 1,
            "wspace": 0.25,  # the default as defined in rcParams
            "hspace": 0.25,  # the default as defined in rcParams
        }
        self.fg = "#FFFFFF"
        self.bg = "#1F1F1F"
        self.font = {
            "family": "serif",
            "weight": "ultralight",
            "size": 10,
        }

    def set(self) -> None:
        mpl.rc("font", **self.font)
        # mpl.rc("text", usetex=True)
        # mpl.rc(
        #     "text.latex",
        #     preamble=r"""
        #     \usepackage{amsmath}
        #     \usepackage{bbm}
        # """,
        # )
        mpl.rc("text", color=self.fg)
        mpl.rc("figure", facecolor=self.bg, edgecolor=self.bg)
        mpl.rc(
            "axes",
            facecolor=self.bg,
            edgecolor=self.fg,
            labelcolor=self.fg,
        )
        mpl.rc("xtick", color=self.fg)
        mpl.rc("ytick", color=self.fg)


class BIMoSStyle(object):
    def __init__(self) -> None:
        self.geometry = {
            "top": 1,
            "bottom": 0,
            "left": 0,
            "right": 1,
            "wspace": 0.25,  # the default as defined in rcParams
            "hspace": 0.25,  # the default as defined in rcParams
        }
        self.fontSize = 10

    def set(self) -> None:
        mpl.rc("font", size=self.fontSize, family="Times New Roman")
        mpl.rc("text", usetex=True)
        mpl.rc(
            "text.latex",
            preamble=r"""
            \usepackage{newtxmath}
            \usepackage{amsmath}
            \usepackage{bbm}
        """,
        )
        mpl.rc("figure", facecolor=normal_text_bg, edgecolor=normal_text_bg)
        mpl.rc(
            "axes",
            facecolor=block_body_bg,
            edgecolor=normal_text_fg,
            labelcolor=normal_text_fg,
        )
        mpl.rc("xtick", color=normal_text_fg)
        mpl.rc("ytick", color=normal_text_fg)


@contextmanager
def save_figure(
    output_path: Path,
    *fig_shape: int,
    fig_width: float = 6.52437486112,
    plot_style: PlotStyle,
) -> t.Iterator[tuple[plt.Figure, plt.Axes]]:
    """
    Context manager to save a figure to a file.

    Parameters
    ----------
    filename : str
    plot_style : PlotStyle
    figwidth : float, optional
        Width of the figure in inches, by default 6.52437486112
    """
    plot_style.set()

    assert len(fig_shape) <= 2
    if len(fig_shape) == 0:
        fig_shape = (1, 1)
    elif len(fig_shape) == 1:
        fig_shape = (1, fig_shape[0])

    phi = (1 + np.sqrt(5)) / 2
    aspect_ratio = phi
    figsize = compute_figsize(
        plot_style.geometry, fig_shape, aspect_ratio=aspect_ratio, figwidth=fig_width
    )

    try:
        fig, ax = plt.subplots(*fig_shape, figsize=figsize, dpi=300)
        fig.patch.set_facecolor(plot_style.bg)
        if isinstance(ax, np.ndarray):
            for axi in ax.ravel():
                axi.set_facecolor(plot_style.bg)
        else:
            ax.set_facecolor(plot_style.bg)
        yield fig, ax
    finally:
        plt.subplots_adjust(**plot_style.geometry)
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        print(f"Saving figure to '{output_path}'")
        plt.savefig(
            output_path,
            format="png",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


def homotopy(x, y, num):
    assert x.shape in {(3,), (4,)}
    assert y.shape in {(3,), (4,)}
    x = x[None]
    y = y[None]
    s = np.linspace(0, 1, num)[:, None]
    return (1 - s) * x + s * y


def compute_figsize(geometry, shape, aspect_ratio=1, figwidth=3.98584):
    subplotwidth = (
        (geometry["right"] - geometry["left"])
        * figwidth
        / (shape[1] + (shape[1] - 1) * geometry["wspace"])
    )  # make as wide as two plots
    subplotheight = subplotwidth / aspect_ratio
    figheight = subplotheight * (shape[0] + (shape[0] - 1) * geometry["hspace"])
    figheight = figheight / (geometry["top"] - geometry["bottom"])
    return (figwidth, figheight)


# @contextmanager
# def save_figure(filename: str, *figshape):
#     BG = "xkcd:white"
#     geometry = BIMoSStyle().geometry

#     assert len(figshape) <= 2
#     if len(figshape) == 0:
#         figshape = (1, 1)
#     elif len(figshape) == 1:
#         figshape = (1, figshape[0])

#     mpl.rc("font", size=10, family="Times New Roman")
#     mpl.rc("text", usetex=True)

#     figwidth = 6.52437486112  # width of the figure in inches
#     phi = (1 + np.sqrt(5)) / 2
#     aspect_ratio = phi
#     figsize = compute_figsize(
#         geometry, figshape, aspect_ratio=aspect_ratio, figwidth=figwidth
#     )

#     try:
#         fig, ax = plt.subplots(*figshape, figsize=figsize, dpi=300)
#         fig.patch.set_facecolor(BG)
#         if isinstance(ax, np.ndarray):
#             for axi in ax.ravel():
#                 axi.set_facecolor(BG)
#         else:
#             ax.set_facecolor(BG)
#         yield fig, ax
#     finally:
#         plt.subplots_adjust(**geometry)
#         dirname = os.path.dirname(filename)
#         if dirname:
#             os.makedirs(dirname, exist_ok=True)
#         plt.savefig(
#             filename,
#             format="png",
#             facecolor=fig.get_facecolor(),
#             edgecolor="none",
#             bbox_inches="tight",
#         )
#         plt.close(fig)
