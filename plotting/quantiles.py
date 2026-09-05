from __future__ import annotations

import typing as t

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

if t.TYPE_CHECKING:
    import jaxtyping as jt
    from matplotlib.axes import Axes


def _order_stats_feasible(
    confidence: float, sample_size: int, p_low: float, p_high: float
) -> bool:
    """Whether the order statistics for the extreme quantiles exist.

    The lower extreme quantile `p_low` requires a binomial rank `>= 2` below
    the sample minimum and the upper extreme quantile `p_high` requires a
    rank `<= sample_size - 1` above the sample maximum; both are monotone in
    `confidence`.
    """
    return bool(
        binom.ppf(1 - confidence, sample_size, p_low) >= 2
        and binom.ppf(confidence, sample_size, p_high) <= sample_size - 1
    )


def _max_confidence(sample_size: int, p_low: float, p_high: float) -> float:
    """Largest feasible confidence for the given extreme quantiles.

    The bounds follow from the feasibility conditions:
    `confidence < 1 - binom.cdf(1, sample_size, p_low)` on the lower side and
    `confidence <= binom.cdf(sample_size - 1, sample_size, p_high)` on the
    upper side. The result is nudged to the largest strictly feasible float.
    """
    confidence = min(
        1 - binom.cdf(1, sample_size, p_low),
        binom.cdf(sample_size - 1, sample_size, p_high),
    )
    while confidence > 0 and not _order_stats_feasible(
        confidence, sample_size, p_low, p_high
    ):
        confidence = np.nextafter(confidence, 0.0)
    return float(confidence)


def _max_num_quantiles(confidence: float, sample_size: int, requested: int) -> int:
    """Largest feasible num_quantiles at the given confidence (0 = no band).

    Feasibility is monotone in num_quantiles: a smaller num_quantiles widens
    the extreme quantile level `1 / (2 * (num_quantiles + 1))`, which makes
    the order statistics easier to estimate.
    """
    for num_quantiles in range(requested - 1, 0, -1):
        p = 1 / (2 * (num_quantiles + 1))
        if _order_stats_feasible(confidence, sample_size, p, 1 - p):
            return num_quantiles
    return 0


def _confidence_order_stats(
    values: jt.Float[np.ndarray, "trials x_locations"],
    qrange: tuple[float, float],
    num_quantiles: int,
    confidence: float,
) -> jt.Float[np.ndarray, "2*num_quantiles+1 x_locations"]:
    """Estimate the quantiles of the underlying function via order statistics.

    Raises
    ------
    ValueError
        If there are too few trials to estimate the extreme quantiles at the
        requested confidence.
    """
    assert 0 < confidence < 1
    if num_quantiles == 0:
        return np.median(values, axis=0)[None]
    ps = np.linspace(*qrange, num=2 * (num_quantiles + 1) + 1)[1:-1]
    ps_low = ps[:num_quantiles]
    ps_high = ps[num_quantiles + 1 :]
    assert np.allclose(ps[num_quantiles], 0.5)
    assert np.all(ps_low < 0.5)
    assert np.allclose(1 - ps_high[::-1], ps_low)
    sample_size = values.shape[0]
    # The extreme quantiles are the hardest to estimate (feasibility is
    # monotone in the quantile level), so checking them suffices.
    if not _order_stats_feasible(confidence, sample_size, ps_low[0], ps_high[-1]):
        max_confidence = _max_confidence(sample_size, ps_low[0], ps_high[-1])
        confidence_option = (
            f"num_quantiles={num_quantiles} requires confidence < {max_confidence:.3g}"
            if max_confidence > 0
            else f"num_quantiles={num_quantiles} supports no quantile band"
        )
        max_num_quantiles = _max_num_quantiles(confidence, sample_size, num_quantiles)
        num_quantiles_option = (
            f"confidence={confidence} requires num_quantiles <= {max_num_quantiles}"
            if max_num_quantiles > 0
            else f"confidence={confidence} supports no quantile band"
        )
        raise ValueError(  # noqa: TRY003
            f"cannot estimate the {ps_low[0]:.3g}- and "
            f"{ps_high[-1]:.3g}-quantiles at confidence {confidence} from "
            f"{sample_size} realisations; {confidence_option}, "
            f"{num_quantiles_option}; use confidence=None to plot the "
            "empirical quantiles"
        )
    us = binom.ppf(confidence, sample_size, ps_high)
    assert us.shape == ps_high.shape
    assert binom.cdf(us, sample_size, ps_high).shape == ps_high.shape
    assert np.all(binom.cdf(us, sample_size, ps_high) >= confidence)
    assert np.all(us.astype(int) == us)
    assert np.all(us >= np.ceil(ps_high * values.shape[0]))
    us = us.astype(int)
    ls = binom.ppf(1 - confidence, values.shape[0], ps_low) - 1
    assert np.all(ls > 0)
    assert ls.shape == ps_low.shape
    assert binom.cdf(ls, sample_size, ps_low).shape == ps_low.shape
    assert np.all(1 - binom.cdf(ls, sample_size, ps_low) >= confidence)
    assert np.all(ls.astype(int) == ls)
    assert np.all(ls <= np.floor(ps_low * values.shape[0]))
    ls = ls.astype(int)
    assert np.all(ls == sample_size - us[::-1] - 1)
    values = np.sort(values, 0)
    assert np.all(values[:-1] <= values[1:])
    return np.concatenate(
        [values[ls], np.median(values, axis=0)[None], values[us]], axis=0
    )


def _step_style(drawstyle: str) -> t.Literal["pre", "post", "mid"] | None:
    """Translate an Axes.plot drawstyle into an Axes.fill_between step style."""
    if drawstyle.startswith("steps-"):
        return t.cast("t.Literal['pre', 'post', 'mid']", drawstyle[len("steps-") :])
    assert drawstyle == "default"
    return None


# TODO: Adapt to google style!
def plot_quantiles(
    nodes: jt.Float[np.ndarray, " x_locations"],
    values: jt.Float[np.ndarray, "trials x_locations"],
    ax: Axes | None = None,
    qrange: tuple[float, float] = (0, 1),
    num_quantiles: int = 4,
    confidence: float | None = None,
    *,
    drawstyle: str = "default",
    **kwargs: t.Any,
) -> list[mpl.lines.Line2D | mpl.collections.PolyCollection]:
    """
    Plot the quantiles for a stochastic process.

    Excludes the 1/num_quantiles and 1-1/num_quantiles quantiles.

    Parameters
    ----------
    nodes : ndarray (shape: (n,))
        Nodes at which the process is measured (index set, plotted on the x-axis).
    values : ndarray (shape: (m,n))
        Realizations of the the process (plotted on the y-axis).
        Each row of values contains a different path of the stochastic process.
        Nan's are considered to be missing values and are ignored.
    ax : matplotlib.axes.Axes, optional
        The axis object used for plotting. (default: matplotlib.pyplot.gca())
    qrange : pair of floats
        Range of quantiles to plot. (default: (0, 1))
    num_quantiles : int (>= 0), optional
        Number of quantiles to plot. (default: 4)
    confidence : float or None
        If confidence is None, the quantiles of the data are plotted.
        If confidence is a float in the interval (0,1), the quantiles of the
        underlying function are estimated with the given confidence.
        (Assumes data points are iid.)
        The number of trials must be large enough compared to num_quantiles
        to estimate the extreme quantiles at the requested confidence.
    drawstyle : str
        See Axes.plot.

    Other Parameters
    ----------------
    **kwargs : typing.Any
        All other keyword arguments are passed on to Axes.plot and Axes.fill_between.

    Raises
    ------
    ValueError
        If the number of trials is insufficient to estimate the extreme
        quantiles at the requested confidence; the error states the feasible
        confidence for the requested num_quantiles and the feasible
        num_quantiles for the requested confidence.
    """
    assert nodes.ndim == 1
    assert len(nodes) == values.shape[1]
    assert num_quantiles >= 0
    if ax is None:
        ax = plt.gca()
    assert len(qrange) == 2 and 0 <= qrange[0] < qrange[1] <= 1

    if confidence is None:
        ps = np.linspace(*qrange, num=2 * num_quantiles + 1)
        qs = np.nanquantile(values, ps, axis=0)
    else:
        qs = _confidence_order_stats(values, qrange, num_quantiles, confidence)

    zorder = max([child.zorder for child in ax.get_children()])
    zorder = kwargs.pop("zorder", zorder)
    color = kwargs.pop("color", None)
    label = kwargs.pop("label", None)
    (base_line,) = ax.plot(
        nodes,
        qs[num_quantiles],
        zorder=zorder + 1,
        drawstyle=drawstyle,
        color=color,
        **kwargs,
    )
    step = _step_style(drawstyle)
    alpha = kwargs.pop("alpha", 1)
    kwargs.pop("lw", None)
    kwargs.pop("linewidth", None)
    all_lines: list[mpl.lines.Line2D | mpl.collections.PolyCollection] = [base_line]
    color = np.array(mpl.colors.to_rgba(base_line.get_color()))
    color[3] = alpha / max(num_quantiles, 1)
    for e in range(num_quantiles):
        line = ax.fill_between(
            nodes,
            qs[e],
            qs[-1 - e],
            color=tuple(color.tolist()),
            lw=0,
            zorder=zorder,
            step=step,
            **kwargs,
        )
        all_lines.append(line)
    if label is not None:
        all_lines.append(
            _quantile_band_legend_artist(ax, base_line, alpha, num_quantiles, label)
        )
    return all_lines


def _quantile_band_legend_artist(
    ax: Axes,
    base_line: mpl.lines.Line2D,
    alpha: float,
    num_quantiles: int,
    label: str,
) -> mpl.lines.Line2D | mpl.collections.PolyCollection:
    """Legend proxy for the quantile band.

    An invisible, empty ``fill_between`` whose legend icon is an opaque patch
    in the band's average perceived colour; for ``num_quantiles == 0`` (no
    band), the median line itself is labelled instead.

    The band consists of ``num_quantiles`` fills of opacity
    ``alpha / num_quantiles`` stacked over the axes facecolour. A region
    covered by ``k`` fills has opacity ``1 - (1 - alpha / num_quantiles)**k``;
    assuming evenly spaced quantile curves (each coverage region occupies the
    same share of the band area), the area-weighted average opacity is
    ``1 - beta * (1 - beta**n) / (n * (1 - beta))`` with ``beta = 1 - alpha/n``.
    """
    if num_quantiles == 0:
        base_line.set_label(label)
        return base_line
    if alpha > 0:
        beta = 1 - alpha / num_quantiles
        mean_opacity = 1 - (
            beta * (1 - beta**num_quantiles) / (num_quantiles * (1 - beta))
        )
    else:
        mean_opacity = 0.0
    base_rgb = np.array(mpl.colors.to_rgb(base_line.get_color()))
    face_rgb = np.array(mpl.colors.to_rgb(ax.get_facecolor()))
    swatch_rgb = mean_opacity * base_rgb + (1 - mean_opacity) * face_rgb
    return ax.fill_between([], [], [], color=(*swatch_rgb, 1.0), label=label)


# NOTE: The subsequent function is deprecated, since it is not purely related to plotting.

# import os
# from tqdm import tqdm

# def plot_approximations(
#     evaluate_function, evaluate_basis, reconstruct, ax, numTrials=10_000, title=None, cachePath=None,
# ):
#     C0 = coloring.mix(coloring.bimosred, 80)
#     C1 = "xkcd:black"

#     cs = []
#     es = []
#     if cachePath is not None and os.path.isfile(cachePath):
#         z = np.load(cachePath)
#         cs = list(z["cs"])
#         es = list(z["es"])
#         assert len(cs) == len(es)

#     with tqdm(total=numTrials, initial=len(cs), desc=f"Reconstruct '{title}'") as pbar:
#         while len(cs) < numTrials:
#             try:
#                 c, e = reconstruct()
#                 cs.append(c)
#                 es.append(e)
#                 pbar.update()
#             except np.linalg.LinAlgError:
#                 pass  # Condition on the event that the problem is well-conditioned.
#             except KeyboardInterrupt:
#                 break
#             # except: pass  # Errors in the optimizer... TODO: make more specific
#     assert len(cs) == len(es) or len(cs) == len(es) + 1
#     cs = cs[: len(es)]
#     cs, es = np.array(cs, dtype=float), np.array(es, dtype=float)

#     if cachePath is not None:
#         np.savez_compressed(cachePath, cs=cs, es=es)

#     xs = np.linspace(-1, 1, 1000)
#     fxs = evaluate_function(xs)
#     measures = evaluate_basis(xs, np.eye(cs.shape[1]))
#     assert measures.shape == (len(xs), cs.shape[1])
#     yss = cs @ measures.T
#     assert yss.shape == (cs.shape[0], len(xs))

#     numQuantiles = 500
#     plot_quantiles(xs, yss, num_quantiles=numQuantiles, axes=ax, color=C1, linewidth=0)
#     ax.plot(xs, fxs, linestyle=(0, (0.25, 1.5)), color=C0, linewidth=3, dash_capstyle="round")
#     ymin = np.min(fxs) - (np.max(fxs) - np.min(fxs)) / 4
#     ymax = np.max(fxs) + (np.max(fxs) - np.min(fxs)) / 4
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(ymin, ymax)

#     if title is not None:
#         # errors = np.max(abs(yss - fxs[None]), axis=1)
#         errors = np.sqrt(np.trapz((yss - fxs[None]) ** 2, xs, axis=1))
#         info = f"(avg.\ min.\ eigenvalue: {np.mean(es):.2e}, avg.\ error: {np.mean(errors):.2e})"
#         ax.set_title(
#             r"{\fontsize{15pt}{18pt}\selectfont{}"
#             + title
#             + r"}"
#             + "\n"
#             + r"{\fontsize{10pt}{12pt}\selectfont{}"
#             + info
#             + "}",
#             multialignment="center",
#         )
