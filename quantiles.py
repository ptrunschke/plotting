import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def plot_quantiles(nodes, values, ax=None, num_quantiles=4, **kwargs):
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
    ax : matplotlib.axes.Axes, optional
        The axis object used for plotting. (default: matplotlib.pyplot.gca())
    num_quantiles : int (>= 0), optional
        Number of quantiles to plot. (default: 4)
    linewidth: float, optional
        Linewidth of the bounding lines of each quantile.
    """
    errors = values
    values = nodes
    assert values.ndim == 1
    assert len(values) == errors.shape[1]
    assert num_quantiles >= 0
    if ax is None:
        ax = plt.gca()

    ps = np.linspace(0,1,2*(num_quantiles+1)+1)[1:-1]
    qs = np.quantile(errors, ps, axis=0)

    # if num_quantiles == 0:
    #     alphas = np.full((1,), kwargs.get('alpha', 1))
    # else:
    #     alphas = np.empty(num_quantiles)
    #     alphas[0] = 2*ps[0]
    #     for i in range(1, num_quantiles):
    #         alphas[i] = 2*(ps[i] - ps[i-1])/(1 - 2*ps[i-1])

    base_line, = ax.plot(values, qs[num_quantiles], **kwargs)
    ls = [base_line]
    line_color = np.array(mpl.colors.to_rgba(base_line.get_color())); line_color[3] = 1
    color = np.array(mpl.colors.to_rgba(ax.get_facecolor())); color[3] = 1
    kwargs.pop('color', None)
    for e in range(num_quantiles):
        # alpha = alphas[e]
        alpha = 1/num_quantiles
        color = alpha*line_color + (1-alpha)*color
        l = ax.fill_between(values, qs[e], qs[-1-e], color=tuple(color.tolist()), **kwargs)
        ls.append(l)
    # return ls, alphas
    return ls


def plot_approximations(evaluate_function, evaluate_basis, reconstruct, ax, numTrials=10_000, title=None, cachePath=None):
    C0 = coloring.mix(coloring.bimosred, 80)
    C1 = "xkcd:black"

    cs = []
    es = []
    if cachePath is not None and os.path.isfile(cachePath):
        z = np.load(cachePath)
        cs = list(z['cs'])
        es = list(z['es'])
        assert len(cs) == len(es)

    with tqdm(total=numTrials, initial=len(cs), desc=f"Reconstruct '{title}'") as pbar:
        while len(cs) < numTrials:
            try:
                c, e = reconstruct()
                cs.append(c)
                es.append(e)
                pbar.update()
            except np.linalg.LinAlgError:
                pass  # Condition on the event that the problem is well-conditioned.
            except KeyboardInterrupt:
                break
            # except: pass  # Errors in the optimizer... TODO: make more specific
    assert len(cs) == len(es) or len(cs) == len(es)+1
    cs = cs[:len(es)]
    cs, es = np.array(cs, dtype=float), np.array(es, dtype=float)

    if cachePath is not None:
        np.savez_compressed(cachePath, cs=cs, es=es)

    xs = np.linspace(-1, 1, 1000)
    fxs = evaluate_function(xs)
    measures = evaluate_basis(xs, np.eye(cs.shape[1]))
    assert measures.shape == (len(xs), cs.shape[1])
    yss = cs @ measures.T
    assert yss.shape == (cs.shape[0], len(xs))

    numQuantiles = 500
    plot_quantiles(xs, yss, num_quantiles=numQuantiles, axes=ax, color=C1, linewidth=0)
    ax.plot(xs, fxs, linestyle=(0,(0.25,1.5)), color=C0, linewidth=3, dash_capstyle='round')
    ymin = np.min(fxs)-(np.max(fxs)-np.min(fxs))/4
    ymax = np.max(fxs)+(np.max(fxs)-np.min(fxs))/4
    ax.set_xlim(-1,1)
    ax.set_ylim(ymin, ymax)

    if title is not None:
        # errors = np.max(abs(yss - fxs[None]), axis=1)
        errors = np.sqrt(np.trapz((yss - fxs[None])**2, xs, axis=1))
        info = f"(avg.\ min.\ eigenvalue: {np.mean(es):.2e}, avg.\ error: {np.mean(errors):.2e})"
        ax.set_title(r"{\fontsize{15pt}{18pt}\selectfont{}"+title+r"}"+"\n"+r"{\fontsize{10pt}{12pt}\selectfont{}"+info+"}", multialignment='center')
