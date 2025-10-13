from __future__ import annotations

import typing as t

import jaxtyping as jt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.ndimage import convolve1d
from sklearn.cluster import KMeans

from .plotting import mix
from .quantiles import plot_quantiles


def fd_bin_width(
    sample: jt.Float[np.ndarray, "..."], axis: t.Optional[int] = None
) -> jt.Float[np.ndarray, "..."]:
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 0 for the bin width.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.

    Notes
    -----
    Copied from numpy.lib._histograms_impl._hist_bin_fd
    """
    if axis is None:
        sample = sample.ravel()
        axis = 0
    iqr = np.subtract(*np.quantile(sample, [0.75, 0.25], axis=axis))
    return 2.0 * iqr * sample.shape[axis] ** (-1.0 / 3.0)


def ash_1d(
    samples: jt.Float[np.ndarray, "*shape n_samples"],
    domain: tuple[float, float],
    *,
    n_bins: t.Optional[int] = None,
    n_shifts: t.Optional[int] = None,
    extend: bool = True,
) -> tuple[
    jt.Float[np.ndarray, "n_bins*n_shifts"],
    jt.Float[np.ndarray, "*shape n_bins*n_shifts+1"],
]:
    """Compute averaged shifted histograms."""
    assert len(domain) == 2
    assert np.all(domain[0] <= samples) and np.all(samples <= domain[1])
    *shape, n_samples = samples.shape
    samples = samples.reshape(-1, n_samples)
    domain_width = domain[1] - domain[0]
    if n_bins is None:
        bin_width = np.max(fd_bin_width(samples, axis=1))
        n_bins = int(np.ceil(domain_width / bin_width))
    assert n_bins >= 1
    if n_shifts is None:
        n_shifts = max(120 // n_bins + 1, 2)
    assert n_shifts >= 2
    if extend:
        bin_width = (domain[1] - domain[0]) / (n_bins * n_shifts)
        edges = np.linspace(
            domain[0] - bin_width, domain[1] + bin_width, n_bins * n_shifts + 3
        )
    else:
        edges = np.linspace(*domain, n_bins * n_shifts + 1)
        assert np.allclose(edges[::n_shifts], np.linspace(*domain, n_bins + 1))
    # frequencies, _ = np.histogram(samples, bins=edges)
    samples = (samples - domain[0]) / domain_width  # scale domain to (0, 1)
    samples = (samples * (n_bins * n_shifts)).astype(int)
    samples = np.minimum(samples, n_bins * n_shifts - 1)
    frequencies = np.zeros((samples.shape[0], n_bins * n_shifts + 2 * extend))
    for bin in range(n_bins * n_shifts + 2 * extend):
        frequencies[:, bin] = np.count_nonzero(samples == bin, axis=-1)
    kernel = 1 - abs(np.arange(1 - n_shifts, n_shifts)) / n_shifts
    assert (
        np.all(kernel[::-1] == kernel) and np.all(0 <= kernel) and np.all(kernel <= 1)
    )
    values = convolve1d(frequencies, kernel, axis=1, mode="constant")
    values = np.maximum(values, 0)
    assert np.all(np.isfinite(values))
    values /= (values @ np.diff(edges))[:, None]
    return edges, values.reshape(*shape, len(edges) - 1)


def midpoints(
    edges: jt.Float[np.ndarray, "n_bins+1"],
) -> jt.Float[np.ndarray, "n_bins"]:
    return (edges[1:] + edges[:-1]) / 2


def hellinger_distance_matrix(
    edges: jt.Float[np.ndarray, "n_bins+1"],
    densities: jt.Float[np.ndarray, "n_clusters n_bins"],
) -> jt.Float[np.ndarray, "n_clusters n_clusters"]:
    assert densities.ndim == 2
    n_clusters = densities.shape[0]
    distance_matrix = (np.sqrt(densities)[:, None] - np.sqrt(densities)[None]) ** 2
    distance_matrix = np.einsum("ijk, k -> ij", distance_matrix, np.diff(edges))
    distance_matrix = np.sqrt(0.5 * distance_matrix)
    assert distance_matrix.shape == (n_clusters, n_clusters)
    assert np.allclose(distance_matrix.T, distance_matrix)
    assert np.all(0 <= distance_matrix) and np.all(distance_matrix <= 1 + 1e-8)
    return distance_matrix


def density_clusters(
    edges: jt.Float[np.ndarray, " n_bins+1"],
    densities: jt.Float[np.ndarray, "n_densities n_bins+1"],
    *,
    max_n_clusters: t.Optional[int] = None,
    max_distance: float = 0.1,
    n_init: int = 10,
) -> list[jt.Float[np.ndarray, "cluster_size n_bins+1"]]:
    assert densities.ndim == 2
    if max_n_clusters is None:
        max_n_clusters = densities.shape[0]
    min_cluster_size = max(
        densities.shape[0] // 10, 100
    )  # 10% of the sample size, but at least 100
    min_cluster_size = min(min_cluster_size, densities.shape[0])
    old_clusters = [densities]
    for n_clusters in range(2, max_n_clusters + 1):
        # Using sqrt(densities) because the Euclidean distance on sqrt(counts) is the
        # square root of the Hellinger distance on counts. The Euclidean distance has
        # the advantage that barycentres are easy to compute.
        clustering = KMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init).fit(
            np.sqrt(densities)
        )
        clusters = [densities[clustering.labels_ == e] for e in range(n_clusters)]
        if min(len(cluster) for cluster in clusters) < min_cluster_size:
            break
        cluster_means = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        assert cluster_means.shape == (n_clusters, len(edges) - 1)
        # Compute the minimum Hellinger dist. between each cluster mean and all others.
        distance_matrix = hellinger_distance_matrix(edges, cluster_means)
        distance_matrix[np.arange(n_clusters), np.arange(n_clusters)] = np.inf
        if np.min(distance_matrix) < max_distance:
            break
        old_clusters = clusters
    assert min(len(cluster) for cluster in old_clusters) >= min_cluster_size
    return old_clusters


def plot_ash_quantiles(
    samples: jt.Float[np.ndarray, "*shape n_samples"],
    domain: tuple[float, float],
    ax: t.Optional[Axes] = None,
    *,
    split_walks: bool = True,
    n_bins: t.Optional[int] = None,
    n_shifts: t.Optional[int] = None,
    extend: bool = True,
    num_quantiles: int = 16,
    confidence: float = 0.99,
    **kwargs: t.Any,
) -> tuple[list[jt.Float[np.ndarray, "cluster_size n_bins"]], float]:
    assert samples.ndim > 1
    samples = samples.reshape(-1, samples.shape[-1])
    if split_walks:
        # samples = samples.reshape(samples.shape[0], 2, samples.shape[1] // 2)
        samples = samples.reshape(2 * samples.shape[0], samples.shape[1] // 2)
    edges, heights = ash_1d(
        samples,
        domain,
        n_bins=n_bins,
        n_shifts=n_shifts,
        extend=extend,
    )
    assert heights.shape == (samples.shape[0], len(edges) - 1)

    # TODO: Currently, the density_clusters squashes sampler and walker dimensions.
    #       This is not problematic for the clustering, but we can not distinguish
    #       clusters in the samplers from clusters in the walkers.
    clusters = density_clusters(edges, heights)

    cluster_means = np.array([np.mean(cluster, axis=0) for cluster in clusters])
    assert cluster_means.shape == (len(clusters), len(edges) - 1)
    distance_matrix = hellinger_distance_matrix(edges, cluster_means)
    distance_matrix[np.arange(len(clusters)), np.arange(len(clusters))] = np.inf
    min_cluster_distance = np.min(distance_matrix)

    if ax is None:
        ax = plt.gca()
    color = kwargs.pop("color", ax._get_patches_for_fill.get_next_color())
    fg = mpl.rcParams["axes.edgecolor"]
    bg = ax.get_facecolor()
    ax.plot(
        midpoints(edges),
        np.median(heights, axis=0),
        color=mix(fg, 75, color),
        zorder=num_quantiles + 1,
        lw=1,
    )
    ax.plot(
        midpoints(edges),
        np.min(heights, axis=0),
        color=mix(fg, 50, color, 50, bg),
        zorder=num_quantiles + 1,
        lw=0.5,
    )
    ax.plot(
        midpoints(edges),
        np.max(heights, axis=0),
        color=mix(fg, 50, color, 50, bg),
        zorder=num_quantiles + 1,
        lw=0.5,
    )
    plot_quantiles(
        midpoints(edges),
        heights,
        ax,
        color=color,
        num_quantiles=num_quantiles,
        confidence=confidence,
        zorder=0,
    )
    for cluster in clusters:
        ax.plot(
            midpoints(edges),
            np.mean(cluster, axis=0),
            color=color,
            alpha=0.9 * cluster.shape[0] / heights.shape[0],
            zorder=num_quantiles + 2,
            lw=1,
        )

    return clusters, min_cluster_distance
