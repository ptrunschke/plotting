import matplotlib as mpl
import numpy as np
from skimage.color import rgb2lab as _rgb2lab, lab2rgb as _lab2rgb

from plotting.plotting import mix, bimosblack, bimosred, bimosyellow, homotopy, example_text_fg


def rgb2lab(cs):
    assert cs.ndim >= 1
    assert cs.shape[-1] in {3, 4}
    has_alpha = cs.shape[-1] == 4
    if has_alpha:
        alpha = cs[..., -1][..., None]
    ndim = cs.ndim
    cs = cs[(None,) * (3 - ndim)][..., :3]
    ret = _rgb2lab(cs)[(0,) * (3 - ndim)]
    if has_alpha:
        ret = np.concatenate([ret, alpha], axis=-1)
    return ret


def lab2rgb(cs):
    assert cs.ndim >= 1
    assert cs.shape[-1] in {3, 4}
    has_alpha = cs.shape[-1] == 4
    if has_alpha:
        alpha = cs[..., -1][..., None]
    ndim = cs.ndim
    cs = cs[(None,) * (3 - ndim)][..., :3]
    ret = _lab2rgb(cs)[(0,) * (3 - ndim)]
    if has_alpha:
        ret = np.concatenate([ret, alpha], axis=-1)
    return ret


assert np.allclose(lab2rgb(rgb2lab(mix(bimosred))) - mix(bimosred), 0)


def lightness(cs):
    return rgb2lab(cs)[..., 0]


def sequential_cmap(*seq, lightness_range=(5, 98), steps=1000, strict=False):
    seq = np.array(sorted([rgb2lab(mix(c, alpha=False)) for c in seq], key=lambda c: c[0]))

    if seq[0][0] > lightness_range[0]:
        # if strict:
        #     raise ValueError("Color with minimum lightness not provided in mode 'strict'")
        c0 = seq[0].copy()
        c0[0] = lightness_range[0]
        seq = np.concatenate([c0[None], seq], axis=0)
    if seq[-1][0] < lightness_range[1]:
        # if strict:
        #     raise ValueError("Color with maximum lightness not provided in mode 'strict'")
        cm1 = seq[-1].copy()
        cm1[0] = lightness_range[1]
        seq = np.concatenate([seq, cm1[None]], axis=0)

    ratios = np.diff((seq[:, 0] - lightness_range[0]) / (lightness_range[1] - lightness_range[0]))
    parts = []
    for i in range(len(seq) - 1):
        parts.append(homotopy(seq[i], seq[i + 1], num=int(steps * ratios[i])))
    return lab2rgb(np.concatenate(parts))


lightness_range = 5, 95
bb = rgb2lab(mix(bimosblack, alpha=False))
bb[0] = lightness_range[0]
bb = lab2rgb(bb)
br = rgb2lab(mix(bimosred, alpha=False))
br[0] = (lightness_range[0] + lightness_range[1]) / 2
br = lab2rgb(br)
by = rgb2lab(mix(bimosyellow, alpha=False))
by[0] = lightness_range[1]
by = lab2rgb(by)

# cmap = sequential_cmap(bb, mix(bimosblack, 20, bimosred), mix(bimosred, 20, br), mix(br, 80, example_text_fg), bimosyellow, lightness_range=lightness_range, strict=True)
cmap = sequential_cmap(
    bb,
    mix(bimosblack, 30, bimosred, 80, br),
    mix(bimosred, 20, br, 80, example_text_fg),
    mix(br, 80, example_text_fg),
    bimosyellow,
    lightness_range=lightness_range,
    strict=True,
)
BIMoSmap = mpl.colors.LinearSegmentedColormap.from_list("BIMoS", cmap)
