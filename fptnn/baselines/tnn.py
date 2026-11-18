"""Low-tubal-rank baselines.

TNN   : order-p tensor nuclear norm denoising, X = argmin 1/2||X-Y||^2 + tau ||X||_TNN.
        For this Frobenius data term the ADMM of the paper's baseline converges to the
        closed-form proximal operator, i.e. t-SVT in the Fourier domain (Lu et al. 2020).
F-tSVT: the same one-step SVT but in the framelet domain (handcrafted counterpart of
        FPTNN; cf. framelet-TNN of Jiang et al. 2020).
Both are tuned with an oracle grid search on tau (best SNR), which favours the baselines.
"""
import numpy as np

from ..framelet import ModeTransform
from ..metrics import snr_db
from ..tensor_ops import tsvt_fft, tsvt_transform


def _perm(Y, image_modes):
    perm = list(image_modes) + [m for m in range(Y.ndim) if m not in image_modes]
    return perm, np.argsort(perm)


def tnn_denoise(Y, sigma, image_modes=(0, 1), c=1.0):
    perm, inv = _perm(Y, image_modes)
    Yp = np.transpose(Y, perm)
    n1, n2 = Yp.shape[:2]
    N3 = int(np.prod(Yp.shape[2:]))
    # noise singular values of an n1 x n2 frequency slice ~ sigma sqrt(N3)(sqrt(n1)+sqrt(n2))
    tau = c * sigma * np.sqrt(N3) * (np.sqrt(n1) + np.sqrt(n2))
    return np.transpose(tsvt_fft(Yp, tau), inv)


def ftsvt_denoise(Y, sigma, image_modes=(0, 1), c=1.0, ftype="linear", levels=1, bc="symmetric"):
    perm, inv = _perm(Y, image_modes)
    Yp = np.transpose(Y, perm)
    T = ModeTransform(Yp.shape, range(2, Y.ndim), "framelet", ftype, levels, bc)
    n1, n2 = Yp.shape[:2]
    std = np.broadcast_to(T.coef_noise_std(sigma), (1, 1) + T.out_shape()[2:]).reshape(-1)
    tau = c * std * (np.sqrt(n1) + np.sqrt(n2))
    return np.transpose(tsvt_transform(Yp, tau, T), inv)


def oracle_grid(fn, Y, X, grid, **kw):
    """Return (best output, best c, list of (c, snr))."""
    best, best_c, best_s, curve = None, None, -np.inf, []
    for c in grid:
        out = fn(Y, c=c, **kw)
        s = snr_db(X, out)
        curve.append((float(c), s))
        if s > best_s:
            best, best_c, best_s = out, float(c), s
    return best, best_c, curve
