"""Per-slice noise-level estimation in the transform domain (Eq. (24))."""
import numpy as np

from .metrics import haar_hh, mad_sigma


def slice_sigmas(Yt_slices, band_of_slice, mode, T=None, sigma_true=None, Y_orig_slices=None):
    """Noise std of every frontal slice of the transformed tensor.

    mode = 'mad_band'  : Eq. (24) -- median(|HH|)/0.6745 pooled over all slices of
                         the same framelet band (B bands in total)
    mode = 'mad_slice' : Eq. (24) applied to every slice separately
    mode = 'mad_global': one MAD estimate on the untransformed data, propagated
                         through the transform analytically (sigma * ||row of W||)
    mode = 'oracle'    : true sigma propagated analytically
    """
    N = Yt_slices.shape[0]
    if mode == "mad_band":
        sig = np.zeros(N)
        hh = haar_hh(Yt_slices)
        for b in np.unique(band_of_slice):
            sel = band_of_slice == b
            sig[sel] = mad_sigma(hh[sel])
        return sig
    if mode == "mad_slice":
        hh = haar_hh(Yt_slices)
        return np.median(np.abs(hh.reshape(N, -1)), axis=1) / 0.6745
    if mode in ("mad_global", "oracle"):
        if mode == "mad_global":
            s0 = mad_sigma(haar_hh(Y_orig_slices))
        else:
            s0 = float(sigma_true)
        std = T.coef_noise_std(s0)  # broadcastable over the transformed tensor
        std = np.broadcast_to(std, (1, 1) + T.out_shape()[2:])
        return np.ascontiguousarray(std).reshape(-1).astype(float)
    raise ValueError(mode)
