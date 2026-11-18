"""Framelet (tight wavelet frame) matrices and mode-n transforms.

A univariate tight framelet system with K filters {h_0 (low-pass), h_1..h_{K-1}}
satisfies the unitary extension principle (UEP):  sum_k |h_k^(w)|^2 = 1 for all w.
The undecimated multi-level analysis operator W in R^{w n x n},
w = (K-1) L + 1, then satisfies W^T W = I (perfect reconstruction, Parseval frame),
while W W^T != I (redundancy).  This is the matrix W of Eq. (8) and Appendix B
of Qian et al. (2022).

Also provided (for ablations): orthonormal DCT-II, the discrete Hartley transform
(a real, orthonormal stand-in for the DFT) and the identity.
"""
import numpy as np

SQ2 = np.sqrt(2.0)
SQ6 = np.sqrt(6.0)

# Ron-Shen B-spline tight framelet filters (taps listed left->right, centred)
FILTERS = {
    "haar": [np.array([1.0, 1.0]) / 2, np.array([1.0, -1.0]) / 2],
    "linear": [np.array([1.0, 2.0, 1.0]) / 4,
               np.array([1.0, 0.0, -1.0]) * SQ2 / 4,
               np.array([-1.0, 2.0, -1.0]) / 4],
    "cubic": [np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16,
              np.array([1.0, 2.0, 0.0, -2.0, -1.0]) / 8,
              np.array([1.0, 0.0, -2.0, 0.0, 1.0]) * SQ6 / 16,
              np.array([-1.0, 2.0, 0.0, -2.0, 1.0]) / 8,
              np.array([1.0, -4.0, 6.0, -4.0, 1.0]) / 16],
}


def uep_residual(ftype, nfreq=512):
    """max_w |sum_k |H_k(w)|^2 - 1| : 0 for a valid tight framelet filter bank."""
    w = np.linspace(-np.pi, np.pi, nfreq)
    tot = np.zeros_like(w)
    for h in FILTERS[ftype]:
        k = np.arange(len(h)) - (len(h) - 1) / 2.0
        H = (h[None, :] * np.exp(-1j * w[:, None] * k[None, :])).sum(1)
        tot += np.abs(H) ** 2
    return float(np.abs(tot - 1).max())


def _conv_matrix(h, n, step=1, bc="periodic"):
    """n x n matrix of the (a-trous, dilation `step`) filter h.

    periodic : circulant (exact tight frame for any n)
    symmetric: half-sample symmetric (Neumann) extension
    """
    L = len(h)
    offs = (np.arange(L) - (L - 1) // 2) * step  # centred taps (left-biased for even L)
    M = np.zeros((n, n))
    for i in range(n):
        for tap, o in zip(h, offs):
            j = i + o
            if bc == "periodic":
                j %= n
            elif bc == "symmetric":
                period = 2 * n
                j %= period
                if j >= n:
                    j = period - 1 - j
            else:
                raise ValueError(bc)
            M[i, j] += tap
    return M


def framelet_matrix(n, ftype="linear", levels=1, bc="periodic", lowpass_first=True):
    """Undecimated multi-level framelet analysis matrix W ((K-1)L+1)n x n.

    Returns W and `band_of_row` (band index of every row, 0 = final low-pass if
    lowpass_first).
    """
    hs = FILTERS[ftype]
    K = len(hs)
    low = np.eye(n)
    highs = []
    for lev in range(levels):
        step = 2 ** lev
        H = [_conv_matrix(h, n, step, bc) for h in hs]
        for k in range(1, K):
            highs.append(H[k] @ low)
        low = H[0] @ low
    blocks = [low] + highs if lowpass_first else highs + [low]
    W = np.vstack(blocks)
    band_of_row = np.repeat(np.arange(len(blocks)), n)
    return W, band_of_row


def dct_matrix(n):
    """Orthonormal DCT-II matrix (n x n)."""
    k = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    C = np.cos(np.pi * (j + 0.5) * k / n) * np.sqrt(2.0 / n)
    C[0] /= SQ2
    return C


def hartley_matrix(n):
    """Orthonormal discrete Hartley transform: real, symmetric, H^T H = I."""
    k = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    a = 2 * np.pi * k * j / n
    return (np.cos(a) + np.sin(a)) / np.sqrt(n)


def transform_matrix(n, kind="framelet", ftype="linear", levels=1, bc="periodic"):
    """Return (M, band_of_row) for a mode of length n."""
    if kind == "framelet":
        return framelet_matrix(n, ftype, levels, bc)
    if kind == "dct":
        return dct_matrix(n), np.zeros(n, dtype=int)
    if kind == "hartley":
        return hartley_matrix(n), np.zeros(n, dtype=int)
    if kind == "identity":
        return np.eye(n), np.zeros(n, dtype=int)
    raise ValueError(kind)


def mode_product(X, M, mode):
    """fold_mode(M @ unfold_mode(X)): multiply tensor X by matrix M along `mode`."""
    Y = np.tensordot(M, X, axes=([1], [mode]))
    return np.moveaxis(Y, 0, mode)


class ModeTransform:
    """Separable linear transform applied along a set of tensor modes.

    forward:  X~ = X x_{m1} M_{m1} x_{m2} M_{m2} ...   (Eq. (8), Alg. 1 lines 2-4)
    inverse:  X  = X~ x_{m} M_m^T ...                  (Alg. 1 lines 17-19)
    The inverse is exact whenever every M_m^T M_m = I (orthonormal or Parseval).
    """

    def __init__(self, shape, modes, kind="framelet", ftype="linear", levels=1,
                 bc="periodic", dtype=np.float64):
        self.shape = tuple(shape)
        self.modes = list(modes)
        self.mats, self.bands = {}, {}
        for m in self.modes:
            M, b = transform_matrix(self.shape[m], kind, ftype, levels, bc)
            self.mats[m] = M.astype(dtype)
            self.bands[m] = b

    def forward(self, X):
        for m in self.modes:
            X = mode_product(X, self.mats[m], m)
        return X

    def inverse(self, Xt):
        for m in reversed(self.modes):
            Xt = mode_product(Xt, self.mats[m].T, m)
        return Xt

    def out_shape(self):
        s = list(self.shape)
        for m in self.modes:
            s[m] = self.mats[m].shape[0]
        return tuple(s)

    def row_norms(self, m):
        return np.linalg.norm(self.mats[m], axis=1)

    def coef_noise_std(self, sigma=1.0):
        """Std of every transformed coefficient when the input noise is i.i.d. N(0, sigma^2).

        Returned with broadcastable shape over the transformed tensor; within each
        frontal slice (modes not transformed) the noise stays white."""
        shp = [1] * len(self.shape)
        out = np.array(sigma, dtype=float).reshape([1] * len(self.shape))
        for m in self.modes:
            s = list(shp)
            s[m] = -1
            out = out * self.row_norms(m).reshape(s)
        return out

    def band_index(self):
        """Integer band label of every transformed coefficient (product of mode bands)."""
        lab = np.zeros([1] * len(self.shape), dtype=np.int64)
        mult = 1
        for m in self.modes:
            s = [1] * len(self.shape)
            s[m] = -1
            b = self.bands[m].reshape(s)
            lab = lab + b * mult
            mult *= int(self.bands[m].max()) + 1
        return lab, mult
