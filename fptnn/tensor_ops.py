"""Order-p tensor algebra: t-product (time, Fourier and generic transform domain), t-SVT.

Notation follows Qian et al. (2022), Appendix A, and Martin, Shafer & LaRue (2013).
A tensor X in R^{n1 x n2 x n3 x ... x np}; frontal slices are X(:, :, i3, ..., ip).
"""
import numpy as np


def unfold(X, mode):
    """Mode-n unfolding X_(n) (n_mode x prod(other dims))."""
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def fold(M, mode, shape):
    shp = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(M.reshape(shp), 0, mode)


# ---------------------------------------------------------------- t-products
def tprod_definition(A, B):
    """Order-p t-product by its recursive block-circulant definition (Defs. 5-6):
    A * B = fold(circ(A) . unfold(B)), circulant over the last mode, recursing on
    order-(p-1) t-products of the slices.  O(n^2) per level -- only for checks."""
    if A.ndim == 2:
        return A @ B
    n = A.shape[-1]
    out = None
    for i in range(n):
        acc = None
        for j in range(n):
            term = tprod_definition(A[..., (i - j) % n], B[..., j])
            acc = term if acc is None else acc + term
        if out is None:
            out = np.zeros(acc.shape + (n,), dtype=np.result_type(A, B))
        out[..., i] = acc
    return out


def _slicewise_matmul(Ah, Bh):
    """C(:,:,l) = A(:,:,l) B(:,:,l) for every frontal index l (Defs. 2 and 7)."""
    return np.einsum("ir...,rj...->ij...", Ah, Bh)


def tprod_fft(A, B):
    """Order-p t-product via the convolution theorem (Def. 7): FFT along modes 3..p,
    frontal-slice matrix products, inverse FFT."""
    axes = tuple(range(2, A.ndim))
    Ah = np.fft.fftn(A, axes=axes)
    Bh = np.fft.fftn(B, axes=axes)
    C = np.fft.ifftn(_slicewise_matmul(Ah, Bh), axes=axes)
    return C.real


def tprod_transform(A, B, mats):
    """Transform-domain order-p product (Def. 2): C~ = A~ (.) B~ slice-wise, then
    C = C~ x_m M_m^T.  `mats[m]` is the analysis matrix of mode m (m >= 2), e.g. the
    framelet matrix W.  For an orthonormal M this is the Kernfeld-Kilmer-Aeron
    *_M product; for a Parseval frame (W^T W = I) it is the framelet t-product."""
    from .framelet import mode_product
    Ah, Bh = A, B
    for m, M in mats.items():
        Ah = mode_product(Ah, M, m)
        Bh = mode_product(Bh, M, m)
    Ch = _slicewise_matmul(Ah, Bh)
    for m in sorted(mats, reverse=True):
        Ch = mode_product(Ch, mats[m].T, m)
    return Ch


# ---------------------------------------------------------------- t-SVT (TNN prox)
def _svt_batched(S, tau):
    """Singular value thresholding of a batch of matrices S (L x n1 x n2)."""
    U, s, Vh = np.linalg.svd(S, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    return np.einsum("lik,lk,lkj->lij", U, s, Vh)


def tsvt_fft(Y, tau):
    """Proximal operator of tau * TNN (Lu et al., 2020 convention, order-p):
    FFT along modes 3..p, SVT of every frontal slice, inverse FFT.
    Frontal slices are modes (1,2) of Y."""
    axes = tuple(range(2, Y.ndim))
    Yh = np.fft.fftn(Y, axes=axes)
    n1, n2 = Y.shape[:2]
    S = np.moveaxis(Yh.reshape(n1, n2, -1), 2, 0)
    Xh = _svt_batched(S, tau)
    Xh = np.moveaxis(Xh, 0, 2).reshape(Yh.shape)
    return np.fft.ifftn(Xh, axes=axes).real


def tsvt_transform(Y, tau, T):
    """One-step transform-domain SVT with a ModeTransform T (analysis -> SVT -> W^T).
    For a framelet T this is the handcrafted (non-learned) counterpart of FPTNN."""
    Yt = T.forward(Y)
    n1, n2 = Yt.shape[:2]
    S = np.moveaxis(Yt.reshape(n1, n2, -1), 2, 0)
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), (S.shape[0],))
    U, s, Vh = np.linalg.svd(S, full_matrices=False)
    s = np.maximum(s - tau_arr[:, None], 0.0)
    Xt = np.einsum("lik,lk,lkj->lij", U, s, Vh)
    Xt = np.moveaxis(Xt, 0, 2).reshape(Yt.shape)
    return T.inverse(Xt)
