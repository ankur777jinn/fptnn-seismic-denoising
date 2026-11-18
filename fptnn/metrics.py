"""SNR, noise generation, MAD noise estimation and local similarity."""
import numpy as np
from scipy.ndimage import uniform_filter1d


def snr_db(clean, est):
    """Eq. (23): SNR = 20 log10(||X|| / ||X - X_hat||)."""
    return float(20 * np.log10(np.linalg.norm(clean) / np.linalg.norm(clean - est)))


def add_noise_snr(clean, snr_target_db, seed=0):
    """Add white Gaussian noise so that 20log10(||X||/||N||) equals the target.
    Returns (noisy, noise, sigma)."""
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(clean.shape)
    n *= np.linalg.norm(clean) / np.linalg.norm(n) / (10 ** (snr_target_db / 20))
    sigma = float(np.linalg.norm(n) / np.sqrt(n.size))
    return clean + n, n, sigma


def haar_hh(slices):
    """Finest diagonal (HH) coefficients of an orthonormal 2-D Haar transform of
    a batch of 2-D slices (N x n1 x n2).  Odd edges are cropped."""
    N, a, b = slices.shape
    s = slices[:, : a - a % 2, : b - b % 2]
    return (s[:, 0::2, 0::2] - s[:, 1::2, 0::2] - s[:, 0::2, 1::2] + s[:, 1::2, 1::2]) / 2.0


def mad_sigma(coeffs):
    """Donoho's robust estimator sigma = median(|HH|)/0.6745 (Eq. (24))."""
    return float(np.median(np.abs(coeffs)) / 0.6745)


# ---------------------------------------------------------------- local similarity
def _tri_smooth(x, rect):
    """Separable triangle smoothing (box filter applied twice), zero boundaries
    so the operator is self-adjoint."""
    for ax, r in enumerate(rect):
        if r > 1:
            size = 2 * (r // 2) + 1
            x = uniform_filter1d(x, size, axis=ax, mode="constant")
            x = uniform_filter1d(x, size, axis=ax, mode="constant")
    return x


def _shaped_division(num, den, rect, niter=30, lam2=None):
    """Smooth division c ~= num/den with shaping regularization (Fomel, 2007):
    c = H [lam^2 I + H^T (D^2 - lam^2 I) H]^{-1} H^T num,  D = diag(den),
    solved by conjugate gradients.  lam^2 defaults to mean(den^2) (as in sfdivn)."""
    if lam2 is None:
        lam2 = float(np.mean(den ** 2)) + 1e-30
    H = lambda v: _tri_smooth(v, rect)
    d2 = den ** 2

    def A(p):
        return lam2 * p + H((d2 - lam2) * H(p))

    b = H(num)
    p = np.zeros_like(num)
    r = b - A(p)
    z = r.copy()
    rr = np.vdot(r, r).real
    for _ in range(niter):
        Az = A(z)
        alpha = rr / (np.vdot(z, Az).real + 1e-30)
        p = p + alpha * z
        r = r - alpha * Az
        rr_new = np.vdot(r, r).real
        if rr_new < 1e-20 * (np.vdot(b, b).real + 1e-30):
            break
        z = r + (rr_new / rr) * z
        rr = rr_new
    return H(p)


def local_similarity(a, b, rect, niter=30):
    """Local similarity (Fomel, 2007; Chen & Fomel, 2015) between a (denoised
    signal) and b (removed noise):
    c1 = argmin ||diag(a) c1 - b||, c2 = argmin ||diag(b) c2 - a||, both with
    shaping regularization (triangle smoother of radii `rect`); similarity = c1 o c2.
    ~0 means no signal leakage into the removed noise; ~1 means strong leakage."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / (np.sqrt(np.mean(a ** 2)) + 1e-30)
    b = b / (np.sqrt(np.mean(b ** 2)) + 1e-30)
    c1 = _shaped_division(a * b, a, rect, niter)  # solves a*c1 ~ b  (num = a*b, den = a)
    c2 = _shaped_division(a * b, b, rect, niter)  # solves b*c2 ~ a
    return c1 * c2
