import numpy as np
def _orth(X: np.ndarray) -> np.ndarray:
    """
    Computes an orthonormal basis for the column space of X.
    Mirrors scipy.linalg.orth behavior without adding a SciPy dependency here.
    """
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        return np.zeros_like(X)
    tol = np.max(X.shape) * np.max(s) * np.finfo(s.dtype).eps
    rank = int(np.sum(s > tol))
    return u[:, :rank]


def makeRaisedCosBasis(nb: int, dt: float, endpoints: list[float], b: float, zflag: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
      nb:         Number of basis vectors.
      dt:         Time step in seconds.
      endpoints:  [first_peak, last_peak] in seconds.
      b:          Offset for log-time warping.
      zflag:      Optional flag from the MATLAB function.

    Returns:
      A tuple (iht, ihbas, ihbasis)
        iht:      Time grid in seconds.
        ihbas:    Orthogonalized basis.
        ihbasis:  Raw raised-cosine basis.
    """
    if b <= 0:
        raise ValueError("b must be greater than 0")
    if dt <= 0:
        raise ValueError("dt must be greater than 0")
    if nb < 1:
        raise ValueError("nb must be positive")
    if len(endpoints) != 2:
        raise ValueError("endpoints must have length 2")

    endpoints = np.asarray(endpoints, dtype=float)
    nlin = lambda x: np.log(x)
    invnl = lambda x: np.exp(x) - 1e-20

    if zflag == 2:
        nb = nb - 1
        if nb < 1:
            raise ValueError("zflag=2 requires nb >= 2")

    yrnge = nlin(endpoints + b)
    db = (yrnge[1] - yrnge[0]) / (nb - 1) if nb > 1 else 1.0
    ctrs = np.linspace(yrnge[0], yrnge[1], nb)
    mxt = invnl(yrnge[1] + 2 * db) - b
    # iht = np.arange(0, mxt + dt * 0.5, dt)
    nT = int(np.floor(mxt / dt)) + 1
    iht = np.arange(nT) * dt

    x = nlin(iht + b)[:, None]
    c = ctrs[None, :]
    arg = (x - c) * np.pi / (2 * db)
    arg = np.clip(arg, -np.pi, np.pi)
    ihbasis = (np.cos(arg) + 1) / 2

    if zflag == 1:
        ii = iht <= endpoints[0]
        ihbasis[ii, 0] = 1
    elif zflag == 2:
        # MATLAB version uses an absref field in this branch; not used in this repo.
        pass

    ihbas = _orth(ihbasis)
    return iht, ihbas, ihbasis
