"""Right-rectangular prism kernels (Nagy et al., 2000) for gravity, gradient tensor and magnetics.

Coordinates: x east, y north, z DOWN (m). Prisms are rows [x1, x2, y1, y2, z1, z2].
Observation points are rows [x, y, z] with z down (an observation at elevation h has z = -h).
"""

from __future__ import annotations

import numpy as np

G = 6.674e-11
CM = 1e-7
SI_TO_MGAL = 1e5
_EPS = 1e-9


def _corners(obs: np.ndarray, prisms: np.ndarray):
    """Yield (x, y, z, sign) arrays of shape (n_obs, n_prisms) for the 8 prism corners."""

    for i, xi in enumerate((0, 1)):
        for j, yj in enumerate((2, 3)):
            for k, zk in enumerate((4, 5)):
                x = prisms[None, :, xi] - obs[:, None, 0]
                y = prisms[None, :, yj] - obs[:, None, 1]
                z = prisms[None, :, zk] - obs[:, None, 2]
                x = np.where(np.abs(x) < _EPS, _EPS, x)
                y = np.where(np.abs(y) < _EPS, _EPS, y)
                z = np.where(np.abs(z) < _EPS, _EPS, z)
                yield x, y, z, (-1.0) ** (i + j + k)


def _safe_log(v: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(v, 1e-300))


def gz_kernel(obs: np.ndarray, prisms: np.ndarray) -> np.ndarray:
    """Vertical attraction per unit density (m/s^2 per kg/m^3), shape (n_obs, n_prisms)."""

    obs = np.atleast_2d(np.asarray(obs, float))
    prisms = np.atleast_2d(np.asarray(prisms, float))
    total = np.zeros((obs.shape[0], prisms.shape[0]))
    for x, y, z, s in _corners(obs, prisms):
        r = np.sqrt(x * x + y * y + z * z)
        total += s * (x * _safe_log(y + r) + y * _safe_log(x + r) - z * np.arctan(x * y / (z * r)))
    return G * total


def tensor_kernels(obs: np.ndarray, prisms: np.ndarray) -> dict[str, np.ndarray]:
    """Second derivatives of the Newtonian volume integral Phi = int 1/r dV (no G, rho)."""

    obs = np.atleast_2d(np.asarray(obs, float))
    prisms = np.atleast_2d(np.asarray(prisms, float))
    shape = (obs.shape[0], prisms.shape[0])
    t = {key: np.zeros(shape) for key in ("xx", "yy", "zz", "xy", "xz", "yz")}
    for x, y, z, s in _corners(obs, prisms):
        r = np.sqrt(x * x + y * y + z * z)
        t["xx"] += s * np.arctan(y * z / (x * r))
        t["yy"] += s * np.arctan(x * z / (y * r))
        t["zz"] += s * np.arctan(x * y / (z * r))
        t["xy"] -= s * _safe_log(z + r)
        t["xz"] -= s * _safe_log(y + r)
        t["yz"] -= s * _safe_log(x + r)
    return t


def unit_vector(inclination: float, declination: float) -> np.ndarray:
    inc, dec = np.radians(inclination), np.radians(declination)
    return np.array([np.cos(inc) * np.sin(dec), np.cos(inc) * np.cos(dec), np.sin(inc)])


def magnetic_kernel(
    obs: np.ndarray,
    prisms: np.ndarray,
    inclination: float,
    declination: float,
    mag_inclination: float | None = None,
    mag_declination: float | None = None,
) -> np.ndarray:
    """Total-field anomaly (nT) per unit magnetization (A/m): dT = Cm M f^T T m (Poisson relation)."""

    f = unit_vector(inclination, declination)
    m = unit_vector(
        inclination if mag_inclination is None else mag_inclination,
        declination if mag_declination is None else mag_declination,
    )
    t = tensor_kernels(obs, prisms)
    full = {
        (0, 0): t["xx"], (1, 1): t["yy"], (2, 2): t["zz"],
        (0, 1): t["xy"], (0, 2): t["xz"], (1, 2): t["yz"],
    }
    total = np.zeros_like(t["xx"])
    for i in range(3):
        for j in range(3):
            total += f[i] * m[j] * full[(min(i, j), max(i, j))]
    return CM * total * 1e9


def susceptibility_to_magnetization(susceptibility: float | np.ndarray, field_nT: float) -> np.ndarray:
    """Induced magnetization M = chi * F / mu0 (A/m) for SI susceptibility and field in nT."""

    return np.asarray(susceptibility) * field_nT * 1e-9 / (4e-7 * np.pi)


def prism_gravity(obs: np.ndarray, prisms: np.ndarray, density: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Vertical gravity (mGal) of prisms with densities (kg/m^3) at observation points."""

    obs = np.atleast_2d(np.asarray(obs, float))
    density = np.asarray(density, float)
    out = np.empty(obs.shape[0])
    for start in range(0, obs.shape[0], chunk):
        out[start:start + chunk] = gz_kernel(obs[start:start + chunk], prisms) @ density
    return out * SI_TO_MGAL
