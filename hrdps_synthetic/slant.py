"""Differential slant delay from two epoch zenith-delay cubes.

Only the projected mapping is implemented (the agreed default until RAiDER
raytracing is validated):

    dSTD = (ZTD_ref - ZTD_sec)(lat, lon, h) / cos(theta)

with ZTD = wet_total + hydro_total from RAiDER. Both epochs of an AOI share one
lat/lon/z grid, so the cubes are differenced first and interpolated once.
Interpolation is trilinear on the epoch grid (the same (y, x, z) linear scheme
as ``RAiDER.delayFcns.getInterpolators``), so differencing first is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates

SLANT_METHODS = ('projected',)


@dataclass
class Epoch:
    path: Path
    total: np.ndarray  # (z, y, x) float32, zenith delay from z to model top (m)
    x: np.ndarray  # lon, ascending, regular
    y: np.ndarray  # lat, ascending, regular
    z: np.ndarray  # height (m), ascending

    @classmethod
    def load(cls, path: Path) -> 'Epoch':
        with xr.open_dataset(path) as ds:
            total = (ds['wet_total'].values.astype(np.float64) + ds['hydro_total'].values).astype(np.float32)
            return cls(Path(path), total, ds.x.values.copy(), ds.y.values.copy(), ds.z.values.copy())


def regular_axis(a: np.ndarray, name: str) -> tuple[float, float]:
    d = np.diff(a)
    if d.size == 0 or not np.allclose(d, d[0], rtol=1e-5, atol=1e-9) or d[0] <= 0:
        raise ValueError(f'Epoch {name} axis is not regular and ascending')
    return float(a[0]), float(d.mean())


def delta_ztd(ref: Epoch, sec: Epoch) -> np.ndarray:
    for k in ('x', 'y', 'z'):
        if not np.array_equal(getattr(ref, k), getattr(sec, k)):
            raise ValueError(f'Epoch grids differ on {k}: {ref.path} vs {sec.path}')
    return ref.total - sec.total


def epoch_indices(epoch: Epoch, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (x, y) indices of lon/lat points in the epoch grid."""
    x0, dx = regular_axis(epoch.x, 'x')
    y0, dy = regular_axis(epoch.y, 'y')
    return (np.asarray(lon) - x0) / dx, (np.asarray(lat) - y0) / dy


def interpolate_cube(cube: np.ndarray, z: np.ndarray, xi: np.ndarray, yi: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a (z, y, x) cube at fractional x/y indices and heights h."""
    ny, nx = cube.shape[1:]
    if xi.min() < 0 or yi.min() < 0 or xi.max() > nx - 1 or yi.max() > ny - 1:
        raise ValueError('Scene extends beyond the epoch grid')
    if np.nanmin(h) < z[0] or np.nanmax(h) > z[-1]:
        raise ValueError(f'Heights {np.nanmin(h):.0f}..{np.nanmax(h):.0f} m outside the epoch z range')
    zi = np.interp(h, z, np.arange(z.size)).astype(np.float32)
    return map_coordinates(cube, [zi, yi, xi], order=1, mode='nearest', prefilter=False, output=np.float32)


def project(dztd: np.ndarray, theta_deg: np.ndarray, method: str = 'projected') -> np.ndarray:
    if method != 'projected':
        raise NotImplementedError(f'slant method {method!r}; only "projected" is implemented')
    return (dztd / np.cos(np.radians(theta_deg))).astype(np.float32)


def to_phase(dstd: np.ndarray, wavelength: float) -> tuple[np.ndarray, np.ndarray]:
    """Unwrapped phase = 4*pi/lambda * dSTD (radians) and its wrap to [-pi, pi)."""
    unw = (4.0 * math.pi / wavelength) * dstd.astype(np.float64)
    wrp = np.mod(unw + math.pi, 2.0 * math.pi) - math.pi
    return unw.astype(np.float32), wrp.astype(np.float32)
