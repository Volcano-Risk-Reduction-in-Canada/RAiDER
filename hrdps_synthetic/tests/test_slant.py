import math

import numpy as np
import pytest

from hrdps_synthetic.slant import Epoch, delta_ztd, epoch_indices, interpolate_cube, project, to_phase
from hrdps_synthetic.tests.conftest import write_epoch


@pytest.fixture
def epochs(tmp_path, all_aois):
    a = next(x for x in all_aois if x.name == 'A4201_Volcano_Meager')
    r = Epoch.load(write_epoch(tmp_path / 'r.nc', a, scale=1.3, tilt=0.4))
    s = Epoch.load(write_epoch(tmp_path / 's.nc', a, scale=0.7, tilt=-0.2))
    return a, r, s


def _points(a, n=500, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(a.lat - 0.1, a.lat + 0.1, n)
    lon = rng.uniform(a.lon - 0.1, a.lon + 0.1, n)
    h = rng.uniform(-20, 2700, n)
    return lat, lon, h


def test_matches_raider_interpolators(epochs):
    import xarray as xr
    from RAiDER.delayFcns import getInterpolators

    a, r, s = epochs
    lat, lon, h = _points(a)
    xi, yi = epoch_indices(r, lon, lat)
    ours = interpolate_cube(delta_ztd(r, s), r.z, xi.astype(np.float32), yi.astype(np.float32), h.astype(np.float32))
    pts = np.stack([lat, lon, h], axis=-1)
    total = []
    for ep in (r, s):
        ds = xr.open_dataset(ep.path)
        w, hy = getInterpolators(ds, 'total')
        total.append(w(pts) + hy(pts))
    assert np.allclose(ours, total[0] - total[1], atol=2e-6)


def test_projection_and_phase(epochs):
    a, r, s = epochs
    lat, lon, h = _points(a, 50)
    xi, yi = epoch_indices(r, lon, lat)
    d = interpolate_cube(delta_ztd(r, s), r.z, xi, yi, h.astype(np.float32))
    assert np.array_equal(project(d, np.zeros_like(d)), d)
    th = np.full_like(d, 40.0)
    assert np.allclose(project(d, th), d / math.cos(math.radians(40.0)))
    unw, wrp = to_phase(project(d, th), 0.055466)
    assert np.allclose(unw, 4 * math.pi / 0.055466 * d / math.cos(math.radians(40.0)), rtol=1e-5)
    k = (unw.astype(np.float64) - wrp) / (2 * math.pi)
    assert np.allclose(k, np.round(k), atol=1e-4) and wrp.min() >= -math.pi and wrp.max() < math.pi


def test_out_of_grid_raises(epochs):
    a, r, s = epochs
    with pytest.raises(ValueError):
        interpolate_cube(delta_ztd(r, s), r.z, np.array([1e6], np.float32), np.array([0.0], np.float32),
                         np.array([0.0], np.float32))
