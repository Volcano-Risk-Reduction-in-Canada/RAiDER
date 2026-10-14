"""Shared fixtures: synthetic epoch files shaped like the real ones."""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hrdps_synthetic import hrdps_archive as ha
from hrdps_synthetic.aois import load_aois, select_aois


def synthetic_ztd(z, lat, lon, scale=1.0, tilt=0.0):
    """Smooth, plausible zenith delays: exponential in height, linear horizontally."""
    Z, Y, X = np.meshgrid(z, lat, lon, indexing='ij')
    wet = scale * 0.08 * np.exp(-np.clip(Z, 0, None) / 2000.0) * (1 + tilt * (Y - lat.mean()))
    hydro = 2.3 * np.exp(-np.clip(Z, 0, None) / 7500.0) * (1 + 1e-3 * (X - lon.mean()))
    return wet.astype(np.float32), hydro.astype(np.float32)


def write_epoch(path: Path, aoi, scale=1.0, tilt=0.0):
    lats, lons = ha.regular_axes(aoi.raw_bounds())
    z = ha.FIXED_Z
    wet, hydro = synthetic_ztd(z, lats, lons, scale, tilt)
    ds = xr.Dataset({'wet_total': (('z', 'y', 'x'), wet), 'hydro_total': (('z', 'y', 'x'), hydro)},
                    coords={'z': z, 'y': lats, 'x': lons})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def write_fake_dem(workdir: Path, aoi, width_m: float, length_m: float, spacing: float = 10.0):
    """Smooth synthetic topography covering every scene position of `aoi` (no network)."""
    import rasterio
    from rasterio.transform import from_origin

    from hrdps_synthetic.dem import dem_extent, dem_path

    xmin, ymin, xmax, ymax = dem_extent(aoi, width_m, length_m, spacing)
    nx, ny = int(round((xmax - xmin) / spacing)), int(round((ymax - ymin) / spacing))
    x = xmin + (np.arange(nx) + 0.5) * spacing
    y = ymax - (np.arange(ny) + 0.5) * spacing
    h = (1000 + 800 * np.sin(x[None, :] / 3000.0) * np.cos(y[:, None] / 4000.0)).astype(np.float32)
    path = dem_path(workdir, aoi, spacing)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', dtype='float32', width=nx, height=ny, count=1,
                       crs=f'EPSG:{aoi.epsg}', transform=from_origin(xmin, ymax, spacing, spacing)) as dst:
        dst.write(h, 1)
    return path


@pytest.fixture(scope='session')
def all_aois():
    return load_aois()


@pytest.fixture
def fake_workdir(tmp_path, all_aois):
    """Workdir with synthetic epochs for 3 AOIs x 2 directions x 6 days."""
    aois = select_aois(all_aois, ['A4201_Volcano_Meager', 'BC_Lillooet', 'BC_Okanagan'])
    rng = np.random.default_rng(0)
    for a in aois:
        for d, t in (('asc', dt.time(2, 2)), ('desc', dt.time(14, 43))):
            for k in range(6):
                when = dt.datetime.combine(dt.date(2026, 10, 1) + dt.timedelta(days=k), t)
                write_epoch(ha.epoch_path(tmp_path, a.name, d, when), a,
                            scale=float(rng.uniform(0.5, 1.5)), tilt=float(rng.uniform(-0.5, 0.5)))
        write_fake_dem(tmp_path, a, 2000.0, 3000.0)  # DEMs sized for the 200 x 300 px test scenes
    return tmp_path, aois
