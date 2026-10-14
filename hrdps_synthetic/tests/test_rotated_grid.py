import os

import numpy as np
import pytest
from pyproj import CRS, Transformer

from hrdps_synthetic.hrdps_archive import RotatedGrid

HRDPS = RotatedGrid(ni=2540, nj=1290, rlat0=-12.302501, rlon0=345.17878 - 360.0, dlat=0.0225, dlon=0.0225,
                    sp_lat=-36.08852, sp_lon=245.305142)


def test_round_trip():
    lat = np.array([49.2, 50.62, 57.05, 59.6]); lon = np.array([-121.8, -123.53, -135.72, -133.4])
    rl, ro = HRDPS.geo_to_rot(lat, lon)
    lat2, lon2 = HRDPS.rot_to_geo(rl, ro)
    assert np.allclose(lat, lat2, atol=1e-9) and np.allclose(lon, lon2, atol=1e-9)


def test_matches_pyproj_rotated_pole():
    crs = CRS.from_cf({'grid_mapping_name': 'rotated_latitude_longitude',
                       'grid_north_pole_latitude': -HRDPS.sp_lat,
                       'grid_north_pole_longitude': HRDPS.sp_lon - 180.0})
    t = Transformer.from_crs(4326, crs, always_xy=True)
    lat = np.linspace(48, 60, 7); lon = np.linspace(-136, -114, 7)
    ro_p, rl_p = t.transform(lon, lat)
    rl, ro = HRDPS.geo_to_rot(lat, lon)
    assert np.allclose(rl, rl_p, atol=1e-7) and np.allclose(ro, ro_p, atol=1e-7)


def test_first_grid_point_matches_cfgrib():
    # cfgrib's 2-D latitude/longitude for HRDPS continental 2.5 km at (j=0, i=0)
    lat, lon = HRDPS.rot_to_geo(HRDPS.rlat0, HRDPS.rlon0)
    assert abs(lat - 39.626032) < 1e-5 and abs(lon - -133.62952) < 1e-5


def test_aois_inside_domain(all_aois):
    for a in all_aois:
        S, N, W, E = a.raw_bounds()
        for la, lo in ((S, W), (S, E), (N, W), (N, E)):
            j, i = HRDPS.geo_to_index(la, lo)
            assert 0 <= j <= HRDPS.nj - 1 and 0 <= i <= HRDPS.ni - 1, a.name


@pytest.mark.skipif(not os.environ.get('HRDPS_TEST_GRIB'), reason='set HRDPS_TEST_GRIB to a downloaded HRDPS GRIB2')
def test_full_grid_matches_cfgrib():
    import cfgrib
    from hrdps_synthetic.hrdps_archive import read_grib

    path = os.environ['HRDPS_TEST_GRIB']
    grid, vals, _ = read_grib(path)
    ds = cfgrib.open_dataset(path, indexpath='')
    j = np.arange(0, grid.nj, 37); i = np.arange(0, grid.ni, 41)
    J, I = np.meshgrid(j, i, indexing='ij')
    lat, lon = grid.rot_to_geo(grid.rlat0 + J * grid.dlat, grid.rlon0 + I * grid.dlon)
    clon = (ds.longitude.values[J, I] + 180) % 360 - 180
    assert np.abs(lat - ds.latitude.values[J, I]).max() < 1e-4
    assert np.abs(((lon - clon) + 180) % 360 - 180).max() < 1e-4
    assert np.allclose(vals, ds[list(ds.data_vars)[0]].values, equal_nan=True)
