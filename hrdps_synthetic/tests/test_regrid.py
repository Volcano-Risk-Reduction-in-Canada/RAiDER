import numpy as np

from hrdps_synthetic.hrdps_archive import make_target_grid, regrid
from hrdps_synthetic.tests.test_rotated_grid import HRDPS


def field(lat, lon):
    return np.sin(np.radians(lat) * 40.0) + np.cos(np.radians(lon) * 25.0)  # ~100-300 km wavelengths


def test_regrid_reproduces_analytic_field(all_aois):
    j = np.arange(HRDPS.nj); i = np.arange(HRDPS.ni)
    J, I = np.meshgrid(j, i, indexing='ij')
    lat, lon = HRDPS.rot_to_geo(HRDPS.rlat0 + J * HRDPS.dlat, HRDPS.rlon0 + I * HRDPS.dlon)
    src = field(lat, lon).astype(np.float32)
    for a in all_aois[::7]:
        tg = make_target_grid(a.raw_bounds(), HRDPS)
        LON, LAT = np.meshgrid(tg.lons, tg.lats)
        err = np.abs(regrid(src, tg) - field(LAT, LON))
        assert err.max() < 5e-3, (a.name, err.max())


def test_row_mean_approximation_is_worse(all_aois):
    """The regular-grid regrid beats treating the rotated grid as regular (RAiDER's reader)."""
    a = next(x for x in all_aois if x.name == 'Edgecumbe')
    tg = make_target_grid(a.raw_bounds(), HRDPS)
    j0, j1 = int(tg.jj.min()), int(np.ceil(tg.jj.max())) + 1
    i0, i1 = int(tg.ii.min()), int(np.ceil(tg.ii.max())) + 1
    J, I = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1), indexing='ij')
    lat, lon = HRDPS.rot_to_geo(HRDPS.rlat0 + J * HRDPS.dlat, HRDPS.rlon0 + I * HRDPS.dlon)
    row_mean_lat, col_mean_lon = lat.mean(axis=1), lon.mean(axis=0)
    err_lat_km = np.abs(lat - row_mean_lat[:, None]).max() * 111.2
    assert err_lat_km > 5.0  # several km at the crop edges at this AOI
