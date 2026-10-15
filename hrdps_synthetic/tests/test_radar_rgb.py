"""Radar-geometry layout, RGB rendering, phase sign, and the RGB post-processor."""

import csv
import json
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from hrdps_synthetic import generate, rgb
from hrdps_synthetic.aois import radar_basis
from hrdps_synthetic.geometry import incidence_field, scene_for
from hrdps_synthetic.tables import CsvAppender, read_rows
from hrdps_synthetic.tests.conftest import fake_height, write_epoch


def _aoi(all_aois, name='A4201_Volcano_Meager'):
    return next(a for a in all_aois if a.name == name)


def test_radar_orientation_matches_slc_convention(all_aois):
    a = _aoi(all_aois)
    asc = scene_for(a, 0, 0, geometry='radar', heading=348.0)
    x0, y0 = asc.pixel_xy(0, 0)
    xr, yr = asc.pixel_xy(1, 0)   # next row: flight direction (NNW)
    xc, yc = asc.pixel_xy(0, 1)   # next column: farther range (ENE, right of track)
    assert yr > y0 and xr < x0 and xc > x0 and yc > y0
    desc = scene_for(a, 0, 0, geometry='radar', heading=192.0)
    x0, y0 = desc.pixel_xy(0, 0)
    xr, yr = desc.pixel_xy(1, 0)  # flight direction (SSW)
    xc, yc = desc.pixel_xy(0, 1)  # farther range (WNW)
    assert yr < y0 and xr < x0 and xc < x0 and yc > y0
    assert asc.shape == (3000, 2000) and asc.crs is None
    assert asc.transform == Affine(10, 0, -10000, 0, -10, 15000)  # magmatic local frame


def test_radar_positions_match_magmatic_basis(all_aois):
    """Same flight/look basis as the magmatic sweep: east = f_e*az + l_e*rg, north = f_n*az + l_n*rg."""
    a = _aoi(all_aois)
    s = scene_for(a, 0, 0, geometry='radar', heading=192.0)
    h = math.radians(s.flight_az_grid_deg)
    f, l = (math.sin(h), math.cos(h)), (math.cos(h), -math.sin(h))
    fb, lb = radar_basis(s.flight_az_grid_deg)
    assert fb == pytest.approx(f) and lb == pytest.approx(l)
    r, c = 1234, 567
    az, rg = (r + 0.5 - s.ny / 2) * 10, (c + 0.5 - s.nx / 2) * 10
    x, y = s.pixel_xy(r, c)
    assert x == pytest.approx(s.cx + f[0] * az + l[0] * rg) and y == pytest.approx(s.cy + f[1] * az + l[1] * rg)


def test_radar_incidence_depends_on_range_only(all_aois):
    s = scene_for(_aoi(all_aois), 0, 0, geometry='radar', heading=348.0)
    th = incidence_field(s, 35.0, 348.0)
    assert np.all(th[0] == th[-1]) and np.all(np.diff(th[0]) > 0)  # constant down columns, rising to far range
    assert abs(float(th[0, 999:1001].mean()) - 35.0) < 0.01


@pytest.mark.parametrize('heading', [348.0, 192.0])
def test_volcano_polygons_inside_rotated_scenes(all_aois, heading):
    from shapely.geometry import Polygon

    for a in all_aois:
        if a.kind != 'volcano':
            continue
        (dx0, dx1), (dy0, dy1) = a.offset_ranges(20_000, 30_000, 'radar', heading)
        f, l = (np.array(v) for v in radar_basis(a.flight_az_grid(heading)))
        xy = np.asarray(a.poly_utm.exterior.coords) - (a.cx, a.cy)
        fits = np.ptp(xy @ l) <= 20_000 and np.ptp(xy @ f) <= 30_000
        if not fits:
            continue
        for dx in (dx0, dx1):
            for dy in (dy0, dy1):
                fp = Polygon(a.footprint_corners(dx, dy, 20_000, 30_000, 'radar', heading)).buffer(1e-3)
                assert fp.contains(a.poly_utm), (a.name, heading)


def test_existing_epoch_crops_cover_radar_scenes(all_aois):
    """Crops sized for north-up scenes also cover rotated radar scenes, so no re-download is needed."""
    for a in all_aois:
        S, N, W, E = a.raw_bounds()
        w, s, e, n = a.reach_lonlat(geometry='radar').bounds
        assert W < w and E > e and S < s and N > n, a.name


def test_radar_dem_resampling(tmp_path, all_aois):
    from hrdps_synthetic.dem import read_scene_dem
    from hrdps_synthetic.tests.conftest import write_fake_dem

    a = _aoi(all_aois, 'BC_Lillooet')
    path = write_fake_dem(tmp_path, a, 20_000, 30_000)
    s = scene_for(a, 1234, -2345, geometry='radar', heading=192.0)
    h = read_scene_dem(path, s)
    x, y = s.pixel_xy(np.arange(0, s.ny, 97)[:, None], np.arange(0, s.nx, 89)[None, :])
    assert np.abs(h[::97, ::89] - fake_height(x, y)).max() < 0.5  # bilinear on a smooth 10 m field


def test_rgb_ramp():
    assert tuple(rgb.phase_to_rgb(np.zeros((1, 1)))[:, 0, 0]) == (255, 0, 0)  # 0 rad = red
    assert tuple(rgb.phase_to_rgb(np.full((1, 1), -np.pi))[:, 0, 0]) == (0, 191, 169)
    stops = (rgb.RAMP_STOPS[:-1] * 2 * np.pi - np.pi).reshape(1, -1)
    assert np.array_equal(rgb.phase_to_rgb(stops)[:, 0, :].T, rgb.RAMP_RGB[:-1].astype(np.uint8))
    eps = 1e-6
    a = rgb.phase_to_rgb(np.array([[np.pi - eps, -np.pi + eps]]))
    assert np.abs(a[:, 0, 0].astype(int) - a[:, 0, 1]).max() <= 1  # cyclic


def test_phase_sign_positive_for_more_delay_on_later_date(tmp_path, all_aois):
    from hrdps_synthetic.sampling import PlanRow
    from hrdps_synthetic.slant import Epoch

    a = _aoi(all_aois)
    ref = Epoch.load(write_epoch(tmp_path / 'ref.nc', a, scale=0.5))
    sec = Epoch.load(write_epoch(tmp_path / 'sec.nc', a, scale=1.5))  # wetter later date
    row = PlanRow('s', 'train', a.name, a.region, 'asc', 348.0, '2026-10-01', '2026-10-02', '', '', 35.0, 0.0, 0.0)
    cfg = generate.GenConfig(workdir=str(tmp_path), aoi_file='', aoi_names=None, width_m=2000, length_m=3000)
    scene = scene_for(a, 0, 0, 2000, 3000, 10, 'radar', 348.0)
    dem = np.full(scene.shape, 500.0, np.float32)
    _, unw, wrp, img, meta = generate.render(row, cfg, a, ref, sec, dem)
    assert unw.min() > 0 and meta['phase_sign'] == 'sec_minus_ref'
    assert np.allclose(np.mod(unw + np.pi, 2 * np.pi) - np.pi, wrp, atol=1e-4)
    assert np.array_equal(img, rgb.phase_to_rgb(wrp))
    same = generate.render(row, cfg, a, ref, ref, dem)
    assert np.all(same[1] == 0) and np.all(same[3][:, 0, 0] == (255, 0, 0))  # no change: 0 rad, red


def _write_wrapped(path, value, tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', dtype='float32', width=8, height=6, count=1, crs=None,
                       transform=Affine(10, 0, 0, 0, -10, 0)) as dst:
        dst.write(np.full((6, 8), value, np.float32), 1)
        dst.update_tags(**tags)


def test_rgb_postprocessor(tmp_path):
    legacy = tmp_path / 'samples/train/X/legacy_wrp.tif'
    current = tmp_path / 'samples/train/X/current_wrp.tif'
    _write_wrapped(legacy, 1.0, {'convention': 'phase = 4*pi/wavelength * (STD_ref - STD_sec); ...'})
    _write_wrapped(current, 1.0, {'phase_sign': 'sec_minus_ref', 'convention': rgb.CONVENTION})
    with open(tmp_path / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'wrp_path'])
        w.writeheader()
        w.writerow({'sample_id': 'legacy', 'wrp_path': 'samples/train/X/legacy_wrp.tif'})
        w.writerow({'sample_id': 'current', 'wrp_path': 'samples/train/X/current_wrp.tif'})
    before = sorted(p.name for p in tmp_path.rglob('*'))
    assert rgb.main(['--workdir', str(tmp_path), '--dry-run']) == 0
    assert sorted(p.name for p in tmp_path.rglob('*')) == before
    assert rgb.main(['--workdir', str(tmp_path), '--workers', '1']) == 0
    with rasterio.open(tmp_path / 'samples/train/X/legacy_rgb.tif') as ds:
        assert tuple(ds.read()[:, 0, 0]) == tuple(rgb.phase_to_rgb(np.full((1, 1), -1.0))[:, 0, 0])  # flipped
        assert ds.tags()['rgb_phase_sign'] == 'sec_minus_ref' and ds.descriptions == ('red', 'green', 'blue')
    with rasterio.open(tmp_path / 'samples/train/X/current_rgb.tif') as ds:
        assert tuple(ds.read()[:, 0, 0]) == tuple(rgb.phase_to_rgb(np.full((1, 1), 1.0))[:, 0, 0])
    fields, rows = read_rows(tmp_path / 'manifest.csv')
    assert 'rgb_path' in fields and all(r['rgb_path'].endswith('_rgb.tif') for r in rows)
    assert rgb.main(['--workdir', str(tmp_path), '--sign', 'keep', '--workers', '1']) == 0  # skips existing


def test_csv_header_upgrade(tmp_path):
    p = tmp_path / 'm.csv'
    p.write_text('a,b\n1,2\n')
    out = CsvAppender(p, ['a', 'b', 'c'])
    out.write({'a': 3, 'b': 4, 'c': 5})
    out.close()
    fields, rows = read_rows(p)
    assert fields == ['a', 'b', 'c'] and rows == [{'a': '1', 'b': '2', 'c': ''}, {'a': '3', 'b': '4', 'c': '5'}]


def test_outputs_guard_refuses_mixing_conventions(fake_workdir):
    wd, aois = fake_workdir
    args = ['--workdir', str(wd), '--aois', *[a.name for a in aois], '--n-train', '4', '--n-val', '1',
            '--workers', '1', '--width-px', '200', '--height-px', '300', '--limit', '1']
    assert generate.main(args) == 0
    meta = json.loads((wd / 'outputs_meta.json').read_text())
    meta['phase_sign'] = 'ref_minus_sec'  # pretend the samples came from the old convention
    (wd / 'outputs_meta.json').write_text(json.dumps(meta))
    assert generate.main(args) == 2
