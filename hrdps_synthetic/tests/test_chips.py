import csv
import shutil

import numpy as np
import pytest
import rasterio

from hrdps_synthetic import generate, tile
from hrdps_synthetic.chips import axis_starts, chip_rows, chip_windows, chips_per_scene, make_spec
from hrdps_synthetic.geometry import incidence_field, scene_for


def test_axis_starts():
    assert axis_starts(2000, 500, 500) == [0, 500, 1000, 1500]
    assert axis_starts(3100, 500, 500, 'drop') == [0, 500, 1000, 1500, 2000, 2500]
    assert axis_starts(3100, 500, 500, 'shift') == [0, 500, 1000, 1500, 2000, 2500, 2600]
    assert axis_starts(1000, 500, 250) == [0, 250, 500]
    with pytest.raises(ValueError):
        axis_starts(400, 500, 500)


def test_default_scene_gives_24_detectron_chips():
    spec = make_spec([500])
    assert chips_per_scene(3000, 2000, spec) == 24
    assert spec.dirname == 'chips_500x500'
    assert make_spec([512], [256], 'shift').dirname == 'chips_512x512_s256x256_shift'
    assert make_spec([400, 300]).height == 300
    with pytest.raises(ValueError):
        make_spec([500, 500, 500])


def test_chip_metadata_matches_scene_geometry(all_aois):
    a = all_aois[0]
    scene = scene_for(a, 0, 0)
    spec = make_spec([500])
    theta = incidence_field(scene, 35.0, 348.0)
    meta = dict(sample_id='s', split='train', aoi=a.name, direction='asc', ref_date='a', sec_date='b',
                theta_c_deg=35.0, heading_deg=348.0, wavelength_m=0.055466)
    rows = chip_rows(scene, np.zeros(scene.shape, np.float32), meta, spec)
    assert len(rows) == 24
    for row, (r0, c0) in rows:
        block = theta[r0:r0 + 500, c0:c0 + 500]
        assert abs(row['theta_center_deg'] - float(block[249:251, 249:251].mean())) < 2e-3
        assert row['theta_min_deg'] <= block.min() + 1e-3 and row['theta_max_deg'] >= block.max() - 1e-3
        assert row['x0'] == scene.x0 + c0 * 10 and row['y0'] == scene.y0 - r0 * 10


def _run_generate(wd, aois, *extra):
    return generate.main(['--workdir', str(wd), '--aois', *[a.name for a in aois], '--n-train', '4', '--n-val', '1',
                          '--workers', '2', '--width-px', '200', '--height-px', '300', *extra])


def test_generate_scenes_and_chips_then_tile_matches(fake_workdir):
    wd, aois = fake_workdir
    assert _run_generate(wd, aois, '--chip-size', '100') == 0
    with open(wd / 'manifest.csv') as f:
        scenes = list(csv.DictReader(f))
    assert len(scenes) == 5 and all(int(r['n_chips']) == 6 for r in scenes)
    chip_csv = wd / 'chips_100x100' / 'chips.csv'
    with open(chip_csv) as f:
        chips = {r['chip_id']: r for r in csv.DictReader(f)}
    assert len(chips) == 30
    for r in chips.values():  # chips inherit the scene split and georeferencing
        with rasterio.open(wd / r['unw_path']) as u, rasterio.open(wd / r['wrp_path']) as w:
            assert u.shape == (100, 100) and u.transform.c == float(r['x0']) and u.transform.f == float(r['y0'])
            k = (u.read(1).astype(float) - w.read(1)) / (2 * np.pi)
            assert np.allclose(k, np.round(k), atol=1e-3)
        assert r['chip_id'].startswith(r['split'])

    # The tile command reproduces the same chips from the written scenes.
    first = {cid: rasterio.open(wd / r['unw_path']).read(1) for cid, r in chips.items()}
    shutil.rmtree(wd / 'chips_100x100')
    assert tile.main(['--workdir', str(wd), '--chip-size', '100', '--workers', '2']) == 0
    with open(chip_csv) as f:
        again = {r['chip_id']: r for r in csv.DictReader(f)}
    assert again.keys() == chips.keys()
    for cid, r in again.items():
        assert r == chips[cid]
        assert np.array_equal(rasterio.open(wd / r['unw_path']).read(1), first[cid])
    assert tile.main(['--workdir', str(wd), '--chip-size', '100']) == 0  # resumable: nothing left to do


def test_chips_only_and_output_guard(fake_workdir):
    wd, aois = fake_workdir
    assert _run_generate(wd, aois, '--chip-size', '100', '150', '--no-scenes', '--limit', '2') == 0
    assert not (wd / 'samples').exists()
    with open(wd / 'chips_100x150' / 'chips.csv') as f:
        assert len(list(csv.DictReader(f))) == 2 * 2 * 2  # 2 x 2 chips per 200 x 300 scene
    # Changing outputs half-way through a plan is refused.
    assert _run_generate(wd, aois, '--chip-size', '50') == 2
    assert _run_generate(wd, aois, '--chip-size', '100', '150', '--no-scenes') == 0


def test_scene_size_beyond_epoch_crop_is_refused(fake_workdir):
    wd, aois = fake_workdir
    rc = generate.main(['--workdir', str(wd), '--aois', *[a.name for a in aois], '--n-train', '4', '--n-val', '1',
                        '--width-km', '80', '--length-km', '80', '--dry-run', '--no-probe'])
    assert rc == 0  # dry run reports it
    rc = generate.main(['--workdir', str(wd), '--aois', *[a.name for a in aois], '--n-train', '4', '--n-val', '1',
                        '--width-km', '80', '--length-km', '80'])
    assert rc == 2


def test_quicklook_outlines_chips(fake_workdir):
    wd, aois = fake_workdir
    assert _run_generate(wd, aois, '--chip-size', '100', '--limit', '1', '--quicklook', '1') == 0
    pngs = list((wd / 'quicklook').glob('*.png'))
    assert len(pngs) == 1 and pngs[0].stat().st_size > 10_000
