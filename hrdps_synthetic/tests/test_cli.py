"""End-to-end checks of the generate command on synthetic epochs."""

import csv
import os

import numpy as np
import pytest
import rasterio

from hrdps_synthetic import generate


def _snapshot(root):
    return sorted((p, os.path.getsize(os.path.join(p))) for p in (str(x) for x in root.rglob('*')))


def test_generate_dry_run_writes_nothing(fake_workdir, capsys):
    wd, aois = fake_workdir
    before = _snapshot(wd)
    rc = generate.main(['--workdir', str(wd), '--aois', *[a.name for a in aois], '--n-train', '20', '--n-val', '2',
                        '--dry-run', '--no-probe'])
    assert rc == 0
    assert _snapshot(wd) == before
    out = capsys.readouterr().out
    assert 'Uncompressed output' in out and 'nothing was written' in out


@pytest.mark.skipif(not os.environ.get('HRDPS_NETWORK_TESTS'), reason='needs GLO-30 network access')
def test_generate_small_run(fake_workdir):
    wd, aois = fake_workdir
    rc = generate.main(['--workdir', str(wd), '--aois', *[a.name for a in aois], '--n-train', '4', '--n-val', '1',
                        '--workers', '2', '--width-km', '2', '--length-km', '3'])
    assert rc == 0
    with open(wd / 'manifest.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    for r in rows:
        with rasterio.open(wd / r['unw_path']) as u, rasterio.open(wd / r['wrp_path']) as w:
            assert u.shape == (300, 200) and u.res == (10.0, 10.0)
            unw, wrp = u.read(1), w.read(1)
        assert np.isfinite(unw).all()
        k = (unw.astype(float) - wrp) / (2 * np.pi)
        assert np.allclose(k, np.round(k), atol=1e-3)
