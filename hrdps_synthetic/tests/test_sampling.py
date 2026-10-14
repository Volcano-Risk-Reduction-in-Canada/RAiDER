import math
from collections import Counter

import pytest

from hrdps_synthetic import hrdps_archive as ha
from hrdps_synthetic.sampling import (ShortfallError, assign_aoi_splits, build_plan, min_aoi_design,
                                      planned_aoi_splits, waterfill)

HEADINGS = {'asc': 348.0, 'desc': 192.0}


def test_design_matches_plan_table():
    d = min_aoi_design(29, 8500, 1500)
    assert (d.val_dates, d.train_dates, d.train_aois, d.val_aois) == (10, 19, 25, 17)
    assert min_aoi_design(28, 8500, 1500).train_aois + min_aoi_design(28, 8500, 1500).val_aois == 45


def test_full_split_respects_groups(all_aois):
    splits, design = planned_aoi_splits(all_aois, 29)
    n_val = sum(v == 'val' for v in splits.values())
    assert n_val >= design.val_aois and len(splits) - n_val >= design.train_aois
    assert splits['A4202_Volcano_Garbld'] == splits['A4203_Volcano_Cayley']
    assert splits['A4204_Volcano_Edz_N'] == splits['A4205_Volcano_Edz_S']
    regions_val = {a.region for a in all_aois if splits[a.name] == 'val'}
    assert len(regions_val) >= 5  # validation spans BC


def test_waterfill():
    q = waterfill(10, [3, 100, 100])
    assert sum(q) == 10 and q[0] <= 3 and max(q) - min(q[1:]) <= 1
    with pytest.raises(ShortfallError):
        waterfill(10, [3, 3])


def test_plan_unique_and_disjoint(fake_workdir):
    wd, aois = fake_workdir
    idx = ha.index_epochs(wd)
    rows, summary = build_plan(aois, idx, 20, 2, 7, (20, 55), HEADINGS)
    keys = [(r.aoi, r.direction, r.ref_date, r.sec_date) for r in rows]
    assert len(keys) == len(set(keys)) == 22
    val_dates = {d.isoformat() for d in summary.val_dates}
    for r in rows:
        assert r.ref_date < r.sec_date
        assert summary.aoi_splits[r.aoi] == r.split
        assert ((r.ref_date in val_dates) and (r.sec_date in val_dates)) == (r.split == 'val')
        assert (r.ref_date in val_dates) == (r.split == 'val')
        assert 20 <= r.theta_c_deg <= 55
    train_aois = {r.aoi for r in rows if r.split == 'train'}
    val_aois = {r.aoi for r in rows if r.split == 'val'}
    assert not train_aois & val_aois
    rows2, _ = build_plan(aois, idx, 20, 2, 7, (20, 55), HEADINGS)
    assert rows == rows2  # deterministic


def test_plan_shortfall(fake_workdir):
    wd, aois = fake_workdir
    with pytest.raises(ShortfallError):
        build_plan(aois, ha.index_epochs(wd), 500, 20, 0, (20, 55), HEADINGS)
