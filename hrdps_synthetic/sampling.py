"""Train/validation split and sample plan.

Rules (agreed with the user):

* Each (AOI, direction, date pair) is rendered at most once. Re-rendering the
  same atmosphere over the same ground with a different incidence would give a
  near-duplicate.
* Validation holds out **both** AOIs and dates: training samples use training
  AOIs on training dates only, validation samples use validation AOIs on
  validation dates only. AOIs whose scenes can overlap are kept in one split.
* The number of validation dates is chosen to minimise the AOIs needed for the
  requested train/validation counts.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from hrdps_synthetic.aois import AOI, RCM_HEADINGS, SCENE_LENGTH_M, SCENE_WIDTH_M, overlap_groups

DIRECTIONS = ('asc', 'desc')
SPLITS = ('train', 'val')


class ShortfallError(RuntimeError):
    """The unique-sample pool is smaller than requested."""


@dataclass(frozen=True)
class SplitDesign:
    n_dates: int
    val_dates: int
    train_dates: int
    train_aois: int  # minimum needed
    val_aois: int  # minimum needed

    def pairs_per_cell(self, split: str) -> int:
        return math.comb(self.train_dates if split == 'train' else self.val_dates, 2)


def min_aoi_design(n_dates: int, n_train: int, n_val: int, n_directions: int = 2) -> SplitDesign:
    """Validation-date count that minimises the AOIs needed (ties: more training dates)."""
    best = None
    for v in range(2, n_dates - 1):
        t = n_dates - v
        ptr, pva = n_directions * math.comb(t, 2), n_directions * math.comb(v, 2)
        ntr = math.ceil(n_train / ptr) if n_train else 0
        nva = math.ceil(n_val / pva) if n_val else 0
        key = (ntr + nva, -t)
        if best is None or key < best[0]:
            best = (key, SplitDesign(n_dates, v, t, ntr, nva))
    if best is None:
        raise ShortfallError(f'Need at least 4 usable dates to split, have {n_dates}')
    return best[1]


def choose_val_dates(dates: Sequence[dt.date], n_val: int, seed: int) -> set:
    """Evenly spaced validation dates with a random phase, so they span the whole window."""
    dates = sorted(dates)
    if n_val <= 0:
        return set()
    u = np.random.default_rng([seed, 1]).uniform()
    idx = sorted({min(int((k + u) * len(dates) / n_val), len(dates) - 1) for k in range(n_val)})
    return {dates[i] for i in idx}


def assign_aoi_splits(aois: Sequence[AOI], design: SplitDesign, seed: int, n_train: int, n_val: int,
                      n_directions: int = 2, width: float = SCENE_WIDTH_M, length: float = SCENE_LENGTH_M,
                      geometry: str = 'north-up', headings: Sequence[float] = RCM_HEADINGS) -> dict[str, str]:
    """Assign AOIs to train/val: overlap groups kept together, stratified by region."""
    need = design.train_aois + design.val_aois
    if len(aois) < need:
        raise ShortfallError(f'{len(aois)} AOIs available but {need} needed '
                             f'({design.train_aois} train + {design.val_aois} val at {design.n_dates} usable days)')
    per_tr = n_directions * design.pairs_per_cell('train')
    per_va = n_directions * design.pairs_per_cell('val')
    n_tr_aois, n_va_aois = design.train_aois, design.val_aois
    for _ in range(len(aois) - need):  # give each spare AOI to the split with less slack
        slack_tr = n_tr_aois * per_tr / max(n_train, 1)
        slack_va = n_va_aois * per_va / max(n_val, 1)
        if slack_va < slack_tr:
            n_va_aois += 1
        else:
            n_tr_aois += 1

    region = {a.name: a.region for a in aois}
    units = overlap_groups(aois, width, length, geometry, headings)
    rng = np.random.default_rng([seed, 2])
    by_region: dict[str, list[list[str]]] = {}
    for u in units:
        regs = [region[n] for n in u]
        by_region.setdefault(max(set(regs), key=regs.count), []).append(u)
    order = sorted(by_region, key=lambda r: (-len(by_region[r]), r))
    for r in order:
        rng.shuffle(by_region[r])

    val: list[str] = []
    while len(val) < n_va_aois and any(by_region.values()):
        progressed = False
        for r in order:
            if len(val) >= n_va_aois:
                break
            queue = by_region[r]
            for k, u in enumerate(queue):
                if len(val) + len(u) <= n_va_aois:
                    val += queue.pop(k)
                    progressed = True
                    break
        if not progressed:  # only groups larger than the remaining gap: overshoot by one group
            for r in order:
                if by_region[r]:
                    val += by_region[r].pop(0)
                    break
    return {a.name: ('val' if a.name in val else 'train') for a in aois}


def planned_aoi_splits(aois: Sequence[AOI], n_days: int, seed: int = 0,
                       n_train: int = 8500, n_val: int = 1500) -> tuple[dict, SplitDesign]:
    design = min_aoi_design(n_days, n_train, n_val)
    return assign_aoi_splits(aois, design, seed, n_train, n_val), design


def waterfill(total: int, capacities: Sequence[int]) -> list[int]:
    """Split `total` as evenly as possible across cells without exceeding capacities."""
    caps = np.asarray(capacities, dtype=int)
    if total > caps.sum():
        raise ShortfallError(f'requested {total} but only {int(caps.sum())} unique pairs available')
    quota = np.zeros_like(caps)
    remaining = total
    open_cells = list(np.flatnonzero(caps > 0))
    while remaining > 0 and open_cells:
        share = max(remaining // len(open_cells), 1)
        still_open = []
        for i in open_cells:
            if remaining == 0:
                break
            give = min(share, caps[i] - quota[i], remaining)
            quota[i] += give
            remaining -= give
            if quota[i] < caps[i]:
                still_open.append(i)
        open_cells = still_open
    return quota.tolist()


# ---------------------------------------------------------------------------
# Sample plan
# ---------------------------------------------------------------------------
PLAN_FIELDS = ('sample_id', 'split', 'aoi', 'region', 'direction', 'heading_deg', 'ref_date', 'sec_date',
               'ref_epoch', 'sec_epoch', 'theta_c_deg', 'offset_x_m', 'offset_y_m')


@dataclass
class PlanRow:
    sample_id: str
    split: str
    aoi: str
    region: str
    direction: str
    heading_deg: float
    ref_date: str  # YYYY-MM-DD
    sec_date: str
    ref_epoch: str  # path relative to the workdir
    sec_epoch: str
    theta_c_deg: float
    offset_x_m: float
    offset_y_m: float


@dataclass
class PlanSummary:
    design: SplitDesign
    usable_dates: dict
    val_dates: list
    aoi_splits: dict
    pools: dict  # split -> unique pairs available
    requested: dict
    groups: list = field(default_factory=list)


def usable_dates(epoch_index: dict, aois: Sequence[AOI]) -> dict[str, list]:
    """Dates with an epoch for every AOI, per direction."""
    out = {}
    for d in DIRECTIONS:
        sets = [set(epoch_index.get((a.name, d), {})) for a in aois]
        out[d] = sorted(set.intersection(*sets)) if sets else []
    return out


def summarise(aois: Sequence[AOI], epoch_index: dict, n_train: int, n_val: int, seed: int,
              width: float = SCENE_WIDTH_M, length: float = SCENE_LENGTH_M, geometry: str = 'north-up',
              headings: Sequence[float] = RCM_HEADINGS) -> PlanSummary:
    usable = usable_dates(epoch_index, aois)
    common = sorted(set(usable['asc']) & set(usable['desc']))
    design = min_aoi_design(len(common), n_train, n_val)
    val_dates = choose_val_dates(common, design.val_dates, seed)
    splits = assign_aoi_splits(aois, design, seed, n_train, n_val, width=width, length=length,
                               geometry=geometry, headings=headings)
    pools = {'train': 0, 'val': 0}
    for a in aois:
        s = splits[a.name]
        for d in DIRECTIONS:
            n = sum(1 for x in usable[d] if (x in val_dates) == (s == 'val'))
            pools[s] += math.comb(n, 2)
    return PlanSummary(design, usable, sorted(val_dates), splits, pools, {'train': n_train, 'val': n_val},
                       [g for g in overlap_groups(aois, width, length, geometry, headings) if len(g) > 1])


def build_plan(aois: Sequence[AOI], epoch_index: dict, n_train: int, n_val: int, seed: int,
               inc_range: tuple[float, float], headings: dict[str, float], width: float = SCENE_WIDTH_M,
               length: float = SCENE_LENGTH_M, geometry: str = 'north-up') -> tuple[list[PlanRow], PlanSummary]:
    """Deterministic plan for scenes of `width` x `length` m. `epoch_index[(aoi, direction)] = {date: path}`.

    Offsets are in the scene's own axes: east/north for north-up scenes,
    across-track/along-track for radar scenes.
    """
    hd = (headings['asc'], headings['desc'])
    summary = summarise(aois, epoch_index, n_train, n_val, seed, width, length, geometry, hd)
    val_dates = set(summary.val_dates)
    rng = np.random.default_rng([seed, 3])
    rows: list[PlanRow] = []
    for split, n_req in (('train', n_train), ('val', n_val)):
        cells = []
        for a in aois:
            if summary.aoi_splits[a.name] != split:
                continue
            for d in DIRECTIONS:
                dates = [x for x in summary.usable_dates[d] if (x in val_dates) == (split == 'val')]
                cells.append((a, d, list(combinations(sorted(dates), 2))))
        quotas = waterfill(n_req, [len(c[2]) for c in cells]) if n_req else [0] * len(cells)
        for (a, d, pairs), q in zip(cells, quotas):
            if q == 0:
                continue
            dxr, dyr = a.offset_ranges(width, length, geometry, headings[d] if geometry == 'radar' else None)
            for k in sorted(rng.choice(len(pairs), size=q, replace=False)):
                ref, sec = pairs[k]
                rows.append(PlanRow(
                    sample_id=f'{split}_{a.name}_{d}_{ref:%Y%m%d}_{sec:%Y%m%d}',
                    split=split, aoi=a.name, region=a.region, direction=d, heading_deg=headings[d],
                    ref_date=ref.isoformat(), sec_date=sec.isoformat(),
                    ref_epoch=str(epoch_index[(a.name, d)][ref]), sec_epoch=str(epoch_index[(a.name, d)][sec]),
                    theta_c_deg=round(float(rng.uniform(*inc_range)), 4),
                    offset_x_m=round(float(rng.uniform(*dxr)), 1),
                    offset_y_m=round(float(rng.uniform(*dyr)), 1),
                ))
    ids = [r.sample_id for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate (AOI, direction, date pair) in plan'
    return rows, summary


def write_plan(rows: Sequence[PlanRow], path: Path) -> None:
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    tmp.replace(path)


def read_plan(path: Path) -> list[PlanRow]:
    with open(path, newline='') as f:
        out = []
        for r in csv.DictReader(f):
            for k in ('heading_deg', 'theta_c_deg', 'offset_x_m', 'offset_y_m'):
                r[k] = float(r[k])
            out.append(PlanRow(**r))
    return out


def format_summary(s: PlanSummary, aois: Sequence[AOI]) -> str:
    d = s.design
    n_aoi = {sp: sum(1 for v in s.aoi_splits.values() if v == sp) for sp in SPLITS}
    lines = [
        f'Usable dates: asc {len(s.usable_dates["asc"])}, desc {len(s.usable_dates["desc"])}, '
        f'both directions {d.n_dates}',
        f'Validation dates ({d.val_dates}): {", ".join(x.isoformat() for x in s.val_dates)}',
        f'Minimum AOIs for this design: {d.train_aois} train + {d.val_aois} val = {d.train_aois + d.val_aois}; '
        f'available {len(aois)}',
        f'{"split":6s} {"AOIs":>5s} {"dates":>6s} {"pairs/cell":>11s} {"pool":>7s} {"requested":>10s}',
    ]
    for sp in SPLITS:
        lines.append(f'{sp:6s} {n_aoi[sp]:5d} {(d.train_dates if sp == "train" else d.val_dates):6d} '
                     f'{d.pairs_per_cell(sp):11d} {s.pools[sp]:7d} {s.requested[sp]:10d}'
                     + ('   SHORT' if s.pools[sp] < s.requested[sp] else ''))
    lines.append(f'Overlap groups kept in one split: {s.groups}')
    return '\n'.join(lines)
