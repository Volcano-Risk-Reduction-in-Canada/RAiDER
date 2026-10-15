"""Cut scenes into fixed-size chips (e.g. 500 x 500 px for Detectron2).

Chips are GeoTIFFs like the scenes: unwrapped and wrapped phase (float32,
radians) plus the false-colour RGB of the wrapped phase (3 x uint8). Each chip
has its own location, incidence range and LOS vector, so it can be summed with
magmatic-source chips before training. Radar-geometry chips carry no CRS, like
the magmatic files; their real corners are in chips.csv and the tags.

All chips of a scene inherit the scene's train/validation split. Chips from one
scene share an atmosphere but cover different ground. The default stride equals
the chip size, so chips do not overlap; with ``edge='drop'`` (default) a
remainder strip narrower than a chip is left out, and with ``edge='shift'`` a
last chip is added flush with the scene edge, overlapping its neighbour.

Directory layout (one folder per chip configuration):

    <workdir>/chips_<W>x<H>[_s<SX>x<SY>][_shift]/
        chips.csv
        <split>/<AOI>/<sample_id>_r<RR>_c<CC>_unw.tif
        <split>/<AOI>/<sample_id>_r<RR>_c<CC>_wrp.tif
        <split>/<AOI>/<sample_id>_r<RR>_c<CC>_rgb.tif
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from affine import Affine

from hrdps_synthetic.aois import to_lonlat
from hrdps_synthetic.geometry import RCM_ALTITUDE_M, R_EARTH_M, SceneGrid, incidence_at, los_ground_to_satellite
from hrdps_synthetic.tables import CsvAppender, read_rows

CHIP_FIELDS = (
    'chip_id', 'sample_id', 'split', 'aoi', 'direction', 'ref_date', 'sec_date', 'heading_deg',
    'chip_row', 'chip_col', 'row0', 'col0', 'height', 'width', 'geometry', 'epsg', 'x0', 'y0', 'spacing_m',
    'center_x', 'center_y', 'center_lat', 'center_lon', 'corners_lonlat', 'theta_center_deg', 'theta_min_deg',
    'theta_max_deg', 'los_east', 'los_north', 'los_up', 'wavelength_m', 'phase_sign', 'dstd_mean_m', 'dstd_std_m',
    'unw_path', 'wrp_path', 'rgb_path',
)
EDGES = ('drop', 'shift')


@dataclass(frozen=True)
class ChipSpec:
    width: int
    height: int
    stride_x: int
    stride_y: int
    edge: str = 'drop'

    @property
    def dirname(self) -> str:
        name = f'chips_{self.width}x{self.height}'
        if (self.stride_x, self.stride_y) != (self.width, self.height):
            name += f'_s{self.stride_x}x{self.stride_y}'
        return name + ('_shift' if self.edge == 'shift' else '')

    def describe(self) -> str:
        return (f'{self.width} x {self.height} px chips, stride {self.stride_x} x {self.stride_y}, '
                f'edge {self.edge}')


def make_spec(size: Optional[Sequence[int]], stride: Optional[Sequence[int]] = None, edge: str = 'drop') -> Optional[ChipSpec]:
    """ChipSpec from CLI values: size W [H], stride SX [SY] (default = size)."""
    if not size:
        return None
    if len(size) > 2 or (stride and len(stride) > 2):
        raise ValueError('chip size and stride take one value (square) or two values (W H)')
    w = int(size[0])
    h = int(size[1]) if len(size) > 1 else w
    if stride:
        sx = int(stride[0])
        sy = int(stride[1]) if len(stride) > 1 else sx
    else:
        sx, sy = w, h
    if min(w, h, sx, sy) <= 0:
        raise ValueError('chip size and stride must be positive')
    if edge not in EDGES:
        raise ValueError(f'edge must be one of {EDGES}')
    return ChipSpec(w, h, sx, sy, edge)


def axis_starts(n: int, size: int, stride: int, edge: str = 'drop') -> list[int]:
    """Start offsets of chips of `size` along an axis of `n` pixels."""
    if size > n:
        raise ValueError(f'chip size {size} exceeds the scene dimension {n}')
    starts = list(range(0, n - size + 1, stride))
    if edge == 'shift' and starts[-1] + size < n:
        starts.append(n - size)
    return starts


def chip_windows(ny: int, nx: int, spec: ChipSpec) -> list[tuple[int, int, int, int]]:
    """(chip_row, chip_col, row0, col0) for every chip, row-major from the north-west corner."""
    rows = axis_starts(ny, spec.height, spec.stride_y, spec.edge)
    cols = axis_starts(nx, spec.width, spec.stride_x, spec.edge)
    return [(i, j, r0, c0) for i, r0 in enumerate(rows) for j, c0 in enumerate(cols)]


def chips_per_scene(ny: int, nx: int, spec: ChipSpec) -> int:
    return len(chip_windows(ny, nx, spec))


def chip_dir(workdir: Path, spec: ChipSpec) -> Path:
    return Path(workdir) / spec.dirname


def chip_transform(scene: SceneGrid, row0: int, col0: int):
    return scene.transform * Affine.translation(col0, row0)


def chip_profile(scene: SceneGrid, row0: int, col0: int, spec: ChipSpec) -> dict:
    return dict(driver='GTiff', dtype='float32', width=spec.width, height=spec.height, count=1,
                crs=scene.crs, transform=chip_transform(scene, row0, col0), compress='deflate', predictor=3)


def chip_rows(scene: SceneGrid, unw: np.ndarray, meta: dict, spec: ChipSpec,
              altitude_m: float = RCM_ALTITUDE_M) -> list[tuple[dict, tuple]]:
    """Per-chip metadata and (row0, col0) windows; no I/O."""
    theta_c, heading = float(meta['theta_c_deg']), float(meta['heading_deg'])
    wavelength = float(meta['wavelength_m'])
    to_m = wavelength / (4.0 * math.pi)
    out = []
    for i, j, r0, c0 in chip_windows(scene.ny, scene.nx, spec):
        h, w = spec.height, spec.width
        kx, ky = scene.edge_xy([r0, r0, r0 + h, r0 + h], [c0, c0 + w, c0 + w, c0])  # corners, pixel order
        cx, cy = (float(v) for v in scene.edge_xy(r0 + h / 2, c0 + w / 2))
        # Incidence varies monotonically across track, so its extremes are at chip corners.
        th = incidence_at(scene, theta_c, heading, np.r_[cx, kx], np.r_[cy, ky], R_EARTH_M, altitude_m)
        lon, lat = to_lonlat(scene.epsg, cx, cy)
        klon, klat = to_lonlat(scene.epsg, kx, ky)
        e, n, u = los_ground_to_satellite(float(th[0]), heading)
        block = unw[r0:r0 + spec.height, c0:c0 + spec.width].astype(np.float64) * to_m
        row = dict(
            chip_id=f'{meta["sample_id"]}_r{i:02d}_c{j:02d}', sample_id=meta['sample_id'], split=meta['split'],
            aoi=meta['aoi'], direction=meta['direction'], ref_date=meta['ref_date'], sec_date=meta['sec_date'],
            heading_deg=heading, chip_row=i, chip_col=j, row0=r0, col0=c0, height=spec.height, width=spec.width,
            geometry=scene.geometry, epsg=scene.epsg, x0=float(kx[0]), y0=float(ky[0]), spacing_m=scene.spacing,
            center_x=cx, center_y=cy, center_lat=round(float(lat), 6), center_lon=round(float(lon), 6),
            corners_lonlat='; '.join(f'{a:.6f} {b:.6f}' for a, b in zip(klon, klat)),
            theta_center_deg=round(float(th[0]), 4), theta_min_deg=round(float(th.min()), 4),
            theta_max_deg=round(float(th.max()), 4), los_east=round(e, 6), los_north=round(n, 6), los_up=round(u, 6),
            wavelength_m=wavelength, phase_sign=meta.get('phase_sign', ''),
            dstd_mean_m=float(block.mean()), dstd_std_m=float(block.std()),
        )
        out.append((row, (r0, c0)))
    return out


def _chip_tags(row: dict, kind: str, convention: str) -> dict:
    keep = ('chip_id', 'sample_id', 'split', 'aoi', 'direction', 'heading_deg', 'ref_date', 'sec_date', 'geometry',
            'epsg', 'center_x', 'center_y', 'corners_lonlat', 'theta_center_deg', 'theta_min_deg', 'theta_max_deg',
            'los_east', 'los_north', 'los_up', 'wavelength_m', 'phase_sign')
    tags = {k: str(row[k]) for k in keep if row.get(k, '') != ''}
    tags.update(units='radians', quantity=f'{kind} interferometric phase, atmosphere only (HRDPS)',
                convention=convention, los_convention='unit vector ground->satellite in ENU, at chip centre')
    return tags


def write_chips(workdir: Path, spec: ChipSpec, scene: SceneGrid, unw: np.ndarray, wrp: np.ndarray, meta: dict,
                altitude_m: float = RCM_ALTITUDE_M, rgb: Optional[np.ndarray] = None, rgb_tags: Optional[dict] = None,
                convention: str = '') -> list[dict]:
    """Write every chip of one scene (unwrapped, wrapped, and RGB if given); return chip manifest rows.

    `rgb` is the scene's (3, H, W) uint8 RGB; `convention` is the scene's phase-convention text.
    """
    import rasterio

    from hrdps_synthetic.rgb import write_rgb

    workdir = Path(workdir)
    d = chip_dir(workdir, spec) / meta['split'] / meta['aoi']
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for row, (r0, c0) in chip_rows(scene, unw, meta, spec, altitude_m):
        win = (slice(r0, r0 + spec.height), slice(c0, c0 + spec.width))
        for arr, kind, key in ((unw, 'unwrapped', 'unw_path'), (wrp, 'wrapped', 'wrp_path')):
            path = d / f'{row["chip_id"]}_{key[:3]}.tif'
            tmp = path.with_name(path.name + '.part')
            with rasterio.open(tmp, 'w', **chip_profile(scene, r0, c0, spec)) as dst:
                dst.write(np.ascontiguousarray(arr[win]), 1)
                dst.update_tags(**_chip_tags(row, kind, convention))
                dst.set_band_description(1, f'{kind} phase (rad)')
            tmp.replace(path)
            row[key] = str(path.relative_to(workdir))
        row['rgb_path'] = ''
        if rgb is not None:
            path = d / f'{row["chip_id"]}_rgb.tif'
            tags = _chip_tags(row, 'wrapped', convention)
            tags.update(rgb_tags or {})
            write_rgb(path, np.ascontiguousarray(rgb[(slice(None),) + win]), scene.crs,
                      chip_transform(scene, r0, c0), tags, tiled=False)
            row['rgb_path'] = str(path.relative_to(workdir))
        rows.append(row)
    return rows


def chip_nbytes_in_memory(spec: ChipSpec, scene: SceneGrid, unw: np.ndarray, wrp: np.ndarray, meta: dict,
                          rgb: Optional[np.ndarray] = None) -> int:
    """Compressed bytes of all chips of one scene, measured without writing files."""
    from rasterio.io import MemoryFile

    from hrdps_synthetic.rgb import rgb_profile

    total = 0
    for row, (r0, c0) in chip_rows(scene, unw, meta, spec):
        win = (slice(r0, r0 + spec.height), slice(c0, c0 + spec.width))
        for arr, kind in ((unw, 'unwrapped'), (wrp, 'wrapped')):
            with MemoryFile() as mf:
                with mf.open(**chip_profile(scene, r0, c0, spec)) as dst:
                    dst.write(np.ascontiguousarray(arr[win]), 1)
                    dst.update_tags(**_chip_tags(row, kind, ''))
                total += len(mf.read())
        if rgb is not None:
            with MemoryFile() as mf:
                with mf.open(**rgb_profile(spec.width, spec.height, scene.crs, chip_transform(scene, r0, c0),
                                           tiled=False)) as dst:
                    dst.write(np.ascontiguousarray(rgb[(slice(None),) + win]))
                total += len(mf.read())
    return total


def read_chipped(path: Path) -> tuple[set, set]:
    """(sample ids, chip ids) already recorded in a chips.csv."""
    _, rows = read_rows(path)
    return {r['sample_id'] for r in rows}, {r['chip_id'] for r in rows}


class ChipManifest:
    """Append-only chips.csv writer (main process only)."""

    def __init__(self, workdir: Path, spec: ChipSpec):
        self.path = chip_dir(workdir, spec) / 'chips.csv'
        self.done_samples, self.done_chips = read_chipped(self.path)
        self._out = CsvAppender(self.path, CHIP_FIELDS)

    def write(self, rows: Sequence[dict]) -> None:
        for r in rows:
            if r['chip_id'] in self.done_chips:  # re-rendered after an interruption
                continue
            self._out.write(r)
            self.done_chips.add(r['chip_id'])
            self.done_samples.add(r['sample_id'])

    def close(self) -> None:
        self._out.close()
