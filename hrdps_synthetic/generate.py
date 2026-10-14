"""Generate synthetic atmosphere-only interferograms from HRDPS epoch cubes.

Each sample is one (AOI, direction, date pair), rendered once:

* north-up UTM grid, 20 km E-W x 30 km N-S at 10 m (2000 x 3000 pixels) by
  default; set the size with --width-km/--length-km or --width-px/--height-px;
* scene centre = AOI centroid + random shift (volcano polygons stay inside);
* scene-centre incidence ~ U(20, 55) deg with the real cross-track ramp;
* dSTD = (ZTD_ref - ZTD_sec)(lat, lon, DEM height) / cos(incidence);
* unwrapped phase = 4*pi/lambda * dSTD (radians, no referencing), and its wrap.

Outputs (under --workdir):

    plan.csv, plan_meta.json            the deterministic sample plan
    samples/<split>/<AOI>/<id>_unw.tif  unwrapped phase, float32, radians
    samples/<split>/<AOI>/<id>_wrp.tif  wrapped phase, float32, radians in [-pi, pi)
    manifest.csv                        one row per finished sample
    chips_<W>x<H>/...                   optional chips (--chip-size), see chips.py
    quicklook/<id>.png                  optional (--quicklook N)

With --chip-size each scene is also cut into chips (e.g. 500 x 500 px for
Detectron2) as it is rendered; --no-scenes keeps only the chips. To chip scenes
that already exist, use ``python -m hrdps_synthetic.tile``.

Training samples use training AOIs on training dates only; validation samples
use validation AOIs on validation dates only.

Examples (from the repository root):

    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --dry-run
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --workers 16
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --limit 5 --quicklook 5
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --chip-size 500          # scenes + 500 px chips
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --width-px 500 --height-px 500   # 5 km scenes
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from hrdps_synthetic import hrdps_archive as ha
from hrdps_synthetic.chips import (EDGES, ChipManifest, ChipSpec, chip_nbytes_in_memory, chip_windows, chips_per_scene,
                                   make_spec, read_chipped, write_chips)
from hrdps_synthetic.aois import DEFAULT_API_DUMP, AOI, load_aois, select_aois, to_lonlat
from hrdps_synthetic.dem import ensure_aoi_dem, estimate_dem_bytes, maybe_dem, read_scene_dem, scene_dem_in_memory
from hrdps_synthetic.geometry import (RCM_ALTITUDE_M, R_EARTH_M, SceneGrid, coarse_lonlat, incidence_field,
                                      los_ground_to_satellite, look_azimuth_deg, scene_for, upsample)
from hrdps_synthetic.sampling import (PLAN_FIELDS, PlanRow, ShortfallError, build_plan, format_summary, read_plan,
                                      summarise, write_plan)
from hrdps_synthetic.slant import SLANT_METHODS, Epoch, delta_ztd, epoch_indices, interpolate_cube, project, to_phase

log = logging.getLogger('hrdps_synthetic')

RCM_WAVELENGTH_M = 0.055466  # 5.405 GHz

MANIFEST_FIELDS = PLAN_FIELDS + (
    'look_azimuth_deg', 'theta_min_deg', 'theta_max_deg', 'los_east', 'los_north', 'los_up',
    'center_lat', 'center_lon', 'center_x', 'center_y', 'epsg', 'x0', 'y0', 'spacing_m', 'nx', 'ny',
    'wavelength_m', 'dem_min_m', 'dem_max_m', 'dstd_mean_m', 'dstd_std_m', 'dstd_min_m', 'dstd_max_m',
    'altitude_m', 'n_chips', 'unw_path', 'wrp_path', 'seconds',
)


@dataclass
class GenConfig:
    workdir: str
    aoi_file: str
    aoi_names: Optional[list]
    width_m: float = 20_000.0
    length_m: float = 30_000.0
    spacing: float = 10.0
    wavelength: float = RCM_WAVELENGTH_M
    slant: str = 'projected'
    altitude_m: float = RCM_ALTITUDE_M
    write_scenes: bool = True
    chip: Optional[dict] = None  # asdict(ChipSpec), or None for no chips

    @property
    def chip_spec(self) -> Optional[ChipSpec]:
        return ChipSpec(**self.chip) if self.chip else None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(row: PlanRow, cfg: GenConfig, aoi: AOI, ref: Epoch, sec: Epoch, dem: np.ndarray):
    t0 = time.time()
    scene = scene_for(aoi, row.offset_x_m, row.offset_y_m, cfg.width_m, cfg.length_m, cfg.spacing)
    if dem.shape != scene.shape:
        raise ValueError(f'DEM shape {dem.shape} != scene {scene.shape}')
    lon_n, lat_n, nodes = coarse_lonlat(scene)
    xi_n, yi_n = epoch_indices(ref, lon_n, lat_n)
    xi = upsample(xi_n, nodes, scene.shape)
    yi = upsample(yi_n, nodes, scene.shape)
    dztd = interpolate_cube(delta_ztd(ref, sec), ref.z, xi, yi, dem)
    del xi, yi
    theta = incidence_field(scene, row.theta_c_deg, row.heading_deg, R_EARTH_M, cfg.altitude_m)
    dstd = project(dztd, theta, cfg.slant)
    unw, wrp = to_phase(dstd, cfg.wavelength)
    cx, cy = scene.center
    lon_c, lat_c = to_lonlat(scene.epsg, cx, cy)
    e, n, u = los_ground_to_satellite(row.theta_c_deg, row.heading_deg)
    meta = dict(
        asdict(row),
        look_azimuth_deg=look_azimuth_deg(row.heading_deg),
        theta_min_deg=round(float(theta.min()), 4), theta_max_deg=round(float(theta.max()), 4),
        los_east=round(e, 6), los_north=round(n, 6), los_up=round(u, 6),
        center_lat=round(float(lat_c), 6), center_lon=round(float(lon_c), 6), center_x=cx, center_y=cy,
        epsg=scene.epsg, x0=scene.x0, y0=scene.y0, spacing_m=scene.spacing, nx=scene.nx, ny=scene.ny,
        wavelength_m=cfg.wavelength, altitude_m=cfg.altitude_m,
        dem_min_m=round(float(dem.min()), 1), dem_max_m=round(float(dem.max()), 1),
        dstd_mean_m=float(dstd.mean()), dstd_std_m=float(dstd.std()),
        dstd_min_m=float(dstd.min()), dstd_max_m=float(dstd.max()),
        seconds=round(time.time() - t0, 2),
    )
    return scene, unw, wrp, meta


def _profile(scene: SceneGrid) -> dict:
    return dict(driver='GTiff', dtype='float32', width=scene.nx, height=scene.ny, count=1,
                crs=f'EPSG:{scene.epsg}', transform=scene.transform, compress='deflate', predictor=3,
                tiled=True, blockxsize=512, blockysize=512)


def _tags(meta: dict, kind: str) -> dict:
    keep = ('sample_id', 'split', 'aoi', 'direction', 'heading_deg', 'look_azimuth_deg', 'ref_date', 'sec_date',
            'theta_c_deg', 'theta_min_deg', 'theta_max_deg', 'los_east', 'los_north', 'los_up', 'wavelength_m')
    tags = {k: str(meta[k]) for k in keep}
    tags.update(units='radians', quantity=f'{kind} interferometric phase, atmosphere only (HRDPS)',
                convention='phase = 4*pi/wavelength * (STD_ref - STD_sec); STD = ZTD/cos(incidence); no referencing',
                los_convention='unit vector ground->satellite in ENU')
    return tags


def write_geotiffs(scene: SceneGrid, unw: np.ndarray, wrp: np.ndarray, meta: dict, unw_path: Path, wrp_path: Path):
    import rasterio

    unw_path.parent.mkdir(parents=True, exist_ok=True)
    for arr, path, kind in ((unw, unw_path, 'unwrapped'), (wrp, wrp_path, 'wrapped')):
        tmp = path.with_name(path.name + '.part')
        with rasterio.open(tmp, 'w', **_profile(scene)) as dst:
            dst.write(arr, 1)
            dst.update_tags(**_tags(meta, kind))
            dst.set_band_description(1, f'{kind} phase (rad)')
        tmp.replace(path)


def sample_paths(workdir: Path, row: PlanRow) -> tuple[Path, Path]:
    d = Path(workdir) / 'samples' / row.split / row.aoi
    return d / f'{row.sample_id}_unw.tif', d / f'{row.sample_id}_wrp.tif'


# Worker state (one copy per process)
_STATE: dict = {}


def _init_worker(cfg: dict) -> None:
    c = GenConfig(**cfg)
    aois = select_aois(load_aois(Path(c.aoi_file)), c.aoi_names)
    _STATE.clear()
    _STATE.update(cfg=c, aois={a.name: a for a in aois}, epochs=OrderedDict())


def _epoch(path: Path) -> Epoch:
    cache = _STATE['epochs']
    if path in cache:
        cache.move_to_end(path)
        return cache[path]
    ep = Epoch.load(path)
    cache[path] = ep
    while len(cache) > 8:
        cache.popitem(last=False)
    return ep


def _work(row: PlanRow) -> dict:
    cfg: GenConfig = _STATE['cfg']
    wd = Path(cfg.workdir)
    aoi = _STATE['aois'][row.aoi]
    scene = scene_for(aoi, row.offset_x_m, row.offset_y_m, cfg.width_m, cfg.length_m, cfg.spacing)
    dem = read_scene_dem(ensure_aoi_dem(aoi, wd, cfg.width_m, cfg.length_m, cfg.spacing), scene)
    scene, unw, wrp, meta = render(row, cfg, aoi, _epoch(wd / row.ref_epoch), _epoch(wd / row.sec_epoch), dem)
    meta.update(unw_path='', wrp_path='', n_chips=0, chips=[])
    if cfg.write_scenes:
        unw_path, wrp_path = sample_paths(wd, row)
        write_geotiffs(scene, unw, wrp, meta, unw_path, wrp_path)
        meta.update(unw_path=str(unw_path.relative_to(wd)), wrp_path=str(wrp_path.relative_to(wd)))
    spec = cfg.chip_spec
    if spec:
        meta['chips'] = write_chips(wd, spec, scene, unw, wrp, meta, cfg.altitude_m)
        meta['n_chips'] = len(meta['chips'])
    return meta


# ---------------------------------------------------------------------------
# Manifest, quicklooks
# ---------------------------------------------------------------------------
def read_manifest_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline='') as f:
        return {r['sample_id'] for r in csv.DictReader(f)}


def quicklook(workdir: Path, row: PlanRow, out_dir: Path, spec: Optional[ChipSpec] = None,
              decimate: int = 4, spacing: float = 10.0) -> Optional[Path]:
    """PNG of DEM, unwrapped and wrapped phase (needs the scene GeoTIFFs); chips outlined if `spec`."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio

    unw_path, wrp_path = sample_paths(workdir, row)
    if not unw_path.exists():
        return None
    with rasterio.open(unw_path) as ds:
        unw = ds.read(1)[::decimate, ::decimate]
        ext = (ds.bounds.left / 1e3, ds.bounds.right / 1e3, ds.bounds.bottom / 1e3, ds.bounds.top / 1e3)
    with rasterio.open(wrp_path) as ds:
        wrp = ds.read(1)[::decimate, ::decimate]
    aoi_dem = Path(workdir) / 'dem' / f'{row.aoi}_glo30_utm{spacing:g}m.tif'
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    if aoi_dem.exists():
        import rasterio.windows

        with rasterio.open(aoi_dem) as ds:
            win = rasterio.windows.from_bounds(ext[0] * 1e3, ext[2] * 1e3, ext[1] * 1e3, ext[3] * 1e3, ds.transform)
            dem = ds.read(1, window=win)[::decimate, ::decimate]
        im = axes[0].imshow(dem, extent=ext, cmap='terrain')
        fig.colorbar(im, ax=axes[0], shrink=0.7, label='height (m)')
        axes[0].set_title('DEM (GLO-30, 10 m)')
    lim = np.nanpercentile(np.abs(unw - np.nanmedian(unw)), 99)
    im = axes[1].imshow(unw, extent=ext, cmap='RdBu_r', vmin=np.nanmedian(unw) - lim, vmax=np.nanmedian(unw) + lim)
    fig.colorbar(im, ax=axes[1], shrink=0.7, label='rad')
    axes[1].set_title('unwrapped phase')
    im = axes[2].imshow(wrp, extent=ext, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im, ax=axes[2], shrink=0.7, label='rad')
    axes[2].set_title('wrapped phase')
    for a in axes:
        a.set_xlabel('UTM easting (km)')
    axes[0].set_ylabel('UTM northing (km)')
    fig.suptitle(f'{row.sample_id}  heading {row.heading_deg:.0f}  incidence {row.theta_c_deg:.1f} deg')
    if spec is not None:  # outline the chips on the phase panels
        from matplotlib.patches import Rectangle

        px_km = (ext[1] - ext[0]) / (unw.shape[1] * decimate)  # pixel size in km
        for _, _, r0, c0 in chip_windows(int(round((ext[3] - ext[2]) / px_km)), int(round((ext[1] - ext[0]) / px_km)), spec):
            for a in axes[1:]:
                a.add_patch(Rectangle((ext[0] + c0 * px_km, ext[3] - (r0 + spec.height) * px_km),
                                      spec.width * px_km, spec.height * px_km, fill=False, lw=0.5, ec='k'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{row.sample_id}.png'
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def probe(row: PlanRow, cfg: GenConfig, aoi: AOI) -> tuple[int, int, float]:
    """Render one sample in memory: (scene bytes, chip bytes, seconds) for both phase products."""
    from rasterio.io import MemoryFile

    wd = Path(cfg.workdir)
    t0 = time.time()
    scene = scene_for(aoi, row.offset_x_m, row.offset_y_m, cfg.width_m, cfg.length_m, cfg.spacing)
    existing = maybe_dem(wd, aoi, cfg.spacing)
    try:
        dem = read_scene_dem(existing, scene) if existing else scene_dem_in_memory(aoi, scene)
    except ValueError:  # existing DEM too small for this scene size
        dem = scene_dem_in_memory(aoi, scene)
    t_dem = time.time() - t0
    t1 = time.time()
    scene, unw, wrp, meta = render(row, cfg, aoi, Epoch.load(wd / row.ref_epoch), Epoch.load(wd / row.sec_epoch), dem)
    scene_bytes = 0
    if cfg.write_scenes:
        for arr, kind in ((unw, 'unwrapped'), (wrp, 'wrapped')):
            with MemoryFile() as mf:
                with mf.open(**_profile(scene)) as dst:
                    dst.write(arr, 1)
                    dst.update_tags(**_tags(meta, kind))
                scene_bytes += len(mf.read())
    spec = cfg.chip_spec
    chip_bytes = chip_nbytes_in_memory(spec, scene, unw, wrp, meta) if spec else 0
    secs = time.time() - t1
    log.info('Probe %s: scenes %.1f MB, chips %.1f MB, %.1f s render+compress (+%.1f s DEM), dSTD %.1f..%.1f mm',
             row.sample_id, scene_bytes / 1e6, chip_bytes / 1e6, secs, t_dem,
             1e3 * meta['dstd_min_m'], 1e3 * meta['dstd_max_m'])
    return scene_bytes, chip_bytes, secs


def check_epoch_coverage(aois: Sequence[AOI], epoch_index: dict, workdir: Path, width: float, length: float) -> list[str]:
    """AOIs whose epoch crops are too small for every possible scene of this size."""
    problems = []
    for a in aois:
        paths = [p for d in ('asc', 'desc') for p in epoch_index.get((a.name, d), {}).values()]
        if not paths:
            continue
        with xr.open_dataset(Path(workdir) / paths[0]) as ds:
            x0, x1, y0, y1 = float(ds.x.min()), float(ds.x.max()), float(ds.y.min()), float(ds.y.max())
        W, S, E, N = a.reach_lonlat(width=width, length=length).bounds
        if W < x0 or E > x1 or S < y0 or N > y1:
            problems.append(a.name)
    return problems


def dry_run(cfg: GenConfig, aois: Sequence[AOI], epoch_index: dict, args) -> int:
    wd = Path(cfg.workdir)
    n_req = args.n_train + args.n_val
    nx, ny = int(round(cfg.width_m / cfg.spacing)), int(round(cfg.length_m / cfg.spacing))
    print(f'Scene: {nx} x {ny} px at {cfg.spacing:g} m ({cfg.width_m / 1e3:g} km E-W x {cfg.length_m / 1e3:g} km N-S)')
    bad = check_epoch_coverage(aois, epoch_index, wd, cfg.width_m, cfg.length_m)
    if bad:
        print(f'Epoch crops too small for this scene size: {bad}. Rerun the download with matching '
              f'--width-km/--length-km into a new workdir.')
    try:
        summary = summarise(aois, epoch_index, args.n_train, args.n_val, args.seed, cfg.width_m, cfg.length_m)
        print(format_summary(summary, aois))
    except ShortfallError as e:
        print(f'Cannot build a plan from the epochs on disk: {e}')
        summary = None

    per_scene_raw = 2 * 4 * nx * ny
    print(f'\nSamples requested: {n_req} ({args.n_train} train + {args.n_val} val)')
    if cfg.write_scenes:
        print(f'Scenes, uncompressed: {per_scene_raw / 1e6:.0f} MB per sample, {n_req * per_scene_raw / 1e9:.1f} GB total')
    spec = cfg.chip_spec
    if spec:
        k = chips_per_scene(ny, nx, spec)
        cov = k * spec.width * spec.height / (nx * ny)
        print(f'Chips: {spec.describe()} -> {k} per scene ({100 * cov:.0f}% of scene pixels), {n_req * k} in total; '
              f'{n_req * k * 2 * 4 * spec.width * spec.height / 1e9:.1f} GB uncompressed')

    src = dst = 0
    missing = [a for a in aois if not maybe_dem(wd, a, cfg.spacing)]
    for a in missing:
        s_, d_ = estimate_dem_bytes(a, cfg.width_m, cfg.length_m, cfg.spacing)
        src, dst = src + s_, dst + d_
    print(f'DEMs to build: {len(missing)} of {len(aois)} AOIs; ~{src / 1e9:.1f} GB of GLO-30 reads, '
          f'~{dst / 1e9:.1f} GB of {cfg.spacing:g} m UTM DEMs (uncompressed; written compressed)')

    if summary is not None and not args.no_probe and not bad:
        rows = None
        try:
            rows, _ = build_plan(aois, epoch_index, args.n_train, args.n_val, args.seed, tuple(args.inc_range),
                                 {'asc': args.heading_asc, 'desc': args.heading_desc}, cfg.width_m, cfg.length_m)
        except ShortfallError as e:
            print(f'Plan would fail: {e}')
        if rows:
            aoi_map = {a.name: a for a in aois}
            sb, cb, secs = probe(rows[0], cfg, aoi_map[rows[0].aoi])
            print(f'Probe sample: scenes {sb / 1e6:.1f} MB, chips {cb / 1e6:.1f} MB compressed, '
                  f'{secs:.1f} s to render and compress')
            print(f'Estimated output: ~{n_req * (sb + cb) / 1e9:.1f} GB on disk; '
                  f'~{n_req * secs / 3600:.2f} CPU-hours (~{n_req * secs / 3600 / max(args.workers, 1):.2f} h '
                  f'with {args.workers} workers, plus DEM building)')
    elif args.no_probe:
        print('Probe skipped (--no-probe).')
    print('Dry run: nothing was written.')
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plan_meta(args, cfg: GenConfig, aois: Sequence[AOI], epoch_index: dict) -> dict:
    return {
        'seed': args.seed, 'n_train': args.n_train, 'n_val': args.n_val, 'inc_range': list(args.inc_range),
        'heading_asc': args.heading_asc, 'heading_desc': args.heading_desc,
        'width_m': cfg.width_m, 'length_m': cfg.length_m, 'spacing_m': cfg.spacing,
        'aois': [a.name for a in aois],
        'epochs': sorted(f'{k[0]}/{k[1]}/{d}' for k, v in epoch_index.items() for d in v),
    }


def outputs_meta(cfg: GenConfig) -> dict:
    return {'scenes': cfg.write_scenes, 'chips': cfg.chip, 'wavelength_m': cfg.wavelength,
            'altitude_m': cfg.altitude_m, 'slant': cfg.slant}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m hrdps_synthetic.generate', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--aoi-file', type=Path, default=DEFAULT_API_DUMP)
    p.add_argument('--aois', nargs='+', default=None, help='AOI names (default: all 45)')
    p.add_argument('--n-train', type=int, default=8500, help='training scenes')
    p.add_argument('--n-val', type=int, default=1500, help='validation scenes')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--inc-range', type=float, nargs=2, default=(20.0, 55.0), metavar=('MIN', 'MAX'))
    p.add_argument('--heading-asc', type=float, default=348.0)
    p.add_argument('--heading-desc', type=float, default=192.0)
    g = p.add_argument_group('scene size (km or pixels; pixels win if both are given)')
    g.add_argument('--width-km', type=float, default=20.0, help='east-west extent (default 20)')
    g.add_argument('--length-km', type=float, default=30.0, help='north-south extent (default 30)')
    g.add_argument('--width-px', type=int, default=None, help='east-west pixels (overrides --width-km)')
    g.add_argument('--height-px', type=int, default=None, help='north-south pixels (overrides --length-km)')
    g.add_argument('--spacing', type=float, default=10.0, help='pixel spacing (m)')
    c = p.add_argument_group('chips (e.g. 500 x 500 px for Detectron2)')
    c.add_argument('--chip-size', type=int, nargs='+', default=None, metavar='PX',
                   help='also cut every scene into chips of W [H] pixels')
    c.add_argument('--chip-stride', type=int, nargs='+', default=None, metavar='PX',
                   help='chip step SX [SY] in pixels (default: chip size, no overlap)')
    c.add_argument('--chip-edge', choices=EDGES, default='drop',
                   help='remainder narrower than a chip: drop it (default) or add a chip flush with the edge')
    c.add_argument('--no-scenes', action='store_true', help='write only chips, not the full scenes')
    c.add_argument('--scenes', dest='no_scenes', action='store_false', default=False,
                   help='write the full scenes (default; overrides no_scenes: true in a --config file)')
    p.add_argument('--wavelength', type=float, default=RCM_WAVELENGTH_M, help='radar wavelength (m)')
    p.add_argument('--altitude-km', type=float, default=RCM_ALTITUDE_M / 1e3, help='satellite altitude for the incidence ramp')
    p.add_argument('--slant', choices=SLANT_METHODS, default='projected')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--limit', type=int, default=None, help='render at most N pending samples this run')
    p.add_argument('--quicklook', type=int, default=0, metavar='N', help='PNG previews of the first N finished samples')
    p.add_argument('--replan', action='store_true', help='discard an existing plan.csv and build a new one')
    p.add_argument('--dry-run', action='store_true', help='report plan, storage and run time; write nothing')
    p.add_argument('--no-probe', action='store_true', help='skip the in-memory probe render in --dry-run')
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def config_from_args(args) -> GenConfig:
    width_m = args.width_px * args.spacing if args.width_px else args.width_km * 1e3
    length_m = args.height_px * args.spacing if args.height_px else args.length_km * 1e3
    try:
        spec = make_spec(args.chip_size, args.chip_stride, args.chip_edge)
    except ValueError as e:
        raise SystemExit(str(e))
    if args.no_scenes and spec is None:
        raise SystemExit('--no-scenes needs --chip-size (otherwise nothing would be written)')
    if spec is not None:
        nx, ny = int(round(width_m / args.spacing)), int(round(length_m / args.spacing))
        if spec.width > nx or spec.height > ny:
            raise SystemExit(f'Chip {spec.width} x {spec.height} px is larger than the {nx} x {ny} px scene')
    return GenConfig(workdir=str(args.workdir), aoi_file=str(args.aoi_file), aoi_names=args.aois,
                     width_m=width_m, length_m=length_m, spacing=args.spacing, wavelength=args.wavelength,
                     slant=args.slant, altitude_m=args.altitude_km * 1e3, write_scenes=not args.no_scenes,
                     chip=asdict(spec) if spec else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from hrdps_synthetic.config import parse_args, save_effective
    from hrdps_synthetic.download import setup_logging

    args = parse_args(build_parser(), argv, 'generate')

    setup_logging(args.verbose)
    aois = select_aois(load_aois(args.aoi_file), args.aois)
    cfg = config_from_args(args)
    epoch_index = ha.index_epochs(args.workdir)
    if not epoch_index:
        log.error('No epoch files under %s/epochs; run python -m hrdps_synthetic.download first', args.workdir)
        return 2

    if args.dry_run:
        return dry_run(cfg, aois, epoch_index, args)

    bad = check_epoch_coverage(aois, epoch_index, args.workdir, cfg.width_m, cfg.length_m)
    if bad:
        log.error('Epoch crops are too small for %.0f x %.0f km scenes at %s. Rerun the download with '
                  '--width-km/--length-km of at least this size into a new workdir.',
                  cfg.width_m / 1e3, cfg.length_m / 1e3, bad)
        return 2

    wd = Path(args.workdir)
    plan_path, meta_path, manifest_path = wd / 'plan.csv', wd / 'plan_meta.json', wd / 'manifest.csv'
    out_meta_path = wd / 'outputs_meta.json'
    meta = plan_meta(args, cfg, aois, epoch_index)
    if plan_path.exists() and not args.replan:
        old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if {k: v for k, v in old.items() if k != 'epochs'} != {k: v for k, v in meta.items() if k != 'epochs'}:
            log.error('plan.csv was built with different settings; rerun with --replan to rebuild it')
            return 2
        rows = read_plan(plan_path)
        log.info('Using existing plan: %d samples', len(rows))
    else:
        try:
            rows, summary = build_plan(aois, epoch_index, args.n_train, args.n_val, args.seed, tuple(args.inc_range),
                                       {'asc': args.heading_asc, 'desc': args.heading_desc}, cfg.width_m, cfg.length_m)
        except ShortfallError as e:
            log.error('Not enough unique samples: %s', e)
            return 2
        print(format_summary(summary, aois))
        write_plan(rows, plan_path)
        meta_path.write_text(json.dumps(meta, indent=1))
        for old_file in (manifest_path, out_meta_path):
            if old_file.exists():
                old_file.rename(old_file.with_suffix(old_file.suffix + '.old'))
        log.info('Wrote plan: %d samples', len(rows))

    done = read_manifest_ids(manifest_path)
    out_meta = outputs_meta(cfg)
    if out_meta_path.exists() and done and json.loads(out_meta_path.read_text()) != out_meta:
        log.error('This workdir already has samples rendered with different outputs (%s). Keep the same '
                  '--chip-size/--no-scenes settings, chip existing scenes with python -m hrdps_synthetic.tile, '
                  'or use --replan.', out_meta_path)
        return 2
    out_meta_path.write_text(json.dumps(out_meta, indent=1))
    log.info('Settings saved to %s', save_effective(args, 'generate', wd, argv))

    pending = [r for r in rows if r.sample_id not in done]
    pending.sort(key=lambda r: (r.aoi, r.direction, r.ref_date, r.sec_date))
    if args.limit is not None:
        pending = pending[:args.limit]
    spec = cfg.chip_spec
    log.info('%d of %d samples done; rendering %d with %d workers (%s%s)', len(done), len(rows), len(pending),
             args.workers, 'scenes' if cfg.write_scenes else 'no scenes',
             f' + {spec.describe()}' if spec else '')

    for name in sorted({r.aoi for r in pending}):  # build DEMs up front (network), one at a time
        ensure_aoi_dem(next(a for a in aois if a.name == name), wd, cfg.width_m, cfg.length_m, cfg.spacing)

    new_file = not manifest_path.exists()
    chips_out = ChipManifest(wd, spec) if spec else None
    t0, n_ok, n_chips, failed = time.time(), 0, 0, []
    try:
        with open(manifest_path, 'a', newline='') as mf:
            writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
            if new_file:
                writer.writeheader()
            ctx = mp.get_context('spawn')  # the parent has used threads (DEM reads); fork could deadlock
            with ctx.Pool(args.workers, initializer=_init_worker, initargs=(asdict(cfg),)) as pool:
                for res in pool.imap_unordered(_safe_work, pending, chunksize=4):
                    if 'error' in res:
                        failed.append(res['sample_id'])
                        log.error('%s failed: %s', res['sample_id'], res['error'])
                        continue
                    if chips_out is not None:
                        chips_out.write(res['chips'])
                        n_chips += len(res['chips'])
                    writer.writerow({k: res[k] for k in MANIFEST_FIELDS})
                    mf.flush()
                    n_ok += 1
                    if n_ok % 50 == 0 or n_ok == len(pending):
                        rate = (time.time() - t0) / n_ok
                        log.info('%d/%d rendered (%.1f s/sample wall, ~%.1f h left)', n_ok, len(pending), rate,
                                 rate * (len(pending) - n_ok) / 3600)
    finally:
        if chips_out is not None:
            chips_out.close()
    log.info('Rendered %d samples%s in %.1f min; %d failed', n_ok, f' ({n_chips} chips)' if spec else '',
             (time.time() - t0) / 60, len(failed))

    if args.quicklook:
        if not cfg.write_scenes:
            log.warning('--quicklook needs the full scenes; skipped because of --no-scenes')
        else:
            finished = read_manifest_ids(manifest_path)
            for row in [r for r in rows if r.sample_id in finished][:args.quicklook]:
                log.info('Quicklook %s', quicklook(wd, row, wd / 'quicklook', spec, spacing=cfg.spacing))
    return 1 if failed else 0


def _safe_work(row: PlanRow) -> dict:
    try:
        return _work(row)
    except Exception as e:  # report and keep going; the sample stays pending for the next run
        return {'sample_id': row.sample_id, 'error': f'{type(e).__name__}: {e}'}


if __name__ == '__main__':
    sys.exit(main())
