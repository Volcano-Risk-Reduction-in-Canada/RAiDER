"""Generate synthetic atmosphere-only interferograms from HRDPS epoch cubes.

Each sample is one (AOI, direction, date pair), rendered once:

* **Layout.** Radar geometry by default, the array layout of the synthetic
  magmatic interferograms: 3000 rows along-track by 2000 columns across-track
  at 10 m (30 x 20 km), rows in flight order, column 0 at near range, no CRS.
  ``--geometry north-up`` gives UTM rasters instead. Set the size with
  --width-km/--length-km or --width-px/--height-px; width is across-track (or
  E-W) and length along-track (or N-S).
* **Placement.** Scene centre = AOI centroid + a random shift; volcano polygons
  stay inside the scene.
* **Incidence.** Scene-centre incidence ~ U(20, 55) deg, with the real
  cross-track ramp.
* **Delay.** dSTD = (ZTD_sec - ZTD_ref)(lat, lon, DEM height) / cos(incidence).
* **Phase.** phase = 4*pi/wavelength * dSTD, no referencing. Positive is more
  delay (range increase) on the later date, as subsidence would produce; zero
  change is 0 rad. Wrapped phase is the same value reduced to [-pi, pi).
* **RGB.** The wrapped phase on the cyclic colour ramp, with 0 rad = red; see rgb.py.

Outputs (under --output-dir, default the workdir):

    plan.csv, plan_meta.json            the deterministic sample plan
    samples/<split>/<AOI>/<id>_unw.tif  unwrapped phase, float32, radians
    samples/<split>/<AOI>/<id>_wrp.tif  wrapped phase, float32, radians in [-pi, pi)
    samples/<split>/<AOI>/<id>_rgb.tif  wrapped phase as false-colour RGB, 3 x uint8 (--no-rgb to skip)
    manifest.csv                        one row per finished sample
    chips_<W>x<H>/...                   optional chips (--chip-size), see chips.py
    quicklook/<id>.png                  optional (--quicklook N)

Epochs and DEMs always live in the workdir, so several output folders (e.g.
different layouts) can share one download.

Training samples use training AOIs on training dates only; validation samples
use validation AOIs on validation dates only.

Examples (from the repository root):

    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --dry-run
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --workers 16
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --limit 5 --quicklook 5
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --chip-size 500          # scenes + 500 px chips
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --output-dir /data/hrdps_synth/northup --geometry north-up
"""

from __future__ import annotations

import argparse
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
from hrdps_synthetic.aois import DEFAULT_API_DUMP, AOI, GEOMETRIES, load_aois, select_aois, to_lonlat
from hrdps_synthetic.chips import (EDGES, ChipManifest, ChipSpec, chip_nbytes_in_memory, chip_windows, chips_per_scene,
                                   make_spec, write_chips)
from hrdps_synthetic.dem import ensure_aoi_dem, estimate_dem_bytes, maybe_dem, read_scene_dem, scene_dem_in_memory
from hrdps_synthetic.geometry import (RCM_ALTITUDE_M, R_EARTH_M, SceneGrid, coarse_lonlat, incidence_field,
                                      los_ground_to_satellite, look_azimuth_deg, scene_for, upsample)
from hrdps_synthetic.rgb import CONVENTION, PHASE_SIGN, RAMP_NAME, phase_to_rgb, rgb_profile, write_rgb
from hrdps_synthetic.sampling import (PLAN_FIELDS, PlanRow, ShortfallError, build_plan, format_summary, read_plan,
                                      summarise, write_plan)
from hrdps_synthetic.slant import SLANT_METHODS, Epoch, delta_ztd, epoch_indices, interpolate_cube, project, to_phase
from hrdps_synthetic.tables import CsvAppender, read_rows

log = logging.getLogger('hrdps_synthetic')

RCM_WAVELENGTH_M = 0.055466  # c / 5.405 GHz

MANIFEST_FIELDS = PLAN_FIELDS + (
    'geometry', 'flight_az_grid_deg', 'look_azimuth_deg', 'theta_min_deg', 'theta_max_deg',
    'los_east', 'los_north', 'los_up', 'center_lat', 'center_lon', 'center_x', 'center_y', 'corners_lonlat',
    'epsg', 'x0', 'y0', 'spacing_m', 'nx', 'ny', 'wavelength_m', 'phase_sign', 'dem_min_m', 'dem_max_m',
    'dstd_mean_m', 'dstd_std_m', 'dstd_min_m', 'dstd_max_m', 'altitude_m', 'n_chips',
    'unw_path', 'wrp_path', 'rgb_path', 'seconds',
)


@dataclass
class GenConfig:
    workdir: str
    aoi_file: str
    aoi_names: Optional[list]
    output_dir: str = ''
    width_m: float = 20_000.0
    length_m: float = 30_000.0
    spacing: float = 10.0
    geometry: str = 'radar'
    headings: tuple = (348.0, 192.0)  # asc, desc
    wavelength: float = RCM_WAVELENGTH_M
    slant: str = 'projected'
    altitude_m: float = RCM_ALTITUDE_M
    write_scenes: bool = True
    rgb: bool = True
    chip: Optional[dict] = None  # asdict(ChipSpec), or None for no chips

    @property
    def chip_spec(self) -> Optional[ChipSpec]:
        return ChipSpec(**self.chip) if self.chip else None

    @property
    def out(self) -> Path:
        return Path(self.output_dir or self.workdir)

    @property
    def footprint(self) -> dict:
        return dict(width_m=self.width_m, length_m=self.length_m, spacing=self.spacing, geometry=self.geometry,
                    headings=tuple(self.headings))


def scene_of_row(row: PlanRow, cfg: GenConfig, aoi: AOI) -> SceneGrid:
    return scene_for(aoi, row.offset_x_m, row.offset_y_m, cfg.width_m, cfg.length_m, cfg.spacing, cfg.geometry,
                     row.heading_deg)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(row: PlanRow, cfg: GenConfig, aoi: AOI, ref: Epoch, sec: Epoch, dem: np.ndarray):
    t0 = time.time()
    scene = scene_of_row(row, cfg, aoi)
    if dem.shape != scene.shape:
        raise ValueError(f'DEM shape {dem.shape} != scene {scene.shape}')
    lon_n, lat_n, nodes = coarse_lonlat(scene)
    xi_n, yi_n = epoch_indices(ref, lon_n, lat_n)
    xi = upsample(xi_n, nodes, scene.shape)
    yi = upsample(yi_n, nodes, scene.shape)
    dztd = interpolate_cube(delta_ztd(sec, ref), ref.z, xi, yi, dem)  # later minus earlier date
    del xi, yi
    theta = incidence_field(scene, row.theta_c_deg, row.heading_deg, R_EARTH_M, cfg.altitude_m)
    dstd = project(dztd, theta, cfg.slant)
    unw, wrp = to_phase(dstd, cfg.wavelength)
    rgb = phase_to_rgb(wrp) if cfg.rgb else None
    lon_c, lat_c = to_lonlat(scene.epsg, scene.cx, scene.cy)
    e, n, u = los_ground_to_satellite(row.theta_c_deg, row.heading_deg)
    meta = dict(
        asdict(row), geometry=scene.geometry, flight_az_grid_deg=scene.flight_az_grid_deg,
        look_azimuth_deg=look_azimuth_deg(row.heading_deg),
        theta_min_deg=round(float(theta.min()), 4), theta_max_deg=round(float(theta.max()), 4),
        los_east=round(e, 6), los_north=round(n, 6), los_up=round(u, 6),
        center_lat=round(float(lat_c), 6), center_lon=round(float(lon_c), 6), center_x=scene.cx, center_y=scene.cy,
        corners_lonlat=scene.corners_lonlat(), epsg=scene.epsg, x0=scene.x0, y0=scene.y0, spacing_m=scene.spacing,
        nx=scene.nx, ny=scene.ny, wavelength_m=cfg.wavelength, phase_sign=PHASE_SIGN, altitude_m=cfg.altitude_m,
        dem_min_m=round(float(dem.min()), 1), dem_max_m=round(float(dem.max()), 1),
        dstd_mean_m=float(dstd.mean()), dstd_std_m=float(dstd.std()),
        dstd_min_m=float(dstd.min()), dstd_max_m=float(dstd.max()),
        seconds=round(time.time() - t0, 2),
    )
    return scene, unw, wrp, rgb, meta


def _profile(scene: SceneGrid) -> dict:
    return dict(driver='GTiff', dtype='float32', width=scene.nx, height=scene.ny, count=1,
                crs=scene.crs, transform=scene.transform, compress='deflate', predictor=3,
                tiled=True, blockxsize=512, blockysize=512)


def _tags(meta: dict, kind: str) -> dict:
    keep = ('sample_id', 'split', 'aoi', 'direction', 'heading_deg', 'look_azimuth_deg', 'ref_date', 'sec_date',
            'geometry', 'flight_az_grid_deg', 'epsg', 'center_x', 'center_y', 'corners_lonlat',
            'theta_c_deg', 'theta_min_deg', 'theta_max_deg', 'los_east', 'los_north', 'los_up', 'wavelength_m',
            'phase_sign')
    tags = {k: str(meta[k]) for k in keep}
    tags.update(units='radians', quantity=f'{kind} interferometric phase, atmosphere only (HRDPS)',
                convention=CONVENTION, los_convention='unit vector ground->satellite in ENU')
    if meta['geometry'] == 'radar':
        tags['layout'] = ('rows along-track in flight order (row 0 = swath start), columns across-track from near '
                          'range (col 0); transform is the local metric frame, origin at the swath centre')
    return tags


def _rgb_tags(meta: dict) -> dict:
    t = _tags(meta, 'wrapped')
    t.update(quantity='wrapped phase, false-colour RGB', colormap=RAMP_NAME, rgb_phase_sign=PHASE_SIGN)
    t.pop('units')
    return t


def sample_paths(out_dir: Path, row: PlanRow) -> tuple[Path, Path, Path]:
    d = Path(out_dir) / 'samples' / row.split / row.aoi
    return d / f'{row.sample_id}_unw.tif', d / f'{row.sample_id}_wrp.tif', d / f'{row.sample_id}_rgb.tif'


def write_scene(scene: SceneGrid, unw, wrp, rgb, meta: dict, paths: tuple[Path, Path, Path]) -> None:
    import rasterio

    unw_path, wrp_path, rgb_path = paths
    unw_path.parent.mkdir(parents=True, exist_ok=True)
    for arr, path, kind in ((unw, unw_path, 'unwrapped'), (wrp, wrp_path, 'wrapped')):
        tmp = path.with_name(path.name + '.part')
        with rasterio.open(tmp, 'w', **_profile(scene)) as dst:
            dst.write(arr, 1)
            dst.update_tags(**_tags(meta, kind))
            dst.set_band_description(1, f'{kind} phase (rad)')
        tmp.replace(path)
    if rgb is not None:
        write_rgb(rgb_path, rgb, scene.crs, scene.transform, _rgb_tags(meta))


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
    wd, od = Path(cfg.workdir), cfg.out
    aoi = _STATE['aois'][row.aoi]
    scene = scene_of_row(row, cfg, aoi)
    fp = cfg.footprint
    dem_file = ensure_aoi_dem(aoi, wd, fp['width_m'], fp['length_m'], fp['spacing'], fp['geometry'], fp['headings'])
    dem = read_scene_dem(dem_file, scene)
    scene, unw, wrp, rgb, meta = render(row, cfg, aoi, _epoch(wd / row.ref_epoch), _epoch(wd / row.sec_epoch), dem)
    meta.update(unw_path='', wrp_path='', rgb_path='', n_chips=0, chips=[])
    if cfg.write_scenes:
        paths = sample_paths(od, row)
        write_scene(scene, unw, wrp, rgb, meta, paths)
        meta.update(unw_path=str(paths[0].relative_to(od)), wrp_path=str(paths[1].relative_to(od)),
                    rgb_path=str(paths[2].relative_to(od)) if rgb is not None else '')
    spec = cfg.chip_spec
    if spec:
        meta['chips'] = write_chips(od, spec, scene, unw, wrp, meta, cfg.altitude_m, rgb=rgb,
                                    rgb_tags={'colormap': RAMP_NAME, 'rgb_phase_sign': PHASE_SIGN}, convention=CONVENTION)
        meta['n_chips'] = len(meta['chips'])
    return meta


# ---------------------------------------------------------------------------
# Manifest, quicklooks
# ---------------------------------------------------------------------------
def read_manifest_ids(path: Path) -> set:
    return {r['sample_id'] for r in read_rows(path)[1]}


def quicklook(cfg: GenConfig, aoi: AOI, row: PlanRow, out_dir: Path, spec: Optional[ChipSpec] = None,
              decimate: int = 4) -> Optional[Path]:
    """PNG of DEM, unwrapped phase and RGB (needs the scene GeoTIFFs); chips outlined if `spec`."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio

    od = cfg.out
    unw_path, wrp_path, rgb_path = sample_paths(od, row)
    if not unw_path.exists():
        return None
    scene = scene_of_row(row, cfg, aoi)
    with rasterio.open(unw_path) as ds:
        unw = ds.read(1)[::decimate, ::decimate]
    if rgb_path.exists():
        with rasterio.open(rgb_path) as ds:
            rgb = np.moveaxis(ds.read()[:, ::decimate, ::decimate], 0, -1)
    else:
        with rasterio.open(wrp_path) as ds:
            rgb = np.moveaxis(phase_to_rgb(ds.read(1)[::decimate, ::decimate]), 0, -1)
    if scene.geometry == 'north-up':
        xmin, ymin, xmax, ymax = scene.bounds
        ext = (xmin / 1e3, xmax / 1e3, ymin / 1e3, ymax / 1e3)
        xlabel, ylabel = 'UTM easting (km)', 'UTM northing (km)'
    else:
        ext = (0, scene.nx * scene.spacing / 1e3, scene.ny * scene.spacing / 1e3, 0)
        xlabel, ylabel = 'across-track (km, near to far range)', 'along-track (km, flight direction down)'
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    dem_file = maybe_dem(Path(cfg.workdir), aoi, cfg.spacing)
    if dem_file is not None:
        try:
            dem = read_scene_dem(dem_file, scene)[::decimate, ::decimate]
            im = axes[0].imshow(dem, extent=ext, cmap='terrain')
            fig.colorbar(im, ax=axes[0], shrink=0.7, label='height (m)')
            axes[0].set_title('DEM (GLO-30)')
        except ValueError:
            pass
    med = float(np.nanmedian(unw))
    lim = float(np.nanpercentile(np.abs(unw - med), 99)) or 1.0
    im = axes[1].imshow(unw, extent=ext, cmap='RdBu_r', vmin=med - lim, vmax=med + lim)
    fig.colorbar(im, ax=axes[1], shrink=0.7, label='rad')
    axes[1].set_title('unwrapped phase')
    axes[2].imshow(rgb, extent=ext)
    axes[2].set_title('wrapped phase (RGB, 0 rad = red)')
    for a in axes:
        a.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f'{row.sample_id}  {scene.geometry}  heading {row.heading_deg:.0f}  '
                 f'incidence {row.theta_c_deg:.1f} deg')
    if spec is not None:  # outline the chips on the phase panels
        from matplotlib.patches import Rectangle

        px_x = (ext[1] - ext[0]) / scene.nx
        px_y = (ext[3] - ext[2]) / scene.ny  # negative for radar (row 0 at the top of the extent)
        for _, _, r0, c0 in chip_windows(scene.ny, scene.nx, spec):
            for a in axes[1:]:
                a.add_patch(Rectangle((ext[0] + c0 * px_x, ext[3] - (r0 + spec.height) * px_y),
                                      spec.width * px_x, spec.height * px_y, fill=False, lw=0.5, ec='k'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{row.sample_id}.png'
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def probe(row: PlanRow, cfg: GenConfig, aoi: AOI) -> tuple[int, int, float]:
    """Render one sample in memory: (scene bytes, chip bytes, seconds) for every product."""
    from rasterio.io import MemoryFile

    wd = Path(cfg.workdir)
    t0 = time.time()
    scene = scene_of_row(row, cfg, aoi)
    existing = maybe_dem(wd, aoi, cfg.spacing)
    try:
        dem = read_scene_dem(existing, scene) if existing else scene_dem_in_memory(aoi, scene)
    except ValueError:  # existing DEM too small for these scenes
        dem = scene_dem_in_memory(aoi, scene)
    t_dem = time.time() - t0
    t1 = time.time()
    scene, unw, wrp, rgb, meta = render(row, cfg, aoi, Epoch.load(wd / row.ref_epoch), Epoch.load(wd / row.sec_epoch),
                                        dem)
    scene_bytes = 0
    if cfg.write_scenes:
        for arr, kind in ((unw, 'unwrapped'), (wrp, 'wrapped')):
            with MemoryFile() as mf:
                with mf.open(**_profile(scene)) as dst:
                    dst.write(arr, 1)
                    dst.update_tags(**_tags(meta, kind))
                scene_bytes += len(mf.read())
        if rgb is not None:
            with MemoryFile() as mf:
                with mf.open(**rgb_profile(scene.nx, scene.ny, scene.crs, scene.transform)) as dst:
                    dst.write(rgb)
                scene_bytes += len(mf.read())
    spec = cfg.chip_spec
    chip_bytes = chip_nbytes_in_memory(spec, scene, unw, wrp, meta, rgb) if spec else 0
    secs = time.time() - t1
    log.info('Probe %s: scenes %.1f MB, chips %.1f MB, %.1f s render+compress (+%.1f s DEM), dSTD %.1f..%.1f mm',
             row.sample_id, scene_bytes / 1e6, chip_bytes / 1e6, secs, t_dem,
             1e3 * meta['dstd_min_m'], 1e3 * meta['dstd_max_m'])
    return scene_bytes, chip_bytes, secs


def check_epoch_coverage(aois: Sequence[AOI], epoch_index: dict, workdir: Path, cfg: GenConfig) -> list[str]:
    """AOIs whose epoch crops are too small for every possible scene of this size and layout."""
    problems = []
    for a in aois:
        paths = [p for d in ('asc', 'desc') for p in epoch_index.get((a.name, d), {}).values()]
        if not paths:
            continue
        with xr.open_dataset(Path(workdir) / paths[0]) as ds:
            x0, x1, y0, y1 = float(ds.x.min()), float(ds.x.max()), float(ds.y.min()), float(ds.y.max())
        W, S, E, N = a.reach_lonlat(width=cfg.width_m, length=cfg.length_m, geometry=cfg.geometry,
                                    headings=cfg.headings).bounds
        if W < x0 or E > x1 or S < y0 or N > y1:
            problems.append(a.name)
    return problems


def dry_run(cfg: GenConfig, aois: Sequence[AOI], epoch_index: dict, args) -> int:
    wd = Path(cfg.workdir)
    n_req = args.n_train + args.n_val
    nx, ny = int(round(cfg.width_m / cfg.spacing)), int(round(cfg.length_m / cfg.spacing))
    axes = 'across x along-track' if cfg.geometry == 'radar' else 'E-W x N-S'
    print(f'Scene: {cfg.geometry}, {nx} x {ny} px at {cfg.spacing:g} m ({cfg.width_m / 1e3:g} x '
          f'{cfg.length_m / 1e3:g} km, {axes}); output folder {cfg.out}')
    bad = check_epoch_coverage(aois, epoch_index, wd, cfg)
    if bad:
        print(f'Epoch crops too small for these scenes: {bad}. Rerun the download with matching '
              f'--width-km/--length-km into a new workdir.')
    try:
        summary = summarise(aois, epoch_index, args.n_train, args.n_val, args.seed, cfg.width_m, cfg.length_m,
                            cfg.geometry, cfg.headings)
        print(format_summary(summary, aois))
    except ShortfallError as e:
        print(f'Cannot build a plan from the epochs on disk: {e}')
        summary = None

    per_scene_raw = (2 * 4 + (3 if cfg.rgb else 0)) * nx * ny
    print(f'\nSamples requested: {n_req} ({args.n_train} train + {args.n_val} val)')
    if cfg.write_scenes:
        print(f'Scenes, uncompressed: {per_scene_raw / 1e6:.0f} MB per sample (unwrapped + wrapped'
              f'{" + RGB" if cfg.rgb else ""}), {n_req * per_scene_raw / 1e9:.1f} GB total')
    spec = cfg.chip_spec
    if spec:
        k = chips_per_scene(ny, nx, spec)
        cov = k * spec.width * spec.height / (nx * ny)
        per_chip = (2 * 4 + (3 if cfg.rgb else 0)) * spec.width * spec.height
        print(f'Chips: {spec.describe()} -> {k} per scene ({100 * cov:.0f}% of scene pixels), {n_req * k} in total; '
              f'{n_req * k * per_chip / 1e9:.1f} GB uncompressed')

    src = dst = 0
    fp = cfg.footprint
    missing = [a for a in aois if not maybe_dem(wd, a, cfg.spacing)]
    for a in missing:
        s_, d_ = estimate_dem_bytes(a, fp['width_m'], fp['length_m'], fp['spacing'], fp['geometry'], fp['headings'])
        src, dst = src + s_, dst + d_
    print(f'DEMs to build: {len(missing)} of {len(aois)} AOIs (existing ones are rebuilt if too small); '
          f'~{src / 1e9:.1f} GB of GLO-30 reads, ~{dst / 1e9:.1f} GB of {cfg.spacing:g} m UTM DEMs uncompressed')

    if summary is not None and not args.no_probe and not bad:
        rows = None
        try:
            rows, _ = build_plan(aois, epoch_index, args.n_train, args.n_val, args.seed, tuple(args.inc_range),
                                 {'asc': args.heading_asc, 'desc': args.heading_desc}, cfg.width_m, cfg.length_m,
                                 cfg.geometry)
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
        'width_m': cfg.width_m, 'length_m': cfg.length_m, 'spacing_m': cfg.spacing, 'geometry': cfg.geometry,
        'aois': [a.name for a in aois],
        'epochs': sorted(f'{k[0]}/{k[1]}/{d}' for k, v in epoch_index.items() for d in v),
    }


def outputs_meta(cfg: GenConfig) -> dict:
    return {'scenes': cfg.write_scenes, 'rgb': cfg.rgb, 'chips': cfg.chip, 'geometry': cfg.geometry,
            'phase_sign': PHASE_SIGN, 'wavelength_m': cfg.wavelength, 'altitude_m': cfg.altitude_m,
            'slant': cfg.slant}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m hrdps_synthetic.generate', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workdir', type=Path, required=True, help='folder with epochs/ (from download) and dem/')
    p.add_argument('--output-dir', type=Path, default=None,
                   help='folder for plan, manifest, samples, chips and quicklooks (default: the workdir)')
    p.add_argument('--aoi-file', type=Path, default=DEFAULT_API_DUMP)
    p.add_argument('--aois', nargs='+', default=None, help='AOI names (default: all 45)')
    p.add_argument('--n-train', type=int, default=8500, help='training scenes')
    p.add_argument('--n-val', type=int, default=1500, help='validation scenes')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--inc-range', type=float, nargs=2, default=(20.0, 55.0), metavar=('MIN', 'MAX'))
    p.add_argument('--heading-asc', type=float, default=348.0)
    p.add_argument('--heading-desc', type=float, default=192.0)
    g = p.add_argument_group('scene layout and size (km or pixels; pixels win if both are given)')
    g.add_argument('--geometry', choices=GEOMETRIES, default='radar',
                   help='radar: rows along-track, columns across-track, like the magmatic interferograms '
                        '(default); north-up: UTM rasters')
    g.add_argument('--width-km', type=float, default=20.0, help='across-track (or E-W) extent (default 20)')
    g.add_argument('--length-km', type=float, default=30.0, help='along-track (or N-S) extent (default 30)')
    g.add_argument('--width-px', type=int, default=None, help='columns (overrides --width-km)')
    g.add_argument('--height-px', type=int, default=None, help='rows (overrides --length-km)')
    g.add_argument('--spacing', type=float, default=10.0, help='pixel spacing (m)')
    o = p.add_argument_group('products')
    o.add_argument('--rgb', action=argparse.BooleanOptionalAction, default=True,
                   help='write the false-colour RGB of the wrapped phase (default on)')
    o.add_argument('--chip-size', type=int, nargs='+', default=None, metavar='PX',
                   help='also cut every scene into chips of W [H] pixels (e.g. 500 for Detectron2)')
    o.add_argument('--chip-stride', type=int, nargs='+', default=None, metavar='PX',
                   help='chip step SX [SY] in pixels (default: chip size, no overlap)')
    o.add_argument('--chip-edge', choices=EDGES, default='drop',
                   help='remainder narrower than a chip: drop it (default) or add a chip flush with the edge')
    o.add_argument('--no-scenes', action='store_true', help='write only chips, not the full scenes')
    o.add_argument('--scenes', dest='no_scenes', action='store_false', default=False,
                   help='write the full scenes (default; overrides no_scenes: true in a --config file)')
    p.add_argument('--wavelength', type=float, default=RCM_WAVELENGTH_M, help='radar wavelength (m), RCM 0.055466')
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
                     output_dir=str(args.output_dir or args.workdir), width_m=width_m, length_m=length_m,
                     spacing=args.spacing, geometry=args.geometry, headings=(args.heading_asc, args.heading_desc),
                     wavelength=args.wavelength, slant=args.slant, altitude_m=args.altitude_km * 1e3,
                     write_scenes=not args.no_scenes, rgb=args.rgb, chip=asdict(spec) if spec else None)


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

    bad = check_epoch_coverage(aois, epoch_index, args.workdir, cfg)
    if bad:
        log.error('Epoch crops are too small for %.0f x %.0f km %s scenes at %s. Rerun the download with '
                  '--width-km/--length-km of at least this size into a new workdir.',
                  cfg.width_m / 1e3, cfg.length_m / 1e3, cfg.geometry, bad)
        return 2

    wd, od = Path(args.workdir), cfg.out
    od.mkdir(parents=True, exist_ok=True)
    plan_path, meta_path, manifest_path = od / 'plan.csv', od / 'plan_meta.json', od / 'manifest.csv'
    out_meta_path = od / 'outputs_meta.json'
    meta = plan_meta(args, cfg, aois, epoch_index)
    if plan_path.exists() and not args.replan:
        old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if {k: v for k, v in old.items() if k != 'epochs'} != {k: v for k, v in meta.items() if k != 'epochs'}:
            log.error('%s was built with different settings; rerun with --replan to rebuild it, or use a new '
                      '--output-dir', plan_path)
            return 2
        rows = read_plan(plan_path)
        log.info('Using existing plan: %d samples', len(rows))
    else:
        try:
            rows, summary = build_plan(aois, epoch_index, args.n_train, args.n_val, args.seed, tuple(args.inc_range),
                                       {'asc': args.heading_asc, 'desc': args.heading_desc}, cfg.width_m,
                                       cfg.length_m, cfg.geometry)
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
        log.error('%s already has samples rendered with different outputs or conventions (%s). Use a new '
                  '--output-dir, keep the same settings, or --replan. For chips or RGB of existing scenes use '
                  'python -m hrdps_synthetic.tile / python -m hrdps_synthetic.rgb.', od, out_meta_path)
        return 2
    out_meta_path.write_text(json.dumps(out_meta, indent=1))
    log.info('Settings saved to %s', save_effective(args, 'generate', od, argv))

    pending = [r for r in rows if r.sample_id not in done]
    pending.sort(key=lambda r: (r.aoi, r.direction, r.ref_date, r.sec_date))
    if args.limit is not None:
        pending = pending[:args.limit]
    spec = cfg.chip_spec
    log.info('%d of %d samples done; rendering %d with %d workers (%s, %s%s%s)', len(done), len(rows), len(pending),
             args.workers, cfg.geometry, 'scenes' if cfg.write_scenes else 'no scenes', ' + RGB' if cfg.rgb else '',
             f' + {spec.describe()}' if spec else '')

    fp = cfg.footprint
    aoi_map = {a.name: a for a in aois}
    for name in sorted({r.aoi for r in pending}):  # build DEMs up front (network), one at a time
        ensure_aoi_dem(aoi_map[name], wd, fp['width_m'], fp['length_m'], fp['spacing'], fp['geometry'], fp['headings'])

    manifest = CsvAppender(manifest_path, MANIFEST_FIELDS)
    chips_out = ChipManifest(od, spec) if spec else None
    t0, n_ok, n_chips, failed = time.time(), 0, 0, []
    try:
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
                manifest.write(res)
                n_ok += 1
                if n_ok % 50 == 0 or n_ok == len(pending):
                    rate = (time.time() - t0) / n_ok
                    log.info('%d/%d rendered (%.1f s/sample wall, ~%.1f h left)', n_ok, len(pending), rate,
                             rate * (len(pending) - n_ok) / 3600)
    finally:
        manifest.close()
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
                log.info('Quicklook %s', quicklook(cfg, aoi_map[row.aoi], row, od / 'quicklook', spec))
    return 1 if failed else 0


def _safe_work(row: PlanRow) -> dict:
    try:
        return _work(row)
    except Exception as e:  # report and keep going; the sample stays pending for the next run
        return {'sample_id': row.sample_id, 'error': f'{type(e).__name__}: {e}'}


if __name__ == '__main__':
    sys.exit(main())
