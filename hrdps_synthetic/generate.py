"""Generate synthetic atmosphere-only interferograms from HRDPS epoch cubes.

Each sample is one (AOI, direction, date pair), rendered once:

* north-up UTM grid, 20 km E-W x 30 km N-S at 10 m (2000 x 3000 pixels);
* scene centre = AOI centroid + random shift (volcano polygons stay inside);
* scene-centre incidence ~ U(20, 55) deg with the real cross-track ramp;
* dSTD = (ZTD_ref - ZTD_sec)(lat, lon, DEM height) / cos(incidence);
* unwrapped phase = 4*pi/lambda * dSTD (radians, no referencing), and its wrap.

Outputs (under --workdir):

    plan.csv, plan_meta.json            the deterministic sample plan
    samples/<split>/<AOI>/<id>_unw.tif  unwrapped phase, float32, radians
    samples/<split>/<AOI>/<id>_wrp.tif  wrapped phase, float32, radians in [-pi, pi)
    manifest.csv                        one row per finished sample
    quicklook/<id>.png                  optional (--quicklook N)

Training samples use training AOIs on training dates only; validation samples
use validation AOIs on validation dates only.

Examples (from the repository root):

    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --dry-run
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --workers 16
    python -m hrdps_synthetic.generate --workdir /data/hrdps_synth --limit 5 --quicklook 5
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

from hrdps_synthetic import hrdps_archive as ha
from hrdps_synthetic.aois import DEFAULT_API_DUMP, AOI, load_aois, select_aois, to_lonlat
from hrdps_synthetic.dem import ensure_aoi_dem, estimate_dem_bytes, maybe_dem, read_scene_dem, scene_dem_in_memory
from hrdps_synthetic.geometry import (RCM_ALTITUDE_M, R_EARTH_M, SceneGrid, coarse_lonlat, incidence_field,
                                      look_azimuth_deg, scene_for, upsample)
from hrdps_synthetic.sampling import (PLAN_FIELDS, PlanRow, ShortfallError, build_plan, format_summary, read_plan,
                                      summarise, write_plan)
from hrdps_synthetic.slant import SLANT_METHODS, Epoch, delta_ztd, epoch_indices, interpolate_cube, project, to_phase

log = logging.getLogger('hrdps_synthetic')

RCM_WAVELENGTH_M = 0.055466  # 5.405 GHz

MANIFEST_FIELDS = PLAN_FIELDS + (
    'look_azimuth_deg', 'theta_min_deg', 'theta_max_deg', 'los_east', 'los_north', 'los_up',
    'center_lat', 'center_lon', 'center_x', 'center_y', 'epsg', 'x0', 'y0', 'spacing_m', 'nx', 'ny',
    'wavelength_m', 'dem_min_m', 'dem_max_m', 'dstd_mean_m', 'dstd_std_m', 'dstd_min_m', 'dstd_max_m',
    'unw_path', 'wrp_path', 'seconds',
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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def los_ground_to_satellite(theta_deg: float, heading_deg: float) -> tuple[float, float, float]:
    """Unit vector (east, north, up) from the ground pixel toward a right-looking satellite.

    Heading is clockwise from true north; the look azimuth is heading + 90.
    A LOS displacement d_los = d . los is positive toward the satellite.
    """
    th, az = math.radians(theta_deg), math.radians(look_azimuth_deg(heading_deg))
    return (-math.sin(th) * math.sin(az), -math.sin(th) * math.cos(az), math.cos(th))


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
        wavelength_m=cfg.wavelength,
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
    unw_path, wrp_path = sample_paths(wd, row)
    write_geotiffs(scene, unw, wrp, meta, unw_path, wrp_path)
    meta.update(unw_path=str(unw_path.relative_to(wd)), wrp_path=str(wrp_path.relative_to(wd)))
    return meta


# ---------------------------------------------------------------------------
# Manifest, quicklooks
# ---------------------------------------------------------------------------
def read_manifest_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline='') as f:
        return {r['sample_id'] for r in csv.DictReader(f)}


def quicklook(workdir: Path, row: PlanRow, out_dir: Path, decimate: int = 4) -> Path:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio

    unw_path, wrp_path = sample_paths(workdir, row)
    with rasterio.open(unw_path) as ds:
        unw = ds.read(1)[::decimate, ::decimate]
        ext = (ds.bounds.left / 1e3, ds.bounds.right / 1e3, ds.bounds.bottom / 1e3, ds.bounds.top / 1e3)
    with rasterio.open(wrp_path) as ds:
        wrp = ds.read(1)[::decimate, ::decimate]
    aoi_dem = Path(workdir) / 'dem' / f'{row.aoi}_glo30_utm10m.tif'
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
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{row.sample_id}.png'
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def probe(row: PlanRow, cfg: GenConfig, aoi: AOI) -> Optional[tuple[int, float]]:
    """Render one sample fully in memory: (compressed bytes for both GeoTIFFs, seconds)."""
    import rasterio
    from rasterio.io import MemoryFile

    wd = Path(cfg.workdir)
    t0 = time.time()
    scene = scene_for(aoi, row.offset_x_m, row.offset_y_m, cfg.width_m, cfg.length_m, cfg.spacing)
    existing = maybe_dem(wd, aoi)
    dem = read_scene_dem(existing, scene) if existing else scene_dem_in_memory(aoi, scene)
    t_dem = time.time() - t0
    t1 = time.time()
    scene, unw, wrp, meta = render(row, cfg, aoi, Epoch.load(wd / row.ref_epoch), Epoch.load(wd / row.sec_epoch), dem)
    nbytes = 0
    for arr, kind in ((unw, 'unwrapped'), (wrp, 'wrapped')):
        with MemoryFile() as mf:
            with mf.open(**_profile(scene)) as dst:
                dst.write(arr, 1)
                dst.update_tags(**_tags(meta, kind))
            nbytes += len(mf.read())
    secs = time.time() - t1
    log.info('Probe %s: %.1f MB for both GeoTIFFs, %.1f s render+compress (+%.1f s DEM), dSTD %.1f..%.1f mm',
             row.sample_id, nbytes / 1e6, secs, t_dem, 1e3 * meta['dstd_min_m'], 1e3 * meta['dstd_max_m'])
    return nbytes, secs


def dry_run(cfg: GenConfig, aois: Sequence[AOI], epoch_index: dict, args) -> int:
    wd = Path(cfg.workdir)
    n_req = args.n_train + args.n_val
    try:
        summary = summarise(aois, epoch_index, args.n_train, args.n_val, args.seed)
        print(format_summary(summary, aois))
    except ShortfallError as e:
        print(f'Cannot build a plan from the epochs on disk: {e}')
        summary = None
    per_sample_raw = 2 * 4 * int(round(cfg.width_m / cfg.spacing)) * int(round(cfg.length_m / cfg.spacing))
    print(f'\nSamples requested: {n_req} ({args.n_train} train + {args.n_val} val)')
    print(f'Uncompressed output: {per_sample_raw / 1e6:.0f} MB per sample, {n_req * per_sample_raw / 1e9:.1f} GB total')

    src = dst = 0
    missing = [a for a in aois if not maybe_dem(wd, a)]
    for a in missing:
        s, d = estimate_dem_bytes(a, cfg.width_m, cfg.length_m, cfg.spacing)
        src, dst = src + s, dst + d
    print(f'DEMs to build: {len(missing)} of {len(aois)} AOIs; ~{src / 1e9:.1f} GB of GLO-30 reads, '
          f'~{dst / 1e9:.1f} GB of 10 m UTM DEMs (uncompressed; written compressed)')

    if summary is not None and not args.no_probe:
        rows = None
        try:
            rows, _ = build_plan(aois, epoch_index, args.n_train, args.n_val, args.seed, tuple(args.inc_range),
                                 {'asc': args.heading_asc, 'desc': args.heading_desc})
        except ShortfallError as e:
            print(f'Plan would fail: {e}')
        if rows:
            aoi_map = {a.name: a for a in aois}
            nbytes, secs = probe(rows[0], cfg, aoi_map[rows[0].aoi])
            print(f'Probe sample: {nbytes / 1e6:.1f} MB compressed (both files), {secs:.1f} s to render and compress')
            print(f'Estimated output: ~{n_req * nbytes / 1e9:.1f} GB on disk; '
                  f'~{n_req * secs / 3600:.2f} CPU-hours (~{n_req * secs / 3600 / max(args.workers, 1):.2f} h '
                  f'with {args.workers} workers, plus DEM building)')
    elif args.no_probe:
        print('Probe skipped (--no-probe).')
    print('Dry run: nothing was written.')
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plan_meta(args, aois: Sequence[AOI], epoch_index: dict) -> dict:
    return {
        'seed': args.seed, 'n_train': args.n_train, 'n_val': args.n_val, 'inc_range': list(args.inc_range),
        'heading_asc': args.heading_asc, 'heading_desc': args.heading_desc,
        'aois': [a.name for a in aois],
        'epochs': sorted(f'{k[0]}/{k[1]}/{d}' for k, v in epoch_index.items() for d in v),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='python -m hrdps_synthetic.generate', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--aoi-file', type=Path, default=DEFAULT_API_DUMP)
    p.add_argument('--aois', nargs='+', default=None, help='AOI names (default: all 45)')
    p.add_argument('--n-train', type=int, default=8500)
    p.add_argument('--n-val', type=int, default=1500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--inc-range', type=float, nargs=2, default=(20.0, 55.0), metavar=('MIN', 'MAX'))
    p.add_argument('--heading-asc', type=float, default=348.0)
    p.add_argument('--heading-desc', type=float, default=192.0)
    p.add_argument('--width-km', type=float, default=20.0)
    p.add_argument('--length-km', type=float, default=30.0)
    p.add_argument('--spacing', type=float, default=10.0, help='pixel spacing (m)')
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
    args = p.parse_args(argv)

    from hrdps_synthetic.download import setup_logging

    setup_logging(args.verbose)
    aois = select_aois(load_aois(args.aoi_file), args.aois)
    cfg = GenConfig(workdir=str(args.workdir), aoi_file=str(args.aoi_file), aoi_names=args.aois,
                    width_m=args.width_km * 1e3, length_m=args.length_km * 1e3, spacing=args.spacing,
                    wavelength=args.wavelength, slant=args.slant, altitude_m=args.altitude_km * 1e3)
    epoch_index = ha.index_epochs(args.workdir)
    if not epoch_index:
        log.error('No epoch files under %s/epochs; run python -m hrdps_synthetic.download first', args.workdir)
        return 2

    if args.dry_run:
        return dry_run(cfg, aois, epoch_index, args)

    wd = Path(args.workdir)
    plan_path, meta_path, manifest_path = wd / 'plan.csv', wd / 'plan_meta.json', wd / 'manifest.csv'
    meta = plan_meta(args, aois, epoch_index)
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
                                       {'asc': args.heading_asc, 'desc': args.heading_desc})
        except ShortfallError as e:
            log.error('Not enough unique samples: %s', e)
            return 2
        print(format_summary(summary, aois))
        write_plan(rows, plan_path)
        meta_path.write_text(json.dumps(meta, indent=1))
        if manifest_path.exists():
            manifest_path.rename(manifest_path.with_suffix('.csv.old'))
        log.info('Wrote plan: %d samples', len(rows))

    done = read_manifest_ids(manifest_path)
    pending = [r for r in rows if r.sample_id not in done]
    pending.sort(key=lambda r: (r.aoi, r.direction, r.ref_date, r.sec_date))
    if args.limit is not None:
        pending = pending[:args.limit]
    log.info('%d of %d samples done; rendering %d with %d workers', len(done), len(rows), len(pending), args.workers)

    for name in sorted({r.aoi for r in pending}):  # build DEMs up front (network), one at a time
        ensure_aoi_dem(next(a for a in aois if a.name == name), wd, cfg.width_m, cfg.length_m, cfg.spacing)

    new_file = not manifest_path.exists()
    t0, n_ok, failed = time.time(), 0, []
    with open(manifest_path, 'a', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if new_file:
            writer.writeheader()
        ctx = mp.get_context('spawn')  # the parent has used threads (DEM reads); fork could deadlock
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=(asdict(cfg),)) as pool:
            results = pool.imap_unordered(_safe_work, pending, chunksize=4)
            for res in results:
                if 'error' in res:
                    failed.append(res['sample_id'])
                    log.error('%s failed: %s', res['sample_id'], res['error'])
                    continue
                writer.writerow({k: res[k] for k in MANIFEST_FIELDS})
                mf.flush()
                n_ok += 1
                if n_ok % 50 == 0 or n_ok == len(pending):
                    rate = (time.time() - t0) / n_ok
                    log.info('%d/%d rendered (%.1f s/sample wall, ~%.1f h left)', n_ok, len(pending), rate,
                             rate * (len(pending) - n_ok) / 3600)
    log.info('Rendered %d samples in %.1f min; %d failed', n_ok, (time.time() - t0) / 60, len(failed))

    if args.quicklook:
        finished = read_manifest_ids(manifest_path)
        for row in [r for r in rows if r.sample_id in finished][:args.quicklook]:
            log.info('Quicklook %s', quicklook(wd, row, wd / 'quicklook'))
    return 1 if failed else 0


def _safe_work(row: PlanRow) -> dict:
    try:
        return _work(row)
    except Exception as e:  # report and keep going; the sample stays pending for the next run
        return {'sample_id': row.sample_id, 'error': f'{type(e).__name__}: {e}'}


if __name__ == '__main__':
    sys.exit(main())
