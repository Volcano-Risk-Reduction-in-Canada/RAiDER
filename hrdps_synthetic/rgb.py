"""False-colour RGB wrapped interferograms, and a post-processor for existing outputs.

Colour ramp
-----------
The 9-stop cyclic ramp of ``applyEVPhaseColorRamp.m``, the same colours the
synthetic magmatic interferograms use (VRRC_Synthetic_magmatics_InSAR,
``colormap.py``). Wrapped phase in [-pi, pi) maps linearly onto the ramp:

    position = (phase + pi) / (2 pi)          (mod 1)

so **0 rad (no phase change) is the ramp midpoint, pure red (255, 0, 0)**, and
+/-pi are the ends, teal (0, 191, 169). Colours are interpolated linearly between
stops and rounded to uint8.

Phase convention
----------------
Everything rendered here uses

    phase = 4 pi / wavelength x (STD_sec - STD_ref)

Positive phase is more delay (a longer range) on the later date, as subsidence
would produce, and zero change is 0 rad. Outputs written before this convention
existed have no ``phase_sign`` tag and record ``(STD_ref - STD_sec)`` in their
``convention`` tag. ``--sign auto`` (the default) flips them so every RGB
follows the current convention. Their float products are not modified.

Post-processing existing outputs
--------------------------------
    python -m hrdps_synthetic.rgb --workdir /data/hrdps_synth --dry-run
    python -m hrdps_synthetic.rgb --workdir /data/hrdps_synth --workers 8

This writes ``<id>_rgb.tif`` next to every ``<id>_wrp.tif``, for scenes listed in
manifest.csv and for chips listed in every chips_*/chips.csv. The RGB files are
3-band uint8 GeoTIFFs with the same georeferencing as their wrapped file. Each
manifest gains an ``rgb_path`` column. Existing RGB files are skipped unless you
pass ``--overwrite``.
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from hrdps_synthetic.tables import read_rows, rewrite

log = logging.getLogger('hrdps_synthetic')

RAMP_STOPS = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
RAMP_RGB = np.array([
    (0, 191, 169), (0, 60, 248), (102, 0, 234), (217, 0, 133), (255, 0, 0),
    (212, 142, 0), (98, 236, 0), (0, 253, 35), (0, 191, 169),
], dtype=np.float64)
RAMP_NAME = 'EV cyclic 9-stop (applyEVPhaseColorRamp.m); position = (phase + pi) / 2pi; 0 rad = red'

PHASE_SIGN = 'sec_minus_ref'
LEGACY_SIGN = 'ref_minus_sec'
CONVENTION = ('phase = 4*pi/wavelength * (STD_sec - STD_ref); positive = more delay (range increase) on the later '
              'date, as for subsidence; zero change = 0 rad; STD = ZTD/cos(incidence); no referencing')
SIGN_MODES = ('auto', 'keep', 'flip')


def phase_to_rgb(wrapped: np.ndarray) -> np.ndarray:
    """(3, H, W) uint8 RGB of wrapped phase (radians) on the cyclic ramp; 0 rad -> red."""
    pos = np.mod((np.asarray(wrapped, dtype=np.float64) + math.pi) / (2.0 * math.pi), 1.0)
    out = np.empty((3,) + pos.shape, dtype=np.uint8)
    for k in range(3):
        out[k] = np.rint(np.interp(pos, RAMP_STOPS, RAMP_RGB[:, k]))
    return out


def rgb_profile(width: int, height: int, crs, transform, tiled: bool = True) -> dict:
    p = dict(driver='GTiff', dtype='uint8', width=width, height=height, count=3, crs=crs, transform=transform,
             compress='deflate', interleave='pixel', photometric='RGB')
    if tiled and width >= 512 and height >= 512:
        p.update(tiled=True, blockxsize=512, blockysize=512)
    return p


def write_rgb(path: Path, rgb: np.ndarray, crs, transform, tags: dict, tiled: bool = True) -> None:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    with rasterio.open(tmp, 'w', **rgb_profile(rgb.shape[2], rgb.shape[1], crs, transform, tiled)) as dst:
        dst.write(rgb)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
        for i, name in enumerate(('red', 'green', 'blue'), start=1):
            dst.set_band_description(i, name)
    tmp.replace(path)


def file_phase_sign(tags: dict) -> Optional[str]:
    """Sign convention recorded in a wrapped/unwrapped GeoTIFF's tags (None if unknown)."""
    if tags.get('phase_sign') in (PHASE_SIGN, LEGACY_SIGN):
        return tags['phase_sign']
    conv = tags.get('convention', '')
    if 'STD_ref - STD_sec' in conv:
        return LEGACY_SIGN
    if 'STD_sec - STD_ref' in conv:
        return PHASE_SIGN
    return None


def needs_flip(tags: dict, mode: str) -> bool:
    if mode == 'keep':
        return False
    if mode == 'flip':
        return True
    sign = file_phase_sign(tags)
    if sign is None:
        raise ValueError('phase sign unknown (no phase_sign/convention tag); use --sign keep or --sign flip')
    return sign == LEGACY_SIGN


def rgb_sign_after(tags: dict, flip: bool) -> str:
    """Sign convention of an RGB rendered from a file with `tags`, negated if `flip`."""
    src = file_phase_sign(tags)
    if src is None:
        return 'unknown (negated)' if flip else 'unknown'
    return {PHASE_SIGN: LEGACY_SIGN, LEGACY_SIGN: PHASE_SIGN}[src] if flip else src


def rgb_path_for(wrp_path: str) -> str:
    if not wrp_path.endswith('_wrp.tif'):
        raise ValueError(f'not a wrapped-phase file name: {wrp_path}')
    return wrp_path[:-len('_wrp.tif')] + '_rgb.tif'


def render_file(args) -> dict:
    """Pool worker: wrapped GeoTIFF -> RGB GeoTIFF. Returns a small status dict."""
    root, wrp_rel, mode = args
    import rasterio

    try:
        wrp_path = Path(root) / wrp_rel
        rgb_rel = rgb_path_for(wrp_rel)
        with rasterio.open(wrp_path) as ds:
            wrp = ds.read(1)
            tags = ds.tags()
            crs, transform = ds.crs, ds.transform
        flip = needs_flip(tags, mode)
        if flip:
            wrp = -wrp
        keep = {k: v for k, v in tags.items() if k not in ('units', 'quantity', 'AREA_OR_POINT')}
        keep.update(quantity='wrapped phase, false-colour RGB', colormap=RAMP_NAME,
                    rgb_phase_sign=rgb_sign_after(tags, flip),
                    rgb_source=Path(wrp_rel).name, rgb_flipped_from_source=str(flip))
        write_rgb(Path(root) / rgb_rel, phase_to_rgb(wrp), crs, transform, keep, tiled=wrp.shape[0] >= 512)
        return {'wrp': wrp_rel, 'rgb': rgb_rel, 'flipped': flip}
    except Exception as e:  # keep going; reported at the end
        return {'wrp': wrp_rel, 'error': f'{type(e).__name__}: {e}'}


def _manifests(root: Path, scenes: bool, chips: bool) -> list[Path]:
    out = []
    if scenes and (root / 'manifest.csv').exists():
        out.append(root / 'manifest.csv')
    if chips:
        out += sorted(root.glob('chips_*/chips.csv'))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m hrdps_synthetic.rgb', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=None,
                   help='folder holding manifest.csv and the samples (default: the workdir)')
    p.add_argument('--scenes', action=argparse.BooleanOptionalAction, default=True, help='render scene RGBs')
    p.add_argument('--chips', action=argparse.BooleanOptionalAction, default=True,
                   help='render chip RGBs (every chips_*/chips.csv)')
    p.add_argument('--sign', choices=SIGN_MODES, default='auto',
                   help='auto: render every file in the current convention (flip legacy files); '
                        'keep: render as stored; flip: always negate')
    p.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False,
                   help='re-render RGB files that already exist')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--limit', type=int, default=None, help='render at most N files this run')
    p.add_argument('--dry-run', action='store_true', help='report what would be rendered; write nothing')
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from hrdps_synthetic.config import parse_args, save_effective
    from hrdps_synthetic.download import setup_logging

    args = parse_args(build_parser(), argv, 'rgb')
    setup_logging(args.verbose)
    root = Path(args.output_dir or args.workdir)
    manifests = _manifests(root, args.scenes, args.chips)
    if not manifests:
        log.error('No manifest.csv or chips_*/chips.csv under %s', root)
        return 2

    jobs, per_manifest = [], {}
    for m in manifests:
        _, rows = read_rows(m)
        todo = [r['wrp_path'] for r in rows if r.get('wrp_path')]
        per_manifest[m] = (len(rows), todo)
        jobs += [w for w in todo if args.overwrite or not (root / rgb_path_for(w)).exists()]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    for m, (n, todo) in per_manifest.items():
        log.info('%s: %d rows, %d with wrapped phase', m.relative_to(root), n, len(todo))
    log.info('%d RGB files to render (sign mode: %s)', len(jobs), args.sign)

    if args.dry_run:
        if jobs:
            from rasterio.io import MemoryFile
            import rasterio

            with rasterio.open(root / jobs[0]) as ds:
                wrp, tags, crs, tr = ds.read(1), ds.tags(), ds.crs, ds.transform
            flip = needs_flip(tags, args.sign)
            t0 = time.time()
            rgb = phase_to_rgb(-wrp if flip else wrp)
            with MemoryFile() as mf:
                with mf.open(**rgb_profile(rgb.shape[2], rgb.shape[1], crs, tr, wrp.shape[0] >= 512)) as dst:
                    dst.write(rgb)
                nb = len(mf.read())
            print(f'Probe {jobs[0]}: {nb / 1e6:.2f} MB, {time.time() - t0:.1f} s'
                  f'{" (legacy sign, flipped)" if flip else ""}')
            print(f'Estimated: ~{nb * len(jobs) / 1e9:.1f} GB for {len(jobs)} files (sizes differ for chips and scenes)')
        print('Dry run: nothing was written.')
        return 0

    log.info('Settings saved to %s', save_effective(args, 'rgb', root, argv))
    t0, done, failed, flipped = time.time(), set(), [], 0
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(render_file, [(str(root), j, args.sign) for j in jobs],
                                                    chunksize=8), start=1):
            if 'error' in res:
                failed.append(res['wrp'])
                log.error('%s failed: %s', res['wrp'], res['error'])
                continue
            done.add(res['wrp'])
            flipped += res['flipped']
            if k % 500 == 0 or k == len(jobs):
                log.info('%d/%d rendered (%.1f min)', k, len(jobs), (time.time() - t0) / 60)

    for m in manifests:  # record rgb_path for every row whose RGB now exists
        fields, rows = read_rows(m)
        if 'rgb_path' not in fields:
            fields = fields + ['rgb_path']
        for r in rows:
            w = r.get('wrp_path')
            if w and (root / rgb_path_for(w)).exists():
                r['rgb_path'] = rgb_path_for(w)
        rewrite(m, fields, rows)
    log.info('Rendered %d RGB files (%d flipped from the legacy sign); %d failed', len(done), flipped, len(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
