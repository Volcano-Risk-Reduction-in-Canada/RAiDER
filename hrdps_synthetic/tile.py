"""Cut already-rendered scenes into chips (e.g. 500 x 500 px for Detectron2).

Reads the scenes listed in manifest.csv and writes chips exactly as
``generate --chip-size`` would, into chips_<W>x<H>[...]/ with a chips.csv
manifest:

* **Products.** Each chip gets unwrapped phase, wrapped phase and (by default)
  the false-colour RGB.
* **Metadata.** Each chip has its own location, incidence range and LOS vector.
* **Splits.** Each chip inherits its scene's train/validation split.

Scenes from older runs work too: legacy north-up manifests are read as they
are. Chip phase keeps the scene's sign, and the RGB follows ``--sign`` (see
rgb.py). Rerunning is safe: scenes already chipped with the same settings are
skipped.

Examples (from the repository root):

    python -m hrdps_synthetic.tile --workdir /data/hrdps_synth --chip-size 500 --dry-run
    python -m hrdps_synthetic.tile --workdir /data/hrdps_synth --chip-size 500 --workers 8
    python -m hrdps_synthetic.tile --workdir /data/hrdps_synth --chip-size 512 --chip-stride 256
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from hrdps_synthetic.chips import (EDGES, ChipManifest, ChipSpec, chip_dir, chip_nbytes_in_memory, chips_per_scene,
                                   make_spec, read_chipped, write_chips)
from hrdps_synthetic.geometry import RCM_ALTITUDE_M, SceneGrid
from hrdps_synthetic.rgb import RAMP_NAME, SIGN_MODES, file_phase_sign, needs_flip, phase_to_rgb, rgb_sign_after
from hrdps_synthetic.tables import read_rows

log = logging.getLogger('hrdps_synthetic')

_FLOATS = ('theta_c_deg', 'heading_deg', 'wavelength_m', 'x0', 'y0', 'spacing_m')


def read_scenes(root: Path) -> list[dict]:
    """Manifest rows of samples whose full scenes were written."""
    _, rows = read_rows(Path(root) / 'manifest.csv')
    rows = [r for r in rows if r.get('unw_path')]
    for r in rows:
        for k in _FLOATS:
            r[k] = float(r[k])
        r['altitude_m'] = float(r['altitude_m']) if r.get('altitude_m') else RCM_ALTITUDE_M
    return rows


def scene_of(row: dict) -> SceneGrid:
    """SceneGrid from a manifest row (current or legacy north-up manifests)."""
    nx, ny, s = int(row['nx']), int(row['ny']), float(row['spacing_m'])
    if row.get('geometry') == 'radar':
        return SceneGrid(int(row['epsg']), float(row['center_x']), float(row['center_y']), nx, ny, s, 'radar',
                         float(row['flight_az_grid_deg']))
    return SceneGrid(int(row['epsg']), row['x0'] + nx * s / 2, row['y0'] - ny * s / 2, nx, ny, s)


def load_scene(root: Path, row: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    import rasterio

    with rasterio.open(Path(root) / row['unw_path']) as u, rasterio.open(Path(root) / row['wrp_path']) as w:
        return u.read(1), w.read(1), w.tags()


def _tile_one(args) -> dict:
    root, row, spec_d, rgb_on, sign_mode = args
    try:
        unw, wrp, tags = load_scene(root, row)
        meta = dict(row, phase_sign=file_phase_sign(tags) or '')
        rgb, rgb_tags = None, {}
        if rgb_on:
            flip = needs_flip(tags, sign_mode)
            rgb = phase_to_rgb(-wrp if flip else wrp)
            rgb_tags = {'colormap': RAMP_NAME, 'rgb_phase_sign': rgb_sign_after(tags, flip),
                        'rgb_flipped_from_source': str(flip)}
        chips = write_chips(root, ChipSpec(**spec_d), scene_of(row), unw, wrp, meta, row['altitude_m'], rgb=rgb,
                            rgb_tags=rgb_tags,
                            convention=tags.get('convention', ''))
        return {'sample_id': row['sample_id'], 'chips': chips}
    except Exception as e:  # keep going; the scene stays pending
        return {'sample_id': row['sample_id'], 'error': f'{type(e).__name__}: {e}'}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m hrdps_synthetic.tile', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=None,
                   help='folder holding manifest.csv and the scenes (default: the workdir)')
    p.add_argument('--chip-size', type=int, nargs='+', required=True, metavar='PX', help='chip W [H] in pixels')
    p.add_argument('--chip-stride', type=int, nargs='+', default=None, metavar='PX',
                   help='chip step SX [SY] in pixels (default: chip size, no overlap)')
    p.add_argument('--chip-edge', choices=EDGES, default='drop',
                   help='remainder narrower than a chip: drop it (default) or add a chip flush with the edge')
    p.add_argument('--rgb', action=argparse.BooleanOptionalAction, default=True, help='also write RGB chips')
    p.add_argument('--sign', choices=SIGN_MODES, default='auto',
                   help='RGB sign: auto renders the current convention (flips legacy scenes); keep; flip')
    p.add_argument('--split', choices=('train', 'val'), default=None, help='only this split')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--limit', type=int, default=None, help='chip at most N scenes this run')
    p.add_argument('--dry-run', action='store_true', help='report chip counts and storage; write nothing')
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from hrdps_synthetic.config import parse_args, save_effective
    from hrdps_synthetic.download import setup_logging

    args = parse_args(build_parser(), argv, 'tile')
    setup_logging(args.verbose)
    root = Path(args.output_dir or args.workdir)
    try:
        spec = make_spec(args.chip_size, args.chip_stride, args.chip_edge)
    except ValueError as e:
        log.error('%s', e)
        return 2
    scenes = read_scenes(root)
    if args.split:
        scenes = [r for r in scenes if r['split'] == args.split]
    if not scenes:
        log.error('No rendered scenes in %s/manifest.csv (were they generated with --no-scenes?)', root)
        return 2
    sizes = {(int(r['ny']), int(r['nx'])) for r in scenes}
    for ny, nx in sizes:
        if spec.width > nx or spec.height > ny:
            log.error('Chip %d x %d px is larger than %d x %d px scenes', spec.width, spec.height, nx, ny)
            return 2
    done, _ = read_chipped(chip_dir(root, spec) / 'chips.csv')
    pending = [r for r in scenes if r['sample_id'] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    n_chips = sum(chips_per_scene(int(r['ny']), int(r['nx']), spec) for r in pending)
    log.info('%s: %d scenes, %d already chipped, %d to chip -> %d chips in %s', spec.describe(), len(scenes),
             len(done), len(pending), n_chips, chip_dir(root, spec))

    if args.dry_run:
        print(f'Chips: {spec.describe()}{" + RGB" if args.rgb else ""}; {n_chips} chips from {len(pending)} scenes '
              f'({", ".join(f"{nx} x {ny} px" for ny, nx in sorted(sizes))})')
        print(f'Uncompressed: {n_chips * (8 + (3 if args.rgb else 0)) * spec.width * spec.height / 1e9:.1f} GB')
        if pending:
            r = pending[0]
            unw, wrp, tags = load_scene(root, r)
            rgb = phase_to_rgb(-wrp if needs_flip(tags, args.sign) else wrp) if args.rgb else None
            t0 = time.time()
            nb = chip_nbytes_in_memory(spec, scene_of(r), unw, wrp, dict(r, phase_sign=''), rgb)
            secs = time.time() - t0
            per_scene = chips_per_scene(int(r['ny']), int(r['nx']), spec)
            print(f'Probe {r["sample_id"]}: {nb / 1e6:.1f} MB for its {per_scene} chips, {secs:.1f} s; '
                  f'estimated ~{nb * len(pending) / 1e9:.1f} GB on disk')
        print('Dry run: nothing was written.')
        return 0

    out = ChipManifest(root, spec)
    log.info('Settings saved to %s', save_effective(args, 'tile', root, argv))
    t0, n_ok, failed = time.time(), 0, []
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            jobs = [(str(root), r, asdict(spec), args.rgb, args.sign) for r in pending]
            for res in pool.imap_unordered(_tile_one, jobs, chunksize=4):
                if 'error' in res:
                    failed.append(res['sample_id'])
                    log.error('%s failed: %s', res['sample_id'], res['error'])
                    continue
                out.write(res['chips'])
                n_ok += 1
                if n_ok % 100 == 0 or n_ok == len(pending):
                    log.info('%d/%d scenes chipped (%.1f min)', n_ok, len(pending), (time.time() - t0) / 60)
    finally:
        out.close()
    log.info('Chipped %d scenes; %d failed; manifest %s', n_ok, len(failed), out.path)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
