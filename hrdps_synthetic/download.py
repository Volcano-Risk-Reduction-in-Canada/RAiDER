"""Backfill HRDPS from the MSC Datamart into per-AOI epoch delay cubes.

For every archived day and pass direction this builds, for each AOI, one epoch
file holding RAiDER's zenith wet and hydrostatic delay cubes, time-interpolated
to the mean acquisition time (ascending 02:02 UTC, descending 14:43 UTC by
default, from the SLC records in the VRRC API dump):

    <workdir>/epochs/<AOI>/<asc|desc>/HRDPS_<AOI>_<YYYYMMDDTHHMM>.nc

Steps per valid hour: download the 84 GRIBs (TMP, SPFH, HGT on 28 levels) once,
regrid them for every AOI, run RAiDER's prepareWeatherModel per AOI, then blend
the two hours around the target time with RAiDER's combine_weather_files.
GRIBs and intermediate files are deleted unless --keep-grib / --keep-hourly.

The Datamart keeps the latest ~30 days, so one run collects everything
available. Rerunning is safe: finished epochs are skipped.

Examples (from the repository root):

    python -m hrdps_synthetic.download --workdir /data/hrdps_synth --dry-run
    python -m hrdps_synthetic.download --workdir /data/hrdps_synth --workers 8
    python -m hrdps_synthetic.download --workdir /data/hrdps_synth --days 20261001 --aois A4201_Volcano_Meager
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import multiprocessing
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import requests

from hrdps_synthetic import hrdps_archive as ha
from hrdps_synthetic.aois import DEFAULT_API_DUMP, AOI, load_aois, select_aois

log = logging.getLogger('hrdps_synthetic')
_SPAWN = multiprocessing.get_context('spawn')  # download threads are running in the parent; avoid fork

DEFAULT_TIMES = {'asc': '02:02', 'desc': '14:43'}


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    # RAiDER logs every step to stdout and propagates to the root logger; keep its
    # debug.log/error.log files, show only its warnings on the console.
    raider = logging.getLogger('RAiDER')
    raider.propagate = verbose
    for h in raider.handlers:
        if type(h) is logging.StreamHandler:
            h.setLevel(logging.DEBUG if verbose else logging.WARNING)


def parse_hhmm(s: str) -> dt.time:
    return dt.datetime.strptime(s, '%H:%M').time()


def parse_days(spec: Optional[Sequence[str]]) -> Optional[list[dt.date]]:
    """'YYYYMMDD' items or 'YYYYMMDD:YYYYMMDD' inclusive ranges."""
    if not spec:
        return None
    out = set()
    for item in spec:
        for part in item.split(','):
            if ':' in part:
                a, b = (dt.datetime.strptime(x, '%Y%m%d').date() for x in part.split(':'))
                out.update(a + dt.timedelta(days=k) for k in range((b - a).days + 1))
            elif part:
                out.add(dt.datetime.strptime(part, '%Y%m%d').date())
    return sorted(out)


def wm_dir(workdir: Path, aoi: AOI) -> Path:
    return Path(workdir) / 'weather' / aoi.name


def pending_tasks(workdir: Path, aois: Sequence[AOI], days: Sequence[dt.date],
                  times: dict[str, dt.time]) -> list[tuple[dt.date, str, dt.datetime, list[AOI]]]:
    tasks = []
    for day in days:
        for direction, t in times.items():
            target = dt.datetime.combine(day, t)
            missing = [a for a in aois if not ha.epoch_path(workdir, a.name, direction, target).exists()]
            if missing:
                tasks.append((day, direction, target, missing))
    return tasks


def _process_hour_job(args):
    wmd, valid, bounds = args
    return ha.process_hour(Path(wmd), valid, bounds)


def _epoch_job(args):
    name, direction, target, hour_files, out_path = args
    return ha.build_epoch(name, direction, target, hour_files, Path(out_path))


def _map(pool, fn, items):
    return list(pool.map(fn, items)) if pool else [fn(x) for x in items]


def run_task(workdir: Path, task, grib_dir: Path, pool, threads: int, keep_grib: bool,
             keep_hourly: bool, grib_only: bool, size: dict) -> int:
    day, direction, target, aois = task
    hours = ha.bracketing_hours(target)
    hour_files: dict[str, list] = {a.name: [] for a in aois}
    for valid, weight in hours:
        raws = {a.name: ha.raw_path(wm_dir(workdir, a), valid) for a in aois}
        need = [a for a in aois if not raws[a.name].exists()]
        if need or grib_only:
            t0 = time.time()
            gribs = ha.fetch_hour(valid, grib_dir, threads=threads)
            log.info('%s %s: %d GRIBs for %s ready (%.0f s)', day, direction, len(gribs), f'{valid:%H}Z', time.time() - t0)
            if grib_only:
                continue
            grid, _, _ = ha.read_grib(gribs[0][2])
            targets = {a.name: ha.make_target_grid(a.raw_bounds(**size), grid) for a in need}
            t0 = time.time()
            ha.build_raw_hour(valid, gribs, targets, {a.name: raws[a.name] for a in need}, executor=pool)
            log.info('%s %s: regridded %s for %d AOIs (%.0f s)', day, direction, f'{valid:%H}Z', len(need), time.time() - t0)
            if not keep_grib:
                for _, _, p in gribs:
                    p.unlink(missing_ok=True)
                try:
                    gribs[0][2].parent.rmdir()
                except OSError:
                    pass
        t0 = time.time()
        jobs = [(str(wm_dir(workdir, a)), valid, a.weather_bounds(**size)) for a in aois]
        processed = _map(pool, _process_hour_job, jobs)
        for a, p in zip(aois, processed):
            hour_files[a.name].append((valid, weight, p))
        log.info('%s %s: RAiDER processed %s for %d AOIs (%.0f s)', day, direction, f'{valid:%H}Z', len(aois), time.time() - t0)
    if grib_only:
        return 0

    jobs = [(a.name, direction, target, hour_files[a.name], str(ha.epoch_path(workdir, a.name, direction, target)))
            for a in aois]
    done = _map(pool, _epoch_job, jobs)

    if not keep_hourly:
        for a in aois:
            for valid, _, p in hour_files[a.name]:
                Path(p).unlink(missing_ok=True)
                ha.raw_path(wm_dir(workdir, a), valid).unlink(missing_ok=True)
    log.info('%s %s: wrote %d epoch files for %s UTC', day, direction, len(done), f'{target:%H:%M}')
    return len(done)


def dry_run(workdir: Path, tasks, aois: Sequence[AOI], size: dict) -> None:
    session = requests.Session()
    total_bytes, n_files, n_missing, peak = 0, 0, 0, 0
    hours_seen = {}
    print(f'{"day":10s} {"dir":4s} {"target":>6s} {"AOIs":>5s} {"hours":>12s} {"GRIB MB":>8s} {"unpublished":>12s}')
    for day, direction, target, missing in tasks:
        task_bytes, task_missing, labels = 0, 0, []
        for valid, _ in ha.bracketing_hours(target):
            run, fxx = ha.run_and_step(valid)
            labels.append(f'{valid:%H}Z')
            key = (run, fxx)
            if key not in hours_seen:
                sizes = ha.listing_sizes(run, fxx, session)
                need = [f for _, _, f in ha.required_files(run, fxx)]
                b = sum(sizes.get(f, 0) for f in need)
                miss = sum(f not in sizes for f in need)
                hours_seen[key] = (b, miss)
            b, miss = hours_seen[key]
            task_bytes += b
            task_missing += miss
            peak = max(peak, b)
            n_files += 84
        total_bytes += task_bytes
        n_missing += task_missing
        print(f'{day.isoformat():10s} {direction:4s} {target:%H:%M} {len(missing):5d} {"+".join(labels):>12s} '
              f'{task_bytes / 1e6:8.0f} {task_missing:12d}')
    n_epochs = sum(len(t[3]) for t in tasks)
    epoch_bytes = sum(ha.epoch_nbytes(a, **size) for t in tasks for a in t[3])
    raw_peak = sum(ha.epoch_nbytes(a, **size) for a in aois) * 4  # raw + processed hourly files per hour, rough
    print()
    print(f'Tasks (day x direction): {len(tasks)}; epoch files to write: {n_epochs}')
    print(f'GRIB downloads: {n_files} files, ~{total_bytes / 1e9:.1f} GB (sizes from Datamart listings, rounded)')
    if n_missing:
        print(f'Not yet published or already purged: {n_missing} files (those days are skipped and retried next run)')
    print(f'Peak transient GRIB disk: ~{peak / 1e6:.0f} MB per valid hour (deleted after regridding)')
    print(f'Peak transient raw/processed hourly files: ~{raw_peak / 1e6:.0f} MB')
    print(f'Final epoch storage: <= {epoch_bytes / 1e9:.2f} GB uncompressed (written zlib-compressed)')
    print('Dry run: nothing was downloaded or written.')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m hrdps_synthetic.download', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--aoi-file', type=Path, default=DEFAULT_API_DUMP)
    p.add_argument('--aois', nargs='+', default=None, help='AOI names (default: all 45)')
    p.add_argument('--days', nargs='+', default=None,
                   help='YYYYMMDD or YYYYMMDD:YYYYMMDD (default: every day on the Datamart)')
    p.add_argument('--asc-time', default=DEFAULT_TIMES['asc'], help='ascending target time, HH:MM UTC')
    p.add_argument('--desc-time', default=DEFAULT_TIMES['desc'], help='descending target time, HH:MM UTC')
    p.add_argument('--directions', nargs='+', choices=('asc', 'desc'), default=['asc', 'desc'])
    p.add_argument('--width-km', type=float, default=20.0,
                   help='largest scene width (E-W) you will render; sizes the weather crops (default 20)')
    p.add_argument('--length-km', type=float, default=30.0,
                   help='largest scene length (N-S) you will render; sizes the weather crops (default 30)')
    p.add_argument('--workers', type=int, default=4, help='processes for GRIB decoding and RAiDER processing')
    p.add_argument('--threads', type=int, default=8, help='parallel GRIB downloads')
    p.add_argument('--grib-dir', type=Path, default=None, help='GRIB cache (default <workdir>/grib)')
    p.add_argument('--grib-only', action=argparse.BooleanOptionalAction, default=False,
                   help='only mirror the needed GRIBs; no processing')
    p.add_argument('--keep-grib', action=argparse.BooleanOptionalAction, default=False,
                   help='keep downloaded GRIBs')
    p.add_argument('--keep-hourly', action=argparse.BooleanOptionalAction, default=False,
                   help='keep regridded raw and processed hourly files')
    p.add_argument('--dry-run', action='store_true', help='report downloads and storage; write nothing')
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    from hrdps_synthetic.config import parse_args, save_effective

    args = parse_args(build_parser(), argv, 'download')
    setup_logging(args.verbose)

    aois = select_aois(load_aois(args.aoi_file), args.aois)
    available = ha.list_available_days()
    days = parse_days(args.days) or available
    gone = [d for d in days if d not in available]
    if gone:
        log.warning('Not on the Datamart (purged or not yet published): %s', ', '.join(map(str, gone)))
        days = [d for d in days if d in available]
    times = {'asc': parse_hhmm(args.asc_time), 'desc': parse_hhmm(args.desc_time)}
    times = {k: v for k, v in times.items() if k in args.directions}
    tasks = pending_tasks(args.workdir, aois, days, times)
    log.info('%d AOIs, %d days (%s to %s), %d day/direction tasks pending', len(aois), len(days),
             days[0] if days else '-', days[-1] if days else '-', len(tasks))

    size = {'width': args.width_km * 1e3, 'length': args.length_km * 1e3}
    if args.dry_run:
        dry_run(args.workdir, tasks, aois, size)
        return 0

    args.workdir.mkdir(parents=True, exist_ok=True)
    log.info('Settings saved to %s', save_effective(args, 'download', args.workdir, argv))
    grib_dir = args.grib_dir or args.workdir / 'grib'
    from RAiDER.models.customExceptions import NoWeatherModelData

    n_done, failed = 0, []
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=_SPAWN) if args.workers > 1 else None
    try:
        for task in tasks:
            try:
                n_done += run_task(args.workdir, task, grib_dir, pool, args.threads, args.keep_grib,
                                   args.keep_hourly, args.grib_only, size)
            except NoWeatherModelData as e:
                log.warning('%s %s skipped, data not on the Datamart yet: %s', task[0], task[1], e)
                failed.append(task[:2])
            except Exception:
                log.exception('%s %s failed', task[0], task[1])
                failed.append(task[:2])
    finally:
        if pool:
            pool.shutdown()
    if not args.keep_grib and grib_dir.exists() and not any(grib_dir.rglob('*.grib2')):
        shutil.rmtree(grib_dir, ignore_errors=True)
    log.info('Finished: %d epoch files written; %d task(s) failed%s', n_done, len(failed),
             (': ' + ', '.join(f'{d} {s}' for d, s in failed)) if failed else '')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
