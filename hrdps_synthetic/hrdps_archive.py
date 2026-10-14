"""HRDPS archive access for the synthetic generator.

This module is a separate reader path. It does not change RAiDER's HRDPS model;
it only imports from it:

* GRIBs are downloaded once per valid hour with ``RAiDER.models.hrdps._fetch_grib``.
* Each GRIB is decoded once (eccodes) and regridded for every AOI from the
  rotated lat/lon grid onto a **true regular lat/lon grid** (bilinear in
  rotated-pole index space). RAiDER's own reader treats the rotated grid as
  regular, which misplaces data by kilometres at the edges of wide crops.
* The 1015 hPa level, which the Datamart publishes but RAiDER's level list
  omits, is added so coastal pixels need less extrapolation.
* Regridded files use exactly the schema ``RAiDER.models.hrdps.load_weather_hrdps``
  reads, so ``RAiDER.processWM.prepareWeatherModel`` and
  ``RAiDER.cli.raider.combine_weather_files`` are reused unchanged.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import requests
import xarray as xr
from pyproj import CRS
from scipy.ndimage import map_coordinates

from RAiDER.models.hrdps import HRDPS_PRESSURE_LEVELS_HPA, _MSC_FNAME, _fetch_grib

log = logging.getLogger('hrdps_synthetic')

DATAMART_ROOT = 'https://dd.weather.gc.ca'
PRODUCT = 'continental/2.5km'
GRIB_VARIABLES = {'TMP': 't', 'SPFH': 'q', 'HGT': 'z'}
LEVELS_HPA = (1015,) + tuple(HRDPS_PRESSURE_LEVELS_HPA)  # surface -> top, as RAiDER expects
TARGET_DLAT = 0.0225  # regular-grid latitude spacing (~2.5 km); longitude spacing scales with 1/cos(lat)

# Fixed height grid for every processed file, so hours and dates share identical
# x/y/z and can be blended and differenced exactly. Starts at RAiDER's _ZMIN
# (-100 m) so RAiDER does not pad or trim the grid; zero refractivity is filled
# above the model top by RAiDER's fillna3D.
FIXED_Z = np.concatenate([
    np.arange(-100.0, 3000.0, 50.0),
    np.arange(3000.0, 6000.0, 100.0),
    np.arange(6000.0, 12000.0, 250.0),
    np.arange(12000.0, 20500.0 + 1, 500.0),
])

HTTP_TIMEOUT = 60
NETWORK_RETRIES = 5  # on top of the 3 quick retries inside RAiDER's _fetch_grib
NETWORK_BACKOFF_S = 30


# ---------------------------------------------------------------------------
# Times
# ---------------------------------------------------------------------------
def run_and_step(valid: dt.datetime) -> tuple[dt.datetime, int]:
    """HRDPS run and forecast hour for a valid hour, as RAiDER's HRDPS._fetch maps them."""
    run = valid.replace(hour=(valid.hour // 6) * 6, minute=0, second=0, microsecond=0)
    return run, int((valid - run).total_seconds() // 3600)


def bracketing_hours(target: dt.datetime) -> list[tuple[dt.datetime, float]]:
    """Hourly valid times around `target` with linear time-interpolation weights."""
    h0 = target.replace(minute=0, second=0, microsecond=0)
    frac = (target - h0).total_seconds() / 3600.0
    if frac == 0:
        return [(h0, 1.0)]
    return [(h0, 1.0 - frac), (h0 + dt.timedelta(hours=1), frac)]


# ---------------------------------------------------------------------------
# Datamart listings
# ---------------------------------------------------------------------------
def hour_dir_url(run: dt.datetime, fxx: int) -> str:
    return f'{DATAMART_ROOT}/{run:%Y%m%d}/WXO-DD/model_hrdps/{PRODUCT}/{run:%H}/{fxx:03d}/'


def required_files(run: dt.datetime, fxx: int) -> list[tuple[str, str, str]]:
    """(variable, level string, file name) for every GRIB needed at one valid hour."""
    out = []
    for lvl in LEVELS_HPA:
        level = f'ISBL_{lvl:04d}'
        for var in GRIB_VARIABLES:
            out.append((var, level, _MSC_FNAME.format(date=run, variable=var, level=level, fxx=fxx)))
    return out


def list_available_days(session: Optional[requests.Session] = None) -> list[dt.date]:
    """Day folders currently on the MSC Datamart rolling archive."""
    s = session or requests
    r = s.get(DATAMART_ROOT + '/', timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    days = sorted({dt.datetime.strptime(d, '%Y%m%d').date() for d in re.findall(r'href="(\d{8})/"', r.text)})
    return days


_UNITS = {'': 1, 'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3}
_LINE_RE = re.compile(r'href="([^"?/]+\.grib2)".*?(\d+(?:\.\d+)?)([KMG]?)\s*$')


def listing_sizes(run: dt.datetime, fxx: int, session: Optional[requests.Session] = None) -> dict[str, int]:
    """File sizes (approximate bytes, as rounded by the Apache index) in one hour folder."""
    s = session or requests
    r = s.get(hour_dir_url(run, fxx), timeout=HTTP_TIMEOUT)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    out = {}
    for line in r.text.splitlines():
        m = _LINE_RE.search(line)
        if m:
            out[m.group(1)] = int(float(m.group(2)) * _UNITS[m.group(3)])
    return out


# ---------------------------------------------------------------------------
# GRIB download and decoding
# ---------------------------------------------------------------------------
def _grib_complete(path: Path) -> bool:
    """A GRIB2 message ends with '7777'; a truncated download does not."""
    try:
        if path.stat().st_size < 8:
            return False
        with open(path, 'rb') as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            return head == b'GRIB' and f.read(4) == b'7777'
    except OSError:
        return False


def fetch_hour(valid: dt.datetime, grib_dir: Path, threads: int = 8) -> list[tuple[str, str, Path]]:
    """Download (or reuse) every GRIB for one valid hour. Returns (variable, level, path)."""
    run, fxx = run_and_step(valid)
    save_dir = Path(grib_dir) / f'{run:%Y%m%d}'
    save_dir.mkdir(parents=True, exist_ok=True)
    files = required_files(run, fxx)
    for _, _, fname in files:  # drop truncated leftovers so _fetch_grib re-downloads them
        p = save_dir / fname
        if p.exists() and not _grib_complete(p):
            p.unlink()

    def one(item):
        var, level, fname = item
        for attempt in range(1, NETWORK_RETRIES + 1):
            try:
                path = _fetch_grib(run, var, level, fxx, PRODUCT, save_dir, overwrite=attempt > 1)
                if _grib_complete(path):
                    return var, level, path
                path.unlink(missing_ok=True)
                err: Exception = IOError(f'Incomplete GRIB: {path}')
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, IOError) as e:
                if 'not found on MSC Datamart' in str(e):
                    raise
                err = e
            if attempt < NETWORK_RETRIES:
                wait = NETWORK_BACKOFF_S * attempt
                log.warning('%s: %s; retrying in %d s (%d/%d)', fname, type(err).__name__, wait, attempt, NETWORK_RETRIES)
                time.sleep(wait)
        raise err

    with ThreadPoolExecutor(max_workers=threads) as ex:
        return list(ex.map(one, files))


@dataclass(frozen=True)
class RotatedGrid:
    """HRDPS rotated lat/lon grid (GRIB2 template 3.1, south-pole convention)."""
    ni: int
    nj: int
    rlat0: float
    rlon0: float  # wrapped to [-180, 180)
    dlat: float
    dlon: float
    sp_lat: float
    sp_lon: float

    @classmethod
    def from_handle(cls, h) -> 'RotatedGrid':
        import eccodes

        g = lambda k: eccodes.codes_get(h, k)  # noqa: E731
        if g('gridType') != 'rotated_ll':
            raise ValueError(f'Expected rotated_ll grid, got {g("gridType")}')
        if abs(g('angleOfRotationInDegrees')) > 1e-9:
            raise ValueError('Non-zero angle of rotation is not supported')
        if g('iScansNegatively') != 0 or g('jScansPositively') != 1 or g('jPointsAreConsecutive') != 0:
            raise ValueError('Unexpected GRIB scanning mode')
        rlon0 = (g('longitudeOfFirstGridPointInDegrees') + 180.0) % 360.0 - 180.0
        return cls(ni=g('Ni'), nj=g('Nj'), rlat0=g('latitudeOfFirstGridPointInDegrees'), rlon0=rlon0,
                   dlat=g('jDirectionIncrementInDegrees'), dlon=g('iDirectionIncrementInDegrees'),
                   sp_lat=g('latitudeOfSouthernPoleInDegrees'), sp_lon=g('longitudeOfSouthernPoleInDegrees'))

    def geo_to_rot(self, lat, lon):
        lat, lon = np.radians(lat), np.radians(np.asarray(lon) - self.sp_lon)
        th = math.radians(90.0 + self.sp_lat)
        x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
        xr_ = math.cos(th) * x + math.sin(th) * z
        zr = -math.sin(th) * x + math.cos(th) * z
        return np.degrees(np.arcsin(np.clip(zr, -1, 1))), np.degrees(np.arctan2(y, xr_))

    def rot_to_geo(self, rlat, rlon):
        rlat, rlon = np.radians(rlat), np.radians(rlon)
        th = math.radians(90.0 + self.sp_lat)
        xr_, y, zr = np.cos(rlat) * np.cos(rlon), np.cos(rlat) * np.sin(rlon), np.sin(rlat)
        x = math.cos(th) * xr_ - math.sin(th) * zr
        z = math.sin(th) * xr_ + math.cos(th) * zr
        lon = (np.degrees(np.arctan2(y, x)) + self.sp_lon + 180.0) % 360.0 - 180.0
        return np.degrees(np.arcsin(np.clip(z, -1, 1))), lon

    def geo_to_index(self, lat, lon):
        """Fractional (row j, column i) of geographic points in the GRIB value array."""
        rlat, rlon = self.geo_to_rot(lat, lon)
        rlon = (rlon - self.rlon0) % 360.0 + self.rlon0  # unwrap relative to the first column
        return (rlat - self.rlat0) / self.dlat, (rlon - self.rlon0) / self.dlon


def read_grib(path: Path) -> tuple[RotatedGrid, np.ndarray, dict]:
    """Decode a single-message HRDPS GRIB2 file into (grid, values[nj, ni] float32, metadata)."""
    import eccodes

    with open(path, 'rb') as f:
        h = eccodes.codes_grib_new_from_file(f)
        try:
            grid = RotatedGrid.from_handle(h)
            vals = eccodes.codes_get_values(h).astype(np.float32).reshape(grid.nj, grid.ni)
            meta = {k: eccodes.codes_get(h, k) for k in ('shortName', 'level', 'units', 'dataDate', 'dataTime', 'stepRange')}
        finally:
            eccodes.codes_release(h)
    return grid, vals, meta


@dataclass
class TargetGrid:
    """Regular lat/lon grid of one AOI and its fractional indices in the HRDPS grid."""
    lats: np.ndarray  # (ny,) ascending
    lons: np.ndarray  # (nx,) ascending
    jj: np.ndarray  # (ny, nx)
    ii: np.ndarray


def regular_axes(bounds_snwe: Sequence[float], dlat: float = TARGET_DLAT) -> tuple[np.ndarray, np.ndarray]:
    S, N, W, E = bounds_snwe
    dlon = round(dlat / math.cos(math.radians(0.5 * (S + N))), 6)
    lats = S + dlat * np.arange(int(math.ceil((N - S) / dlat)) + 1)
    lons = W + dlon * np.arange(int(math.ceil((E - W) / dlon)) + 1)
    return np.round(lats, 6), np.round(lons, 6)


def make_target_grid(bounds_snwe: Sequence[float], grid: RotatedGrid, dlat: float = TARGET_DLAT) -> TargetGrid:
    lats, lons = regular_axes(bounds_snwe, dlat)
    LON, LAT = np.meshgrid(lons, lats)
    jj, ii = grid.geo_to_index(LAT, LON)
    if jj.min() < 0 or ii.min() < 0 or jj.max() > grid.nj - 1 or ii.max() > grid.ni - 1:
        raise ValueError(f'Crop {bounds_snwe} extends beyond the HRDPS domain')
    return TargetGrid(lats, lons, jj.astype(np.float64), ii.astype(np.float64))


def regrid(values: np.ndarray, target: TargetGrid) -> np.ndarray:
    return map_coordinates(values, [target.jj, target.ii], order=1, mode='nearest', prefilter=False).astype(np.float32)


def _decode_and_regrid(args) -> tuple[str, int, dict]:
    """Process-pool worker: decode one GRIB and regrid it for every AOI."""
    var, level_hpa, path, targets = args
    _, vals, _ = read_grib(path)
    return var, level_hpa, {name: regrid(vals, tg) for name, tg in targets.items()}


def write_raw(path: Path, valid: dt.datetime, t: np.ndarray, q: np.ndarray, z: np.ndarray,
              target: TargetGrid, source: str) -> Path:
    """Write a regridded hour in the schema RAiDER's load_weather_hrdps reads."""
    levels = np.asarray(LEVELS_HPA, dtype=float)
    pres = np.broadcast_to((levels * 100.0)[:, None, None], t.shape).astype(np.float32)
    LON, LAT = np.meshgrid(target.lons, target.lats)
    ds = xr.Dataset(
        {
            't': (['levels', 'y', 'x'], t),
            'q': (['levels', 'y', 'x'], q),
            'z': (['levels', 'y', 'x'], z),
            'pres': (['levels', 'y', 'x'], pres),
            'latitude': (['y', 'x'], LAT),
            'longitude': (['y', 'x'], LON),
        },
        coords={'levels': levels, 'y': target.lats, 'x': target.lons},
        attrs={
            'title': 'HRDPS regridded to a regular lat/lon grid (hrdps_synthetic)',
            'valid_time': valid.isoformat(),
            'source': source,
            'regrid': 'bilinear in rotated-pole index space from the GRIB grid definition',
        },
    )
    ds['proj'] = 0
    for k, v in CRS.from_epsg(4326).to_cf().items():
        ds.proj.attrs[k] = v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    ds.to_netcdf(tmp, engine='netcdf4')
    tmp.replace(path)
    return path


def build_raw_hour(valid: dt.datetime, gribs: Sequence[tuple[str, str, Path]], targets: dict[str, TargetGrid],
                   out_paths: dict[str, Path], executor=None) -> dict[str, Path]:
    """Decode each GRIB once, regrid for every AOI, and write one raw file per AOI.

    `executor` is an optional concurrent.futures executor used to decode GRIBs in parallel.
    """
    jobs = [(var, int(level.split('_')[1]), path, targets) for var, level, path in gribs]
    results = list(executor.map(_decode_and_regrid, jobs)) if executor else [_decode_and_regrid(j) for j in jobs]

    lev_index = {lvl: k for k, lvl in enumerate(LEVELS_HPA)}
    run, fxx = run_and_step(valid)
    out = {}
    for name, tg in targets.items():
        arrays = {v: np.full((len(LEVELS_HPA), tg.lats.size, tg.lons.size), np.nan, np.float32) for v in GRIB_VARIABLES}
        for var, lvl, per_aoi in results:
            arrays[var][lev_index[lvl]] = per_aoi[name]
        for v, a in arrays.items():
            if np.isnan(a).any():
                raise RuntimeError(f'Missing {v} levels for {name} at {valid}')
        out[name] = write_raw(out_paths[name], valid, arrays['TMP'], arrays['SPFH'], arrays['HGT'], tg,
                              source=f'MSC Datamart HRDPS {PRODUCT} run {run:%Y-%m-%dT%HZ} +{fxx:03d}h')
    return out


# ---------------------------------------------------------------------------
# RAiDER processing (imported, unchanged)
# ---------------------------------------------------------------------------
def raw_path(wm_dir: Path, valid: dt.datetime) -> Path:
    from RAiDER.models.weatherModel import make_raw_weather_data_filename

    return Path(make_raw_weather_data_filename(str(wm_dir), 'HRDPS', valid))


def process_hour(wm_dir: Path, valid: dt.datetime, wm_bounds: Sequence[float]) -> Path:
    """Process an existing regridded raw file with RAiDER's HRDPS weather model.

    This runs the same WeatherModel calls as ``RAiDER.processWM.prepareWeatherModel``
    after its download decision (set bounds and time, ``load``, ``write``).
    ``prepareWeatherModel`` itself is not used because its raw-file containment
    check (``checkContainment_raw``) returns False for any regional file, which
    would trigger RAiDER's own HRDPS download and overwrite the regridded file.
    """
    from RAiDER.models.hrdps import HRDPS

    raw = raw_path(wm_dir, valid)
    if not raw.exists():
        raise FileNotFoundError(raw)
    S, N, W, E = wm_bounds
    with xr.open_dataset(raw) as ds:
        if not (ds.y.min() <= S and ds.y.max() >= N and ds.x.min() <= W and ds.x.max() >= E):
            raise ValueError(f'{raw} does not cover {wm_bounds}')
    wm = HRDPS()
    wm.set_wmLoc(str(wm_dir))
    wm._zlevels = FIXED_Z.copy()  # instance attribute read by WeatherModel._uniform_in_z
    wm.set_latlon_bounds(np.asarray(wm_bounds, dtype=float))
    wm.setTime(valid)
    f = wm.load()  # returns the processed path if it already exists
    if f is None:
        f = wm.write()
    return Path(f)


def build_epoch(aoi_name: str, direction: str, target: dt.datetime,
                hour_files: Sequence[tuple[dt.datetime, float, Path]], out_path: Path) -> Path:
    """Time-interpolate processed hourly files to `target` and keep only the zenith-delay cubes."""
    from RAiDER.cli.raider import combine_weather_files

    paths = [Path(p) for _, _, p in hour_files]
    with xr.open_dataset(paths[0]) as d0:
        x0, y0, z0 = d0.x.values, d0.y.values, d0.z.values
    for p in paths[1:]:
        with xr.open_dataset(p) as d:
            if not (np.array_equal(d.x.values, x0) and np.array_equal(d.y.values, y0) and np.array_equal(d.z.values, z0)):
                raise RuntimeError(f'Hourly grids differ, cannot blend exactly: {paths}')

    combined = paths[0] if len(paths) == 1 else Path(
        combine_weather_files(paths, target.replace(tzinfo=None), 'HRDPS', interp_method='center_time'))
    with xr.open_dataset(combined) as ds:
        wet = ds['wet_total'].values.astype(np.float32)
        hydro = ds['hydro_total'].values.astype(np.float32)
        x, y, z = ds.x.values, ds.y.values, ds.z.values
    out = xr.Dataset(
        {'wet_total': (('z', 'y', 'x'), wet, {'units': 'm', 'description': 'zenith wet delay from z to model top'}),
         'hydro_total': (('z', 'y', 'x'), hydro, {'units': 'm', 'description': 'zenith hydrostatic delay from z to model top'})},
        coords={'z': ('z', z, {'units': 'm'}), 'y': ('y', y, {'units': 'degrees_north'}),
                'x': ('x', x, {'units': 'degrees_east'})},
        attrs={
            'aoi': aoi_name,
            'direction': direction,
            'target_time_utc': target.replace(tzinfo=None).isoformat(),
            'sources': '; '.join(f'{v:%Y-%m-%dT%H:%M}Z w={w:.4f}' for v, w, _ in hour_files),
            'processing': 'hrdps_synthetic regrid -> RAiDER HRDPS WeatherModel load+write -> RAiDER combine_weather_files (center_time)',
            'created_utc': dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds'),
        },
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + '.part')
    enc = {v: {'zlib': True, 'complevel': 4} for v in ('wet_total', 'hydro_total')}
    out.to_netcdf(tmp, encoding=enc)
    tmp.replace(out_path)
    if combined not in paths:
        combined.unlink(missing_ok=True)
    return out_path


def epoch_path(workdir: Path, aoi_name: str, direction: str, target: dt.datetime) -> Path:
    return Path(workdir) / 'epochs' / aoi_name / direction / f'HRDPS_{aoi_name}_{target:%Y%m%dT%H%M}.nc'


_EPOCH_RE = re.compile(r'HRDPS_(?P<aoi>.+)_(?P<ts>\d{8}T\d{4})\.nc$')


def index_epochs(workdir: Path) -> dict[tuple[str, str], dict[dt.date, Path]]:
    """{(aoi, direction): {date: path relative to workdir}} for every epoch file on disk."""
    root = Path(workdir) / 'epochs'
    out: dict = {}
    if not root.exists():
        return out
    for p in sorted(root.glob('*/*/HRDPS_*.nc')):
        m = _EPOCH_RE.match(p.name)
        if not m:
            continue
        aoi, direction = p.parent.parent.name, p.parent.name
        when = dt.datetime.strptime(m['ts'], '%Y%m%dT%H%M').date()
        out.setdefault((aoi, direction), {})[when] = p.relative_to(workdir)
    return out


def epoch_nbytes(aoi, n_z: int = FIXED_Z.size) -> int:
    """Uncompressed size of one epoch file (two float32 cubes)."""
    lats, lons = regular_axes(aoi.raw_bounds())
    return 2 * 4 * n_z * lats.size * lons.size
