import datetime as dt
from pathlib import Path

import cfgrib
import numpy as np
import requests
import xarray as xr
from pyproj import CRS
from shapely.geometry import Polygon, box

from RAiDER.logger import logger
from RAiDER.models.customExceptions import NoWeatherModelData
from RAiDER.models.hrrr import get_bounds_indices
from RAiDER.models.model_levels import LEVELS_50_HEIGHTS
from RAiDER.models.weatherModel import TIME_RES, WeatherModel
from RAiDER.utilFcns import round_date


# HRDPS Continental domain (approximate; ~41–87°N, ~148–40°W)
HRDPS_COVERAGE_POLYGON = Polygon(((-148, 41), (-148, 87), (-40, 87), (-40, 41)))

# Pressure levels available in HRDPS Continental (hPa), ordered surface→top
HRDPS_PRESSURE_LEVELS_HPA = [
    1000, 985, 970, 950, 925, 900, 875, 850, 800, 750, 700, 650, 600,
    550, 500, 450, 400, 350, 300, 275, 250, 225, 200, 175, 150, 100, 50,
]

# MSC Datamart base URL (date-partitioned layout introduced ~2025)
_MSC_BASE = 'https://dd.weather.gc.ca/{date:%Y%m%d}/WXO-DD/model_hrdps/{product}/{date:%H}/{fxx:03d}'
_MSC_FNAME = '{date:%Y%m%dT%HZ}_MSC_HRDPS_{variable}_{level}_RLatLon0.0225_PT{fxx:03d}H.grib2'


_DOWNLOAD_TIMEOUT = 180      # seconds per request attempt
_DOWNLOAD_RETRIES = 3        # total attempts before giving up
_DOWNLOAD_RETRY_WAIT = 10    # seconds between retries


def _fetch_grib(date: dt.datetime, variable: str, level: str, fxx: int,
                product: str, save_dir: Path, overwrite: bool = False) -> Path:
    """Download one HRDPS GRIB2 file from MSC Datamart and return its local path."""
    import time
    fname = _MSC_FNAME.format(date=date, variable=variable, level=level, fxx=fxx)
    local = save_dir / fname
    if local.exists() and local.stat().st_size > 0 and not overwrite:
        return local

    url = _MSC_BASE.format(date=date, product=product, fxx=fxx) + '/' + fname
    logger.info('Fetching %s', url)

    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            r = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT)
            if r.status_code == 404:
                raise NoWeatherModelData(f'HRDPS file not found on MSC Datamart: {url}')
            r.raise_for_status()
            with open(local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return local
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            local.unlink(missing_ok=True)  # remove any partial file
            if attempt == _DOWNLOAD_RETRIES:
                raise
            logger.warning('Download attempt %d/%d failed (%s), retrying in %ds...',
                           attempt, _DOWNLOAD_RETRIES, e, _DOWNLOAD_RETRY_WAIT)
            time.sleep(_DOWNLOAD_RETRY_WAIT)
        except Exception:
            local.unlink(missing_ok=True)
            raise

    return local  # unreachable, satisfies type checkers


def check_hrdps_dataset_availability(datetime: dt.datetime) -> bool:
    """Note a file could still be missing within the model's valid range."""
    fname = _MSC_FNAME.format(date=datetime, variable='TMP', level='Sfc', fxx=0)
    url = _MSC_BASE.format(date=datetime, product='continental/2.5km', fxx=0) + '/' + fname
    try:
        r = requests.head(url, timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


def download_hrdps_file(ll_bounds, DATE, out: Path, product='continental/2.5km', fxx=0, verbose=False) -> None:
    """
    Download HRDPS weather model data from MSC Datamart.

    HRDPS serves one GRIB2 file per variable+level. We loop over all pressure
    levels and the three required variables (TMP, SPFH, HGT), stack into a
    single netCDF4 output.

    Args:
        ll_bounds           - SNWE bounding box
        DATE (datetime)     - Analysis run datetime (00/06/12/18 UTC)
        out (Path)          - Output file path
        product (string)    - Herbie product string, e.g. 'continental/2.5km'
        fxx (int)           - Forecast hour offset from analysis run
        verbose (bool)      - Log each file download
    """
    save_dir = out.parent / 'hrdps_grib' / DATE.strftime('%Y%m%d')
    if save_dir.exists() and not save_dir.is_dir():
        save_dir.unlink()
    save_dir.mkdir(parents=True, exist_ok=True)

    var_grib_to_nc = {'TMP': 't', 'SPFH': 'q', 'HGT': 'z'}
    accumulated = {v: [] for v in var_grib_to_nc}
    ref_lats = ref_lons = ref_proj = None

    for level_hpa in HRDPS_PRESSURE_LEVELS_HPA:
        level_str = f'ISBL_{level_hpa:04d}'
        for var_grib in var_grib_to_nc:
            grib_path = _fetch_grib(DATE, var_grib, level_str, fxx, product, save_dir)
            if verbose:
                logger.info('Downloaded %s', grib_path)

            datasets = cfgrib.open_datasets(str(grib_path))
            main_vars = [v for ds in datasets for v in ds.data_vars
                         if 'projection' not in str(v).lower()]
            if not main_vars:
                raise RuntimeError(f'No data variable in GRIB for {var_grib} at {level_str}')

            ds = next(ds for ds in datasets if main_vars[0] in ds.data_vars)
            da = ds[main_vars[0]]
            accumulated[var_grib].append(da.values)

            if ref_lats is None:
                ref_lats = ds['latitude'].values
                ref_lons = ds['longitude'].values
                # HRDPS uses a rotated lat/lon grid; treat as geographic
                ref_proj = CRS.from_epsg(4326)

    # Stack: shape (n_levels, ny, nx)
    t_3d = np.stack(accumulated['TMP'], axis=0)
    q_3d = np.stack(accumulated['SPFH'], axis=0)
    z_3d = np.stack(accumulated['HGT'], axis=0)

    levels_hpa = np.array(HRDPS_PRESSURE_LEVELS_HPA, dtype=float)
    ny, nx = ref_lats.shape
    pres_3d = np.broadcast_to(
        (levels_hpa * 100)[:, np.newaxis, np.newaxis],
        (len(levels_hpa), ny, nx),
    ).copy()

    try:
        x_min, x_max, y_min, y_max = get_bounds_indices(ll_bounds, ref_lats, ref_lons)
    except NoWeatherModelData as e:
        logger.error(e)
        logger.error('lat/lon bounds: %s', ll_bounds)
        raise

    lats_sub = ref_lats[y_min:y_max, x_min:x_max]
    lons_sub = ref_lons[y_min:y_max, x_min:x_max]

    # Build 1D coordinate arrays the WeatherModel base class expects.
    # Row-mean lat and column-mean lon are good approximations for the
    # nearly-regular HRDPS rotated lat/lon grid.
    y_1d = np.mean(lats_sub, axis=1)
    x_1d = np.mean(lons_sub, axis=0)

    ds_out = xr.Dataset(
        {
            't':         (['levels', 'y', 'x'], t_3d[:, y_min:y_max, x_min:x_max]),
            'q':         (['levels', 'y', 'x'], q_3d[:, y_min:y_max, x_min:x_max]),
            'z':         (['levels', 'y', 'x'], z_3d[:, y_min:y_max, x_min:x_max]),
            'pres':      (['levels', 'y', 'x'], pres_3d[:, y_min:y_max, x_min:x_max]),
            'latitude':  (['y', 'x'], lats_sub),
            'longitude': (['y', 'x'], lons_sub),
        },
        coords={'levels': levels_hpa, 'y': y_1d, 'x': x_1d},
    )

    ds_out['proj'] = 0
    for k, v in ref_proj.to_cf().items():
        ds_out.proj.attrs[k] = v

    ds_out.to_netcdf(out, engine='netcdf4')


def load_weather_hrdps(filename):
    """Load a weather model from an HRDPS netCDF file."""
    ds = xr.open_dataset(filename, engine='netcdf4')

    # Transpose (levels, y, x) → (y, x, levels) — RAiDER convention
    pres    = ds['pres'].values.transpose(1, 2, 0).copy()
    temps   = ds['t'].values.transpose(1, 2, 0).copy()
    qs      = ds['q'].values.transpose(1, 2, 0).copy()
    geo_hgt = ds['z'].values.transpose(1, 2, 0).copy()
    lats    = ds['latitude'].values.copy()   # 2D geographic lat
    lons    = ds['longitude'].values.copy()  # 2D geographic lon

    # 1D coordinate arrays stored at download time (row-mean lat, col-mean lon)
    xArr = ds['x'].values.copy()  # shape (nx,)
    yArr = ds['y'].values.copy()  # shape (ny,)

    proj = CRS.from_cf(ds['proj'].attrs)
    lons[lons > 180] -= 360
    xArr[xArr > 180] -= 360

    # Broadcast 1D arrays to 3D — matches what HRRR does
    _xs = np.broadcast_to(xArr[np.newaxis, :, np.newaxis], geo_hgt.shape).copy() 
    _ys = np.broadcast_to(yArr[:, np.newaxis, np.newaxis], geo_hgt.shape).copy() 

    return _xs, _ys, lons, lats, qs, temps, pres, geo_hgt, proj


class HRDPS(WeatherModel):
    def __init__(self) -> None:
        super().__init__()

        self._humidityType = 'q'
        self._model_level_type = 'pl'
        self._expver = '0001'
        self._classname = 'hrdps'
        self._dataset = 'hrdps'

        self._time_res = TIME_RES[self._dataset.upper()]

        # MSC Datamart keeps a ~30-day rolling archive under dd.weather.gc.ca/{YYYYMMDD}/WXO-DD/
        self._valid_range = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30),
            dt.datetime.now(dt.timezone.utc),
        )
        self._lag_time = dt.timedelta(hours=2)

        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # 2.5 km horizontal grid spacing
        self._lat_res = 2.5 / 111
        self._lon_res = 2.5 / 111
        self._x_res = 2.5
        self._y_res = 2.5

        self._Nproc = 1
        self._Name = 'HRDPS'
        self._Npl = 0
        self.files = None
        self._bounds = None

        self._proj = CRS.from_epsg(4326)
        self._valid_bounds = HRDPS_COVERAGE_POLYGON

    def __pressure_levels__(self):
        self._levels = len(HRDPS_PRESSURE_LEVELS_HPA)
        self._zlevels = np.flipud(LEVELS_50_HEIGHTS)

    def __model_levels__(self):
        self._levels = len(HRDPS_PRESSURE_LEVELS_HPA)
        self._zlevels = np.flipud(LEVELS_50_HEIGHTS)

    def _fetch(self, out) -> None:
        """Fetch weather model data from HRDPS."""
        self._files = out
        corrected_DT = round_date(self._time, dt.timedelta(hours=self._time_res))
        self.checkTime(corrected_DT)
        if not corrected_DT == self._time:
            logger.info('Rounded given datetime from %s to %s', self._time, corrected_DT)

        # HRDPS has analysis runs at 00/06/12/18 UTC; derive run time and fxx
        analysis_hour = (corrected_DT.hour // 6) * 6
        analysis_run = corrected_DT.replace(hour=analysis_hour, minute=0, second=0, microsecond=0)
        fxx = int((corrected_DT - analysis_run).total_seconds() / 3600)

        bounds = self._ll_bounds.copy()
        bounds[2:] = np.mod(bounds[2:], 360)

        download_hrdps_file(bounds, analysis_run, out, fxx=fxx)

    def load_weather(self, f=None, *args, **kwargs) -> None:
        if f is None:
            f = self.files[0] if isinstance(self.files, list) else self.files

        _xs, _ys, _lons, _lats, qs, temps, pres, geo_hgt, proj = load_weather_hrdps(f)

        self._get_heights(_lats, geo_hgt)

        self._t = temps
        self._q = qs
        self._p = pres
        self._xs = _xs
        self._ys = _ys
        self._lats = _lats
        self._lons = _lons
        self._proj = proj

    def checkValidBounds(self, ll_bounds: np.ndarray) -> None:
        S, N, W, E = ll_bounds
        aoi = box(W, S, E, N)
        if self._valid_bounds.contains(aoi):
            pass
        elif aoi.intersects(self._valid_bounds):
            logger.critical('The HRDPS weather model extent does not completely cover your AOI!')
        else:
            raise ValueError('The requested location is unavailable for HRDPS')
