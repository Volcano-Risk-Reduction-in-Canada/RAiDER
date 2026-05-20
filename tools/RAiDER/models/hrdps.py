import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from herbie import Herbie
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box

from RAiDER.logger import logger
from RAiDER.models.customExceptions import NoWeatherModelData
from RAiDER.models.hrrr import get_bounds_indices
from RAiDER.models.model_levels import LEVELS_50_HEIGHTS
from RAiDER.models.weatherModel import TIME_RES, WeatherModel
from RAiDER.utilFcns import round_date


# HRDPS Continental domain (~41–87°N, ~148–40°W)
HRDPS_COVERAGE_POLYGON = Polygon(((-148, 41), (-148, 87), (-40, 87), (-40, 41)))


def check_hrdps_dataset_availability(datetime: dt.datetime) -> bool:
    """Note a file could still be missing within the model's valid range."""
    herbie = Herbie(
        datetime,
        model='hrdps',
        product='continental',
        fxx=0,
    )
    return herbie.grib_source is not None


def download_hrdps_file(ll_bounds, DATE, out: Path, product='continental', fxx=0, verbose=False) -> None:
    """
    Download an HRDPS weather model using Herbie.

    Args:
        ll_bounds           - SNWE bounding box
        DATE (datetime)     - Analysis run datetime (00/06/12/18 UTC)
        out (Path)          - Output file path
        product (string)    - 'continental' or 'national'
        fxx (int)           - Forecast hour offset from analysis run
        verbose (bool)      - True for extra printout
    """
    herbie = Herbie(
        DATE.strftime('%Y-%m-%d %H:%M'),
        model='hrdps',
        product=product,
        fxx=fxx,
        overwrite=False,
        verbose=True,
        save_dir=out.parent,
    )

    try:
        ds_list = herbie.xarray(':(SPFH|PRES|TMP|HGT):', verbose=verbose)
    except ValueError as e:
        logger.error(e)
        raise

    ds_list_filt_0 = [ds for ds in ds_list if 'hybrid' in ds._coord_names]
    ds_list_filt_1 = [ds for ds in ds_list if 'isobaricInhPa' in ds._coord_names]
    if ds_list_filt_0:
        ds_out = ds_list_filt_0[0]
        coord = 'hybrid'
    elif ds_list_filt_1:
        ds_out = ds_list_filt_1[0]
        coord = 'isobaricInhPa'
    else:
        raise RuntimeError('Herbie did not obtain an HRDPS dataset with the expected layers and coordinates')

    try:
        x_min, x_max, y_min, y_max = get_bounds_indices(
            ll_bounds,
            ds_out.latitude.to_numpy(),
            ds_out.longitude.to_numpy(),
        )
    except NoWeatherModelData as e:
        logger.error(e)
        logger.error('lat/lon bounds: %s', ll_bounds)
        raise

    ds_out = ds_out.rename({'gh': 'z', coord: 'levels'})

    ds_out['proj'] = 0
    for k, v in CRS.from_user_input(ds_out.herbie.crs).to_cf().items():
        ds_out.proj.attrs[k] = v
    for var in ds_out.data_vars:
        ds_out[var].attrs['grid_mapping'] = 'proj'

    proj = CRS.from_cf(ds_out['proj'].attrs)
    t = Transformer.from_crs(4326, proj, always_xy=True)

    xl, yl = t.transform(ds_out['longitude'].values, ds_out['latitude'].values)
    W, E, S, N = np.nanmin(xl), np.nanmax(xl), np.nanmin(yl), np.nanmax(yl)

    grid_x = 2500  # meters
    grid_y = 2500  # meters
    xs = np.arange(W, E + grid_x / 2, grid_x)
    ys = np.arange(S, N + grid_y / 2, grid_y)

    ds_out['x'] = xs
    ds_out['y'] = ys
    ds_sub = ds_out.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    ds_sub.to_netcdf(out, engine='netcdf4')


def load_weather_hrdps(filename):
    """Load a weather model from an HRDPS file."""
    ds = xr.open_dataset(filename, engine='netcdf4')
    pres = ds['pres'].values.transpose(1, 2, 0)
    xArr = ds['x'].values
    yArr = ds['y'].values
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    temps = ds['t'].values.transpose(1, 2, 0)
    qs = ds['q'].values.transpose(1, 2, 0)
    geo_hgt = ds['z'].values.transpose(1, 2, 0)

    proj = CRS.from_cf(ds['proj'].attrs)

    lons[lons > 180] -= 360

    _xs = np.broadcast_to(xArr[np.newaxis, :, np.newaxis], geo_hgt.shape)
    _ys = np.broadcast_to(yArr[:, np.newaxis, np.newaxis], geo_hgt.shape)

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

        self._valid_range = (
            dt.datetime(2019, 9, 1).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
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

        # Placeholder projection — overwritten from file on load.
        # HRDPS Continental uses a Polar Stereographic projection.
        self._proj = CRS.from_epsg(4326)
        self._valid_bounds = HRDPS_COVERAGE_POLYGON
        self.setLevelType('nat')

    def __model_levels__(self):
        self._levels = 50
        self._zlevels = np.flipud(LEVELS_50_HEIGHTS)

    def __pressure_levels__(self):
        raise NotImplementedError('Pressure levels do not go high enough for HRDPS.')

    def _fetch(self, out) -> None:
        """Fetch weather model data from HRDPS."""
        self._files = out
        corrected_DT = round_date(self._time, dt.timedelta(hours=self._time_res))
        self.checkTime(corrected_DT)
        if not corrected_DT == self._time:
            logger.info('Rounded given datetime from %s to %s', self._time, corrected_DT)

        # HRDPS has analysis runs at 00/06/12/18 UTC; compute analysis time and fxx offset
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
