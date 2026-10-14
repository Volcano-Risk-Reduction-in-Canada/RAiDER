"""Per-AOI 10 m UTM DEMs from Copernicus GLO-30.

Heights follow RAiDER's convention for DEMs it downloads itself
(``RAiDER.dem.download_dem``): GLO-30 converted to WGS84 ellipsoidal heights by
dem_stitcher. ``download_dem`` itself is not called because it rounds the
request out to whole-degree tiles; this module asks dem_stitcher for the AOI's
extent only, with the same options.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from hrdps_synthetic.aois import AOI
from hrdps_synthetic.geometry import SceneGrid

log = logging.getLogger('hrdps_synthetic')

GLO30_ARCSEC = 1.0

# Remote COG reads: do not probe for sidecar files (avoids slow 403 round trips).
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.TIF,.tiff,.vrt')


def dem_path(workdir: Path, aoi: AOI) -> Path:
    return Path(workdir) / 'dem' / f'{aoi.name}_glo30_utm10m.tif'


def dem_extent(aoi: AOI, width_m: float, length_m: float, spacing: float, margin_m: float = 200.0) -> tuple:
    """UTM extent (snapped to `spacing`) covering every possible scene of the AOI."""
    xmin, ymin, xmax, ymax = aoi.reach_utm(width=width_m, length=length_m)
    snap = lambda v, f: f(v / spacing) * spacing  # noqa: E731
    return (snap(xmin - margin_m, np.floor), snap(ymin - margin_m, np.floor),
            snap(xmax + margin_m, np.ceil), snap(ymax + margin_m, np.ceil))


def fetch_glo30_utm(aoi: AOI, extent: tuple, spacing: float) -> tuple[np.ndarray, rasterio.Affine]:
    """GLO-30 (ellipsoidal heights) bilinear-reprojected onto a UTM grid; in memory only."""
    from dem_stitcher.stitcher import stitch_dem

    W, S, E, N = aoi.utm_rect_to_lonlat(extent).bounds
    bounds = [W - 0.01, S - 0.01, E + 0.01, N + 0.01]
    for attempt in range(1, 4):
        try:
            src, prof = stitch_dem(bounds, dem_name='glo_30', dst_ellipsoidal_height=True, dst_area_or_point='Area')
            break
        except Exception:
            if attempt == 3:
                raise
            log.warning('GLO-30 read failed for %s; retrying in %d s', aoi.name, 30 * attempt)
            time.sleep(30 * attempt)
    xmin, ymin, xmax, ymax = extent
    nx, ny = int(round((xmax - xmin) / spacing)), int(round((ymax - ymin) / spacing))
    transform = from_origin(xmin, ymax, spacing, spacing)
    dst = np.full((ny, nx), np.nan, dtype=np.float32)
    reproject(src.astype(np.float32), dst, src_transform=prof['transform'], src_crs=prof['crs'],
              dst_transform=transform, dst_crs=f'EPSG:{aoi.epsg}', resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    if np.isnan(dst).any():
        raise ValueError(f'GLO-30 has gaps over {aoi.name}')
    return dst, transform


def ensure_aoi_dem(aoi: AOI, workdir: Path, width_m: float = 20_000.0, length_m: float = 30_000.0,
                   spacing: float = 10.0) -> Path:
    path = dem_path(workdir, aoi)
    if path.exists():
        return path
    extent = dem_extent(aoi, width_m, length_m, spacing)
    t0 = time.time()
    arr, transform = fetch_glo30_utm(aoi, extent, spacing)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    profile = dict(driver='GTiff', dtype='float32', width=arr.shape[1], height=arr.shape[0], count=1,
                   crs=f'EPSG:{aoi.epsg}', transform=transform, compress='deflate', predictor=3,
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    with rasterio.open(tmp, 'w', **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(source='Copernicus GLO-30 via dem_stitcher', heights='WGS84 ellipsoidal (RAiDER convention)',
                        resampling='bilinear', aoi=aoi.name)
    tmp.replace(path)
    log.info('DEM for %s: %d x %d at %.0f m (%.0f s)', aoi.name, arr.shape[1], arr.shape[0], spacing, time.time() - t0)
    return path


def read_scene_dem(path: Path, scene: SceneGrid) -> np.ndarray:
    """Read the scene window from an AOI DEM (both on the same snapped grid)."""
    with rasterio.open(path) as ds:
        t = ds.transform
        if abs(t.a - scene.spacing) > 1e-6 or ds.crs.to_epsg() != scene.epsg:
            raise ValueError(f'{path} does not match the scene grid')
        col0 = int(round((scene.x0 - t.c) / t.a))
        row0 = int(round((t.f - scene.y0) / -t.e))
        if col0 < 0 or row0 < 0 or col0 + scene.nx > ds.width or row0 + scene.ny > ds.height:
            raise ValueError(f'Scene outside the DEM extent of {path}')
        return ds.read(1, window=Window(col0, row0, scene.nx, scene.ny)).astype(np.float32)


def scene_dem_in_memory(aoi: AOI, scene: SceneGrid) -> np.ndarray:
    """DEM for one scene without writing anything (used by --dry-run probes)."""
    xmin, ymin, xmax, ymax = scene.bounds
    arr, _ = fetch_glo30_utm(aoi, (xmin, ymin, xmax, ymax), scene.spacing)
    return arr


def estimate_dem_bytes(aoi: AOI, width_m: float, length_m: float, spacing: float) -> tuple[int, int]:
    """(GLO-30 bytes read, 10 m UTM DEM bytes uncompressed) for one AOI."""
    xmin, ymin, xmax, ymax = dem_extent(aoi, width_m, length_m, spacing)
    W, S, E, N = aoi.utm_rect_to_lonlat((xmin, ymin, xmax, ymax)).bounds
    lat = abs(0.5 * (S + N))
    lon_factor = 1 if lat < 50 else 1.5 if lat < 60 else 2  # GLO-30 longitude spacing widens poleward
    n_src = ((N - S + 0.02) * 3600 / GLO30_ARCSEC) * ((E - W + 0.02) * 3600 / (GLO30_ARCSEC * lon_factor))
    n_dst = ((xmax - xmin) / spacing) * ((ymax - ymin) / spacing)
    return int(4 * n_src), int(4 * n_dst)


def maybe_dem(workdir: Optional[Path], aoi: AOI) -> Optional[Path]:
    if workdir is None:
        return None
    p = dem_path(workdir, aoi)
    return p if p.exists() else None
