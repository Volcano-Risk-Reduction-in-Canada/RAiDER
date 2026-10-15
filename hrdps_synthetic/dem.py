"""Per-AOI 10 m UTM DEMs from Copernicus GLO-30.

Heights follow RAiDER's convention for DEMs it downloads itself
(``RAiDER.dem.download_dem``): GLO-30 converted to WGS84 ellipsoidal heights by
dem_stitcher. ``download_dem`` itself is not called because it rounds the
request out to whole-degree tiles; this module asks dem_stitcher for the AOI's
extent only, with the same options.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from scipy.ndimage import map_coordinates

from hrdps_synthetic.aois import AOI, RCM_HEADINGS
from hrdps_synthetic.geometry import SceneGrid

log = logging.getLogger('hrdps_synthetic')

GLO30_ARCSEC = 1.0

# Remote COG reads: do not probe for sidecar files (avoids slow 403 round trips).
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.TIF,.tiff,.vrt')


def dem_path(workdir: Path, aoi: AOI, spacing: float = 10.0) -> Path:
    return Path(workdir) / 'dem' / f'{aoi.name}_glo30_utm{spacing:g}m.tif'


def dem_extent(aoi: AOI, width_m: float, length_m: float, spacing: float, margin_m: float = 200.0,
               geometry: str = 'north-up', headings: Sequence[float] = RCM_HEADINGS) -> tuple:
    """UTM extent (snapped to `spacing`) covering every possible scene of the AOI."""
    xmin, ymin, xmax, ymax = aoi.reach_utm(width=width_m, length=length_m, geometry=geometry, headings=headings)
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


def dem_covers(path: Path, extent: tuple, epsg: int, spacing: float) -> bool:
    with rasterio.open(path) as ds:
        b = ds.bounds
        return (ds.crs.to_epsg() == epsg and abs(ds.res[0] - spacing) < 1e-6 and b.left <= extent[0] + 1e-6
                and b.bottom <= extent[1] + 1e-6 and b.right >= extent[2] - 1e-6 and b.top >= extent[3] - 1e-6)


def ensure_aoi_dem(aoi: AOI, workdir: Path, width_m: float = 20_000.0, length_m: float = 30_000.0,
                   spacing: float = 10.0, geometry: str = 'north-up',
                   headings: Sequence[float] = RCM_HEADINGS) -> Path:
    """Build (or reuse) the AOI DEM; rebuild it if it is too small for these scenes."""
    path = dem_path(workdir, aoi, spacing)
    extent = dem_extent(aoi, width_m, length_m, spacing, geometry=geometry, headings=headings)
    if path.exists():
        if dem_covers(path, extent, aoi.epsg, spacing):
            return path
        log.info('DEM for %s is too small for %.0f x %.0f km scenes; rebuilding', aoi.name, width_m / 1e3, length_m / 1e3)
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


def resample_to_scene(block: np.ndarray, block_transform, scene: SceneGrid) -> np.ndarray:
    """Bilinear heights at every scene pixel centre from a north-up UTM DEM block."""
    t = block_transform
    rv, cv = scene.basis()
    ox, oy = scene.origin
    # Fractional (row, col) in the block are affine in the scene's (row, col).
    c0 = (ox - t.c) / t.a - 0.5
    r0 = (t.f - oy) / -t.e - 0.5
    rr = (np.arange(scene.ny) + 0.5)[:, None]
    cc = (np.arange(scene.nx) + 0.5)[None, :]
    ci = (c0 + (rr * rv[0] + cc * cv[0]) / t.a).astype(np.float32)
    ri = (r0 - (rr * rv[1] + cc * cv[1]) / -t.e).astype(np.float32)
    if ci.min() < 0 or ri.min() < 0 or ci.max() > block.shape[1] - 1 or ri.max() > block.shape[0] - 1:
        raise ValueError('Scene outside the DEM block')
    return map_coordinates(block, [ri, ci], order=1, mode='nearest', prefilter=False, output=np.float32)


def read_scene_dem(path: Path, scene: SceneGrid) -> np.ndarray:
    """Scene heights from an AOI DEM: a window read for north-up scenes, bilinear resampling for radar scenes."""
    with rasterio.open(path) as ds:
        t = ds.transform
        if abs(t.a - scene.spacing) > 1e-6 or ds.crs.to_epsg() != scene.epsg:
            raise ValueError(f'{path} does not match the scene grid')
        if scene.geometry == 'north-up':
            col0 = int(round((scene.x0 - t.c) / t.a))
            row0 = int(round((t.f - scene.y0) / -t.e))
            if col0 < 0 or row0 < 0 or col0 + scene.nx > ds.width or row0 + scene.ny > ds.height:
                raise ValueError(f'Scene outside the DEM extent of {path}')
            return ds.read(1, window=Window(col0, row0, scene.nx, scene.ny)).astype(np.float32)
        xmin, ymin, xmax, ymax = scene.bounds
        col0 = int(math.floor((xmin - t.c) / t.a)) - 2
        row0 = int(math.floor((t.f - ymax) / -t.e)) - 2
        col1 = int(math.ceil((xmax - t.c) / t.a)) + 2
        row1 = int(math.ceil((t.f - ymin) / -t.e)) + 2
        if col0 < 0 or row0 < 0 or col1 > ds.width or row1 > ds.height:
            raise ValueError(f'Scene outside the DEM extent of {path}')
        win = Window(col0, row0, col1 - col0, row1 - row0)
        block = ds.read(1, window=win).astype(np.float32)
        return resample_to_scene(block, ds.window_transform(win), scene)


def scene_dem_in_memory(aoi: AOI, scene: SceneGrid) -> np.ndarray:
    """DEM for one scene without writing anything (used by --dry-run probes)."""
    xmin, ymin, xmax, ymax = scene.bounds
    pad = 4 * scene.spacing
    snap = lambda v, f: f(v / scene.spacing) * scene.spacing  # noqa: E731
    extent = (snap(xmin - pad, math.floor), snap(ymin - pad, math.floor),
              snap(xmax + pad, math.ceil), snap(ymax + pad, math.ceil))
    arr, transform = fetch_glo30_utm(aoi, extent, scene.spacing)
    return resample_to_scene(arr, transform, scene)


def estimate_dem_bytes(aoi: AOI, width_m: float, length_m: float, spacing: float, geometry: str = 'north-up',
                       headings: Sequence[float] = RCM_HEADINGS) -> tuple[int, int]:
    """(GLO-30 bytes read, 10 m UTM DEM bytes uncompressed) for one AOI."""
    xmin, ymin, xmax, ymax = dem_extent(aoi, width_m, length_m, spacing, geometry=geometry, headings=headings)
    W, S, E, N = aoi.utm_rect_to_lonlat((xmin, ymin, xmax, ymax)).bounds
    lat = abs(0.5 * (S + N))
    lon_factor = 1 if lat < 50 else 1.5 if lat < 60 else 2  # GLO-30 longitude spacing widens poleward
    n_src = ((N - S + 0.02) * 3600 / GLO30_ARCSEC) * ((E - W + 0.02) * 3600 / (GLO30_ARCSEC * lon_factor))
    n_dst = ((xmax - xmin) / spacing) * ((ymax - ymin) / spacing)
    return int(4 * n_src), int(4 * n_dst)


def maybe_dem(workdir: Optional[Path], aoi: AOI, spacing: float = 10.0) -> Optional[Path]:
    if workdir is None:
        return None
    p = dem_path(workdir, aoi, spacing)
    return p if p.exists() else None
