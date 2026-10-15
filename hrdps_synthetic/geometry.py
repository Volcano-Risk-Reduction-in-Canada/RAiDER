"""Scene grid, placement and incidence geometry.

Two layouts are supported:

* **radar** (default): the same array layout as the synthetic magmatic
  interferograms, so atmosphere and deformation can be summed pixel for pixel.
  - Rows run along-track in flight order: row 0 is the start of the swath, and
    the satellite flies toward increasing rows.
  - Columns run across-track from near range (column 0, on the satellite side)
    to far range.
  - The default size is 30 km along-track by 20 km across-track, 3000 × 2000 px
    at 10 m. This is the usual SLC orientation. Ascending scenes therefore look
    flipped north–south, and descending scenes mirrored east–west, compared with
    a map.
  - GeoTIFFs carry no CRS. Their transform is the local metric frame of the
    magmatic files (origin at the swath centre). The real location is in the
    manifest and the GeoTIFF tags.
* **north-up**: rasters in the AOI's UTM zone (20 km E–W by 30 km N–S by default).

In both layouts the radar looks right of the heading, so the look azimuth is
heading + 90°. Incidence varies across track on a spherical Earth with the
satellite at RCM altitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from affine import Affine
from scipy.ndimage import map_coordinates

from hrdps_synthetic.aois import (AOI, GEOMETRIES, grid_azimuth_of_true_north, radar_basis,  # noqa: F401
                                  to_lonlat, to_utm)

R_EARTH_M = 6_371_008.8
RCM_ALTITUDE_M = 600_000.0


@dataclass(frozen=True)
class SceneGrid:
    epsg: int
    cx: float  # scene centre, UTM metres
    cy: float
    nx: int  # columns (east-west, or across-track)
    ny: int  # rows (north-south, or along-track)
    spacing: float
    geometry: str = 'north-up'
    flight_az_grid_deg: float = 0.0  # radar only: flight azimuth clockwise from UTM grid north

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def center(self) -> tuple[float, float]:
        return self.cx, self.cy

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """UTM displacement (m) of one row step and one column step."""
        if self.geometry == 'north-up':
            return np.array([0.0, -self.spacing]), np.array([self.spacing, 0.0])
        f, l = radar_basis(self.flight_az_grid_deg)
        return self.spacing * np.array(f), self.spacing * np.array(l)

    @property
    def origin(self) -> tuple[float, float]:
        """UTM position of the outer corner of pixel (row 0, col 0)."""
        rv, cv = self.basis()
        o = np.array([self.cx, self.cy]) - self.ny / 2 * rv - self.nx / 2 * cv
        return float(o[0]), float(o[1])

    @property
    def x0(self) -> float:
        return self.origin[0]

    @property
    def y0(self) -> float:
        return self.origin[1]

    def edge_xy(self, r, c) -> tuple[np.ndarray, np.ndarray]:
        """UTM position of pixel-edge coordinates (r, c); pixel centres are at r + 0.5, c + 0.5."""
        rv, cv = self.basis()
        ox, oy = self.origin
        r, c = np.asarray(r, dtype=np.float64), np.asarray(c, dtype=np.float64)
        return ox + r * rv[0] + c * cv[0], oy + r * rv[1] + c * cv[1]

    def pixel_xy(self, rows, cols) -> tuple[np.ndarray, np.ndarray]:
        return self.edge_xy(np.asarray(rows) + 0.5, np.asarray(cols) + 0.5)

    def corners(self) -> np.ndarray:
        """(4, 2) UTM corners in pixel order: (0,0), (0,nx), (ny,nx), (ny,0)."""
        x, y = self.edge_xy([0, 0, self.ny, self.ny], [0, self.nx, self.nx, 0])
        return np.stack([x, y], axis=1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        c = self.corners()
        return float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())

    @property
    def crs(self) -> Optional[str]:
        return f'EPSG:{self.epsg}' if self.geometry == 'north-up' else None

    @property
    def transform(self) -> Affine:
        if self.geometry == 'north-up':
            return Affine(self.spacing, 0.0, self.x0, 0.0, -self.spacing, self.y0)
        # Local metric frame of the magmatic GeoTIFFs: origin at the swath centre.
        return Affine(self.spacing, 0.0, -self.nx * self.spacing / 2, 0.0, -self.spacing, self.ny * self.spacing / 2)

    def corners_lonlat(self) -> str:
        """'lon lat; ...' for the four corners in pixel order (UL, UR, LR, LL as displayed)."""
        c = self.corners()
        lon, lat = to_lonlat(self.epsg, c[:, 0], c[:, 1])
        return '; '.join(f'{a:.6f} {b:.6f}' for a, b in zip(lon, lat))


def scene_for(aoi: AOI, dx: float, dy: float, width_m: float = 20_000.0, length_m: float = 30_000.0,
              spacing: float = 10.0, geometry: str = 'north-up', heading: Optional[float] = None) -> SceneGrid:
    """Scene of an AOI shifted by (dx, dy) in its own axes (see AOI.offset_ranges).

    north-up scenes are snapped to the `spacing` grid so they line up with the AOI
    DEM; radar scenes are rotated with the heading and resample the DEM.
    """
    nx, ny = int(round(width_m / spacing)), int(round(length_m / spacing))
    if geometry == 'north-up':
        x0 = round((aoi.cx + dx - width_m / 2) / spacing) * spacing
        y0 = round((aoi.cy + dy + length_m / 2) / spacing) * spacing
        return SceneGrid(aoi.epsg, x0 + nx * spacing / 2, y0 - ny * spacing / 2, nx, ny, spacing)
    if geometry != 'radar':
        raise ValueError(f'geometry must be one of {GEOMETRIES}')
    if heading is None:
        raise ValueError('radar geometry needs a heading')
    phi = aoi.flight_az_grid(heading)
    f, l = (np.array(v) for v in radar_basis(phi))
    c = np.array([aoi.cx, aoi.cy]) + dy * f + dx * l
    return SceneGrid(aoi.epsg, float(c[0]), float(c[1]), nx, ny, spacing, 'radar', phi)


def look_azimuth_deg(heading_deg: float) -> float:
    """Right-looking SAR: look direction is 90 degrees clockwise from the heading."""
    return (heading_deg + 90.0) % 360.0


def ground_range_from_incidence(theta_deg, R: float = R_EARTH_M, H: float = RCM_ALTITUDE_M):
    """Arc distance from nadir to a target seen at incidence theta (spherical Earth)."""
    th = np.radians(theta_deg)
    look = np.arcsin(R * np.sin(th) / (R + H))
    return R * (th - look)


def incidence_from_ground_range(G, R: float = R_EARTH_M, H: float = RCM_ALTITUDE_M):
    """Incidence (degrees) of a target at arc distance G from nadir (spherical Earth)."""
    gamma = np.asarray(G, dtype=np.float64) / R
    look = np.arctan2(R * np.sin(gamma), R + H - R * np.cos(gamma))
    return np.degrees(look + gamma)


def _look_unit(scene: SceneGrid, heading_deg: float) -> tuple[float, float]:
    if scene.geometry == 'radar':
        return radar_basis(scene.flight_az_grid_deg)[1]
    lon_c, lat_c = to_lonlat(scene.epsg, scene.cx, scene.cy)
    az = math.radians(look_azimuth_deg(heading_deg) + grid_azimuth_of_true_north(scene.epsg, lon_c, lat_c))
    return math.sin(az), math.cos(az)


def incidence_at(scene: SceneGrid, theta_c: float, heading_deg: float, x, y,
                 R: float = R_EARTH_M, H: float = RCM_ALTITUDE_M) -> np.ndarray:
    """Incidence (deg) at UTM points (x, y), given the scene-centre incidence and heading."""
    lx, ly = _look_unit(scene, heading_deg)
    # Signed cross-track distance, positive away from the satellite (farther range).
    r = (np.asarray(x) - scene.cx) * lx + (np.asarray(y) - scene.cy) * ly
    return incidence_from_ground_range(ground_range_from_incidence(theta_c, R, H) + r, R, H)


def incidence_field(scene: SceneGrid, theta_c: float, heading_deg: float,
                    R: float = R_EARTH_M, H: float = RCM_ALTITUDE_M) -> np.ndarray:
    """Per-pixel incidence (deg, float32) given the scene-centre incidence and heading."""
    if scene.geometry == 'radar':  # depends on the column (across-track distance) only
        g = (np.arange(scene.nx) + 0.5 - scene.nx / 2) * scene.spacing
        th = incidence_from_ground_range(ground_range_from_incidence(theta_c, R, H) + g, R, H).astype(np.float32)
        return np.broadcast_to(th[None, :], scene.shape)
    x, y = scene.pixel_xy(np.arange(scene.ny)[:, None], np.arange(scene.nx)[None, :])
    return incidence_at(scene, theta_c, heading_deg, x, y, R, H).astype(np.float32)


def los_ground_to_satellite(theta_deg: float, heading_deg: float) -> tuple[float, float, float]:
    """Unit vector (east, north, up) from the ground pixel toward a right-looking satellite.

    Heading is clockwise from true north; the look azimuth is heading + 90.
    A LOS displacement d_los = d . los is positive toward the satellite.
    """
    th, az = math.radians(theta_deg), math.radians(look_azimuth_deg(heading_deg))
    return (-math.sin(th) * math.sin(az), -math.sin(th) * math.cos(az), math.cos(th))


def coarse_lonlat(scene: SceneGrid, step_px: int = 10) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Lon/lat on a coarse node grid of pixel centres (every `step_px` pixels plus the last)."""
    rows = np.unique(np.r_[np.arange(0, scene.ny, step_px), scene.ny - 1])
    cols = np.unique(np.r_[np.arange(0, scene.nx, step_px), scene.nx - 1])
    X, Y = scene.pixel_xy(rows[:, None], cols[None, :])
    lon, lat = to_lonlat(scene.epsg, X, Y)
    return np.asarray(lon), np.asarray(lat), (rows, cols)


def upsample(coarse: np.ndarray, nodes: tuple[np.ndarray, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Bilinear upsampling of a coarse node field to every pixel (float32)."""
    rows, cols = nodes
    rr = np.interp(np.arange(shape[0]), rows, np.arange(rows.size)).astype(np.float32)
    cc = np.interp(np.arange(shape[1]), cols, np.arange(cols.size)).astype(np.float32)
    R_, C_ = np.meshgrid(rr, cc, indexing='ij')
    return map_coordinates(np.asarray(coarse, dtype=np.float64), [R_, C_], order=1, mode='nearest',
                           prefilter=False, output=np.float32)
