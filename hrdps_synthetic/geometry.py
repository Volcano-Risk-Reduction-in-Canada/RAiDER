"""Scene grid, placement and incidence geometry.

Scenes are north-up rasters in the AOI's UTM zone (default 2000 columns x
3000 rows at 10 m). The radar looks to the right of the heading, so the look
azimuth is heading + 90 degrees. Incidence varies across track: the scene-centre
incidence is the sampled value, and the rest of the scene follows a spherical
Earth with the satellite at RCM altitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine
from scipy.ndimage import map_coordinates

from hrdps_synthetic.aois import AOI, to_lonlat, to_utm

R_EARTH_M = 6_371_008.8
RCM_ALTITUDE_M = 600_000.0


@dataclass(frozen=True)
class SceneGrid:
    epsg: int
    x0: float  # west edge (m)
    y0: float  # north edge (m)
    nx: int
    ny: int
    spacing: float

    @property
    def transform(self) -> Affine:
        return Affine(self.spacing, 0.0, self.x0, 0.0, -self.spacing, self.y0)

    @property
    def center(self) -> tuple[float, float]:
        return self.x0 + self.nx * self.spacing / 2, self.y0 - self.ny * self.spacing / 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0 - self.ny * self.spacing, self.x0 + self.nx * self.spacing, self.y0

    def pixel_centers(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.spacing
        y = self.y0 - (np.arange(self.ny) + 0.5) * self.spacing
        return x, y


def scene_for(aoi: AOI, dx: float, dy: float, width_m: float = 20_000.0, length_m: float = 30_000.0,
              spacing: float = 10.0) -> SceneGrid:
    """Scene centred on the AOI centroid + (dx, dy), snapped to the `spacing` grid."""
    cx, cy = aoi.cx + dx, aoi.cy + dy
    x0 = round((cx - width_m / 2) / spacing) * spacing
    y0 = round((cy + length_m / 2) / spacing) * spacing
    return SceneGrid(aoi.epsg, float(x0), float(y0), int(round(width_m / spacing)), int(round(length_m / spacing)), spacing)


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


def grid_azimuth_of_true_north(epsg: int, lon: float, lat: float) -> float:
    """Clockwise angle (deg) from UTM grid north to true north at (lon, lat)."""
    x0, y0 = to_utm(epsg, lon, lat)
    x1, y1 = to_utm(epsg, lon, lat + 0.01)
    return math.degrees(math.atan2(x1 - x0, y1 - y0))


def incidence_at(scene: SceneGrid, theta_c: float, heading_deg: float, x, y,
                 R: float = R_EARTH_M, H: float = RCM_ALTITUDE_M) -> np.ndarray:
    """Incidence (deg) at UTM points (x, y), given the scene-centre incidence and heading."""
    cx, cy = scene.center
    lon_c, lat_c = to_lonlat(scene.epsg, cx, cy)
    az = math.radians(look_azimuth_deg(heading_deg) + grid_azimuth_of_true_north(scene.epsg, lon_c, lat_c))
    # Signed cross-track distance, positive away from the satellite (farther range).
    r = (np.asarray(x) - cx) * math.sin(az) + (np.asarray(y) - cy) * math.cos(az)
    return incidence_from_ground_range(ground_range_from_incidence(theta_c, R, H) + r, R, H)


def incidence_field(scene: SceneGrid, theta_c: float, heading_deg: float,
                    R: float = R_EARTH_M, H: float = RCM_ALTITUDE_M) -> np.ndarray:
    """Per-pixel incidence (deg, float32) given the scene-centre incidence and heading."""
    x, y = scene.pixel_centers()
    return incidence_at(scene, theta_c, heading_deg, x[None, :], y[:, None], R, H).astype(np.float32)


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
    xs = scene.x0 + (cols + 0.5) * scene.spacing
    ys = scene.y0 - (rows + 0.5) * scene.spacing
    X, Y = np.meshgrid(xs, ys)
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
