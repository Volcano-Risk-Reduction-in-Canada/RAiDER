"""
Plot a vertical 2-D cross-section from an HRDPS RAiDER netCDF, optionally
overlaying a GLO-30 (or any GeoTIFF) DEM to show terrain context.

Works with either:
  * processed weather-model files  (dims: z/y/x, z in metres)
  * raw HRDPS download files       (dims: levels/y/x, levels in hPa)

Usage
-----
python plot_hrdps_profile.py <netcdf> <lat0> <lon0> <lat1> <lon1> \
    [variable] [n_points] [output.png] [dem_file]

Example (N-S cross-section with DEM overlay, total wet delay):
    python plot_hrdps_profile.py \
        test/hrdps_test_meager_std/weather_files/HRDPS_2026_05_09_T14_24_11_50N_51N_124W_123W.nc \
        50.50 -123.40 50.76 -123.40 \
        wet_total 300 profile_NS.png \
        test/hrdps_test_meager_std/GLO30.dem
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

try:
    import contextily as ctx
    _HAVE_CONTEXTILY = True
except ImportError:
    _HAVE_CONTEXTILY = False


# --------------------------------------------------------------------------- #
def plot_map(lat0, lon0, lat1, lon1, out_png=None):
    """
    Save (or show) a map PNG of the two profile endpoints connected by a line.

    Uses a contextily tile basemap when available; falls back to a plain
    axes with gridlines.
    """
    pad = max(abs(lat1 - lat0), abs(lon1 - lon0)) * 0.4 + 0.05

    fig, ax = plt.subplots(figsize=(6, 6))

    if _HAVE_CONTEXTILY:
        import pyproj
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

        x0, y0 = transformer.transform(lon0, lat0)
        x1, y1 = transformer.transform(lon1, lat1)

        # profile line + endpoints
        ax.plot([x0, x1], [y0, y1], '-', color='crimson', lw=2, zorder=5)
        ax.plot([x0, x1], [y0, y1], 'o', color='crimson', ms=8, zorder=6)

        # labels
        offset = max(abs(y1 - y0), abs(x1 - x0)) * 0.04 + 500
        ax.text(x0, y0 + offset, 'A', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='crimson', zorder=7)
        ax.text(x1, y1 + offset, 'B', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='crimson', zorder=7)

        # set extent before adding tiles
        xs = [x0, x1];  ys = [y0, y1]
        pad_m = pad * 111_000
        ax.set_xlim(min(xs) - pad_m, max(xs) + pad_m)
        ax.set_ylim(min(ys) - pad_m, max(ys) + pad_m)

        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto')
        ax.set_axis_off()
    else:
        # plain fallback
        ax.plot([lon0, lon1], [lat0, lat1], '-', color='crimson', lw=2)
        ax.plot([lon0, lon1], [lat0, lat1], 'o', color='crimson', ms=8)
        offset = pad * 0.1
        ax.text(lon0, lat0 + offset, 'A', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='crimson')
        ax.text(lon1, lat1 + offset, 'B', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='crimson')
        ax.set_xlim(min(lon0, lon1) - pad, max(lon0, lon1) + pad)
        ax.set_ylim(min(lat0, lat1) - pad, max(lat0, lat1) + pad)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_title(
        f'Profile: A ({lat0:.4f}°N, {lon0:.4f}°E) → B ({lat1:.4f}°N, {lon1:.4f}°E)',
        fontsize=9,
    )
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f'Saved map: {out_png}')
    else:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
def profile_points(lat0, lon0, lat1, lon1, n):
    """Return n evenly-spaced (lat, lon, dist_km) points along a straight line."""
    lats = np.linspace(lat0, lat1, n)
    lons = np.linspace(lon0, lon1, n)
    dlat_m = (lats - lat0) * 111_000
    mean_lat = np.deg2rad((lat0 + lat1) / 2)
    dlon_m = (lons - lon0) * 111_000 * np.cos(mean_lat)
    dist_km = np.sqrt(dlat_m**2 + dlon_m**2) / 1000
    return lats, lons, dist_km


def sample_dem(dem_file, lats, lons):
    """
    Sample a GeoTIFF DEM at the given lat/lon positions.

    Returns elevation in metres (NaN where outside the DEM extent).
    """
    import rasterio
    coords = list(zip(lons, lats))  # rasterio wants (x=lon, y=lat)
    with rasterio.open(dem_file) as src:
        sampled = np.array(
            list(src.sample(coords, masked=True)), dtype=np.float32
        ).squeeze()
    elev = sampled.astype(float)
    nodata = None
    with rasterio.open(dem_file) as src:
        nodata = src.nodata
    if nodata is not None:
        elev[elev == nodata] = np.nan
    return elev


def build_interpolator(ds, var):
    """
    Build per-level horizontal RegularGridInterpolators.

    Returns
    -------
    interp_fn  : callable(lats, lons) → ndarray (n_vert, n_pts)
    vert_axis  : 1-D array of vertical-coordinate values
    vert_label : str
    is_height  : bool  (True → z in metres; False → pressure in hPa)
    """
    lats = ds['y'].values
    lons = ds['x'].values

    if 'z' in ds.dims:
        vert_key, vert_label, is_height = 'z', 'Height (km)', True
    else:
        vert_key, vert_label, is_height = 'levels', 'Pressure (hPa)', False

    vert_axis = ds[vert_key].values
    data = ds[var].values  # (n_vert, n_lat, n_lon)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[:, ::-1, :]

    interps = [
        RegularGridInterpolator(
            (lats, lons), data[k], method='linear',
            bounds_error=False, fill_value=np.nan,
        )
        for k in range(len(vert_axis))
    ]

    def interp_fn(p_lats, p_lons):
        pts = np.column_stack([p_lats, p_lons])
        return np.array([f(pts) for f in interps])

    return interp_fn, vert_axis, vert_label, is_height


def plot_profile(
    nc_file, lat0, lon0, lat1, lon1,
    var='wet_total', n=300,
    max_height_km=6,
    cmap='RdBu_r',
    dem_file=None,
    out_png=None,
):
    ds = xr.open_dataset(nc_file)

    if var not in ds:
        raise KeyError(f"Variable '{var}' not in dataset. Available: {list(ds.data_vars)}")

    interp_fn, vert_axis, vert_label, is_height = build_interpolator(ds, var)

    p_lats, p_lons, dist_km = profile_points(lat0, lon0, lat1, lon1, n)
    values = interp_fn(p_lats, p_lons)  # (n_vert, n_pts)

    # --- vertical axis ---
    if is_height:
        vert_km = vert_axis / 1000.0
        keep = vert_axis <= max_height_km * 1000
        vert_km = vert_km[keep]
        values  = values[keep]
    elif 'z' in ds.data_vars:
        # Use the mean geopotential height across the profile for each pressure
        # level as the y-axis, preserving the actual model level spacing.
        z_interp_fn, _, _, _ = build_interpolator(ds, 'z')
        z_profile = z_interp_fn(p_lats, p_lons)  # (n_levels, n_pts), metres
        vert_km = z_profile.mean(axis=1) / 1000.0  # one representative height per level

        keep = vert_km <= max_height_km
        vert_km = vert_km[keep]
        values  = values[keep]
        is_height = True
        vert_label = 'Height (km)'
    else:
        # No z available — fall back to pressure axis, surface at bottom
        vert_km = vert_axis[::-1]
        values  = values[::-1]

    # --- DEM ---
    elev_km = None
    if dem_file is not None:
        elev_m  = sample_dem(dem_file, p_lats, p_lons)
        elev_km = elev_m / 1000.0

    # --- plot ---
    _KNOWN_UNITS = {'t': 'K', 'q': 'kg/kg', 'z': 'm', 'pres': 'hPa', 'p': 'hPa', 'e': 'Pa'}
    unit = ds[var].attrs.get('units', '') or _KNOWN_UNITS.get(var, '')

    # Convert pressure variables from Pa → hPa for readability
    if var in ('pres', 'p') and unit in ('Pa', 'hPa'):
        values = values / 100.0
        unit = 'hPa'
    fig, ax = plt.subplots(figsize=(11, 5))

    img = ax.pcolormesh(
        np.tile(dist_km, (len(vert_km), 1)),
        np.tile(vert_km[:, None], (1, n)),
        values,
        shading='auto',
        cmap=cmap,
    )
    fig.colorbar(img, ax=ax, label=f'{var} [{unit}]' if unit else var)

    # Terrain silhouette drawn on top — no cell masking needed, the solid fill
    # paints over any atmosphere cells that sit below the terrain surface.
    if elev_km is not None and is_height:
        ax.fill_between(
            dist_km, 0.0, elev_km,
            color='0.45', zorder=3, label='Terrain (GLO-30)',
        )
        ax.plot(dist_km, elev_km, color='0.2', lw=0.8, zorder=4)
        ax.legend(loc='upper right', fontsize=8)

    ax.set_xlabel('Along-profile distance (km)')
    ax.set_ylabel('Height (km)' if is_height else vert_label)
    if is_height:
        ax.set_ylim(bottom=0)

    src_file = Path(nc_file).name
    ax.set_title(
        f'{var}  |  {src_file}\n'
        f'({lat0:.4f}°N, {lon0:.4f}°E) → ({lat1:.4f}°N, {lon1:.4f}°E)',
        fontsize=9,
    )

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f'Saved: {out_png}')
        map_png = Path(out_png).with_stem(Path(out_png).stem + '_map')
        plot_map(lat0, lon0, lat1, lon1, out_png=str(map_png))
    else:
        plt.show()

    ds.close()
    return values, dist_km, vert_km


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)

    nc_file          = sys.argv[1]
    lat0, lon0, lat1, lon1 = map(float, sys.argv[2:6])
    var      = sys.argv[6]  if len(sys.argv) > 6  else 'wet_total'
    n_pts    = int(sys.argv[7]) if len(sys.argv) > 7 else 300
    out_png  = sys.argv[8]  if len(sys.argv) > 8  else None
    dem_file = sys.argv[9]  if len(sys.argv) > 9  else None

    plot_profile(
        nc_file, lat0, lon0, lat1, lon1,
        var=var, n=n_pts, dem_file=dem_file, out_png=out_png,
    )
