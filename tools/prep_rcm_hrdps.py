#!/usr/bin/env python3
"""
prep_rcm_hrdps.py

For each RCM SLC zip in a directory:
  1. Parse date/time/bbox from metadata/product.xml
  2. Parse incidence angles from metadata/calibration/incidenceAngles.xml
  3. Geocode the incidence angle array to a GeoTIFF (2-band ISCE convention: inc, heading)
  4. Write a RAiDER YAML populated with date, time, bbox, and the incidence angle GeoTIFF

Usage:
    python tools/prep_rcm_hrdps.py <zip_or_dir> [<zip_or_dir> ...] [--template template.yaml] [--res 0.002]

Examples:
    # single zip
    python tools/prep_rcm_hrdps.py test/test_nazko_202605/RCM1_OK3967508_PK4135778_1_3M31_20260425_142325_HH_SLC.zip

    # all zips in a directory
    python tools/prep_rcm_hrdps.py test/test_nazko_202605 --template template.yaml

    # multiple explicit zips
    python tools/prep_rcm_hrdps.py path/to/scene1.zip path/to/scene2.zip
"""

import argparse
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata


_NS = {'r': 'rcmGsProductSchema'}


# --------------------------------------------------------------------------- #
# XML helpers
# --------------------------------------------------------------------------- #

def _open_xml(zf, suffix):
    """Find and parse the first entry in the zip whose path ends with *suffix*."""
    for name in zf.namelist():
        if name.endswith(suffix):
            with zf.open(name) as f:
                return ET.parse(f).getroot()
    raise FileNotFoundError(f'{suffix} not found in {zf.filename}')


# --------------------------------------------------------------------------- #
# product.xml
# --------------------------------------------------------------------------- #

def parse_product_xml(zf):
    """
    Parse metadata/product.xml.

    Returns
    -------
    dict with keys:
        date_str, time_str, end_time_str  — YYYYMMDD / HH:MM:SS strings
        bbox_snwe                          — [S, N, W, E] floats
        pass_dir                           — 'Ascending' | 'Descending'
        look_dir                           — 'right' | 'left'
        heading_deg                        — ISCE heading = -(look azimuth from north) (°)
        tie_points                         — list of (line, pixel, lat, lon)
    """
    root = _open_xml(zf, 'metadata/product.xml')

    # --- timing ---
    first_str  = root.find('.//r:zeroDopplerTimeFirstLine', _NS).text
    center_str = root.find('.//r:zeroDopplerAzimuthTime',   _NS).text
    last_str   = root.find('.//r:zeroDopplerTimeLastLine',  _NS).text
    first_dt   = datetime.strptime(first_str,  '%Y-%m-%dT%H:%M:%S.%fZ')
    center_dt  = datetime.strptime(center_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    last_dt    = datetime.strptime(last_str,   '%Y-%m-%dT%H:%M:%S.%fZ')

    # --- geometry ---
    pass_dir  = root.find('.//r:passDirection',   _NS).text   # Ascending / Descending
    antenna   = root.find('.//r:antennaPointing', _NS).text   # Right / Left
    look_dir  = antenna.lower()

    # --- geolocation tie points ---
    tps = root.findall('.//r:imageTiePoint', _NS)
    lines, pixels, lats, lons = [], [], [], []
    for tp in tps:
        lines.append( float(tp.find('r:imageCoordinate/r:line',          _NS).text))
        pixels.append(float(tp.find('r:imageCoordinate/r:pixel',         _NS).text))
        lats.append(  float(tp.find('r:geodeticCoordinate/r:latitude',   _NS).text))
        lons.append(  float(tp.find('r:geodeticCoordinate/r:longitude',  _NS).text))

    bbox_snwe = [min(lats), max(lats), min(lons), max(lons)]
    tie_points = list(zip(lines, pixels, lats, lons))

    # --- ISCE heading ---
    # Use tie points at one range edge to get flight direction from line ordering.
    # RAiDER's inc_hd_to_enu formula: east = sin(inc)*cos(heading+90) = -sin(inc)*sin(heading)
    # This correctly places the satellite when heading = -look_azimuth (mod 360).
    # look_azimuth = flight_az + 90° for right-looking (satellite 90° clockwise of flight dir).
    # So ISCE heading = -(flight_az + sign*90).
    edge_pts = sorted(
        [(l, la, lo) for l, p, la, lo in tie_points if p == 0.0],
        key=lambda x: x[0],
    )
    if len(edge_pts) >= 2:
        la0, lo0 = edge_pts[0][1],  edge_pts[0][2]
        la1, lo1 = edge_pts[-1][1], edge_pts[-1][2]
        mean_lat = np.deg2rad((la0 + la1) / 2)
        flight_az = np.rad2deg(
            np.arctan2((lo1 - lo0) * np.cos(mean_lat), la1 - la0)
        ) % 360
        sign = 1 if look_dir == 'right' else -1
        heading_deg = -(flight_az + sign * 90) % 360
    else:
        heading_deg = 78.0  # fallback: descending right-looking

    return dict(
        date_str=center_dt.strftime('%Y%m%d'),
        start_time_str=first_dt.strftime('%H:%M:%S'),
        center_time_str=center_dt.strftime('%H:%M:%S'),
        end_time_str=last_dt.strftime('%H:%M:%S'),
        bbox_snwe=bbox_snwe,
        pass_dir=pass_dir,
        look_dir=look_dir,
        heading_deg=heading_deg,
        tie_points=tie_points,
    )


# --------------------------------------------------------------------------- #
# incidenceAngles.xml
# --------------------------------------------------------------------------- #

def parse_incidence_angles(zf):
    """
    Parse metadata/calibration/incidenceAngles.xml.

    Returns
    -------
    inc_by_pixel : ndarray, shape (max_pixel+1,)
        inc_by_pixel[p] is the incidence angle (degrees) at range pixel p.
    """
    root = _open_xml(zf, 'metadata/calibration/incidenceAngles.xml')

    first_pixel = int(root.find('r:pixelFirstAnglesValue', _NS).text)
    step_size   = int(root.find('r:stepSize',              _NS).text)
    angles      = np.array([float(el.text) for el in root.findall('r:angles', _NS)])

    # pixel index for each angle value
    pixel_idx = first_pixel + np.arange(len(angles)) * step_size  # may be descending

    order = np.argsort(pixel_idx)
    pixel_sorted = pixel_idx[order]
    angle_sorted = angles[order]

    max_pixel = int(pixel_sorted[-1])
    all_pixels = np.arange(max_pixel + 1)
    return np.interp(all_pixels, pixel_sorted, angle_sorted)


# --------------------------------------------------------------------------- #
# Geocoding
# --------------------------------------------------------------------------- #

def make_incidence_tif(meta, inc_by_pixel, out_path, res_deg=0.002):
    """
    Geocode the incidence angle array to a regular lat/lon grid and write a
    2-band ISCE-convention GeoTIFF (band 1 = incidence, band 2 = heading).
    """
    S, N, W, E = meta['bbox_snwe']
    heading     = meta['heading_deg']
    tie_points  = meta['tie_points']

    tp_lats  = np.array([la for _, _, la, _  in tie_points])
    tp_lons  = np.array([lo for _, _, _,  lo in tie_points])
    tp_pixels = np.array([p  for _, p,  _,  _  in tie_points])

    max_p = len(inc_by_pixel) - 1
    tp_inc = inc_by_pixel[np.clip(tp_pixels.astype(int), 0, max_p)]

    lons_out = np.arange(W, E, res_deg)
    lats_out = np.arange(N, S, -res_deg)
    Lons, Lats = np.meshgrid(lons_out, lats_out)

    pts = np.column_stack([tp_lons, tp_lats])
    inc_grid = griddata(pts, tp_inc, (Lons, Lats), method='linear')

    nan_mask = np.isnan(inc_grid)
    if nan_mask.any():
        inc_grid[nan_mask] = griddata(pts, tp_inc, (Lons[nan_mask], Lats[nan_mask]), method='nearest')

    heading_grid = np.full_like(inc_grid, heading, dtype='float32')
    transform = from_bounds(W, S, E, N, len(lons_out), len(lats_out))

    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=len(lats_out), width=len(lons_out),
        count=2,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(inc_grid.astype('float32'), 1)
        dst.write(heading_grid, 2)

    print(f'  Wrote incidence GeoTIFF: {out_path}')
    return out_path


# --------------------------------------------------------------------------- #
# YAML
# --------------------------------------------------------------------------- #

def write_yaml(meta, inc_tif_path, template_path, out_dir, interpolate_time='none'):
    """Write a populated RAiDER YAML for the scene."""
    with open(template_path) as f:
        cfg = yaml.safe_load(f)

    S, N, W, E = meta['bbox_snwe']

    cfg['look_dir']      = meta['look_dir']
    cfg['weather_model'] = 'HRDPS'

    cfg['date_group'] = {
        'date_start': None, 'date_end': None, 'date_step': None,
        'date_list': [int(meta['date_str'])],
    }
    cfg['time_group'] = {
        'time':             meta['start_time_str'],
        'end_time':         meta['end_time_str'],
        'interpolate_time': interpolate_time,
    }
    cfg['aoi_group'] = {
        'bounding_box':  [round(S, 4), round(N, 4), round(W, 4), round(E, 4)],
        'geocoded_file': None,
        'lat_file':      None,
        'lon_file':      None,
        'station_file':  None,
    }
    cfg['height_group'] = {
        'dem':             None,
        'use_dem_latlon':  True,
        'height_file_rdr': None,
        'height_levels':   None,
    }
    cfg['los_group'] = {
        'ray_trace':      False,
        'zref':           None,
        'los_file':       str(inc_tif_path),
        'los_convention': 'isce',
        'los_cube':       None,
        'orbit_file':     None,
    }
    cfg['runtime_group']['output_directory'] = str(out_dir)

    out_yaml = out_dir / f"{meta['date_str']}_raider.yaml"
    with open(out_yaml, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'  Wrote YAML: {out_yaml}')
    return out_yaml


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def process_zip(zip_path, template_path, res_deg, interpolate_time='none'):
    out_dir = zip_path.parent
    print(f'\nProcessing: {zip_path.name}')

    with zipfile.ZipFile(zip_path, 'r') as zf:
        meta         = parse_product_xml(zf)
        inc_by_pixel = parse_incidence_angles(zf)

    print(f'  Date/time : {meta["date_str"]}  {meta["start_time_str"]} – {meta["end_time_str"]} UTC  (center {meta["center_time_str"]})')
    print(f'  Bbox SNWE : {[round(v,4) for v in meta["bbox_snwe"]]}')
    print(f'  Pass/look : {meta["pass_dir"]}, {meta["look_dir"]}-looking  '
          f'(look heading {meta["heading_deg"]:.1f}°)')

    inc_tif = out_dir / f"{meta['date_str']}_incidence.tif"
    make_incidence_tif(meta, inc_by_pixel, inc_tif, res_deg=res_deg)
    write_yaml(meta, inc_tif, template_path, out_dir, interpolate_time=interpolate_time)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare RAiDER YAML + incidence-angle GeoTIFF from RCM SLC zip files.',
        epilog=(
            'Pass one or more zip files directly, or a directory to process all '
            'RCM*_HH_SLC.zip files found inside it.'
        ),
    )
    parser.add_argument(
        'inputs', nargs='+', type=Path,
        metavar='ZIP_OR_DIR',
        help='RCM SLC zip file(s) or a directory containing them',
    )
    parser.add_argument('--template', type=Path, default=Path('template.yaml'),
                        help='Path to template.yaml (default: template.yaml)')
    parser.add_argument('--res',      type=float, default=0.002,
                        help='Output grid resolution in degrees (default: 0.002 ≈ 220 m)')
    parser.add_argument('--interpolate-time', dest='interpolate_time',
                        choices=['none', 'center_time', 'azimuth_time_grid'],
                        default='none',
                        help='RAiDER time interpolation method (default: none)')
    args = parser.parse_args()

    zips = []
    for inp in args.inputs:
        if inp.is_dir():
            found = sorted(inp.glob('RCM*_HH_SLC.zip'))
            if not found:
                raise SystemExit(f'No RCM SLC zip files found in {inp}')
            zips.extend(found)
        elif inp.suffix == '.zip':
            if not inp.exists():
                raise SystemExit(f'File not found: {inp}')
            zips.append(inp)
        else:
            raise SystemExit(f'Expected a .zip file or directory, got: {inp}')

    for zp in zips:
        process_zip(zp, args.template, args.res, interpolate_time=args.interpolate_time)

    print('\nDone.')


if __name__ == '__main__':
    main()
