"""AOI registry for the HRDPS synthetic-interferogram generator.

Two kinds of AOI are used:

* **volcano** - the nine volcano targets. Their polygons ship with this folder
  (``volcano_aois.geojson``, extracted from the VRRC API dump
  ``test/HRDPS_testing_August2026/api_dump.json``); either file can be passed
  as ``--aoi-file``. Each has a polygon. A scene
  may be shifted randomly as long as the whole polygon stays inside it. When the
  polygon is wider (or longer) than the scene, the shift on that axis is limited
  to +/- 1 km around the centroid.
* **point** - 36 extra centroids spread across BC so that 10,000 samples can be
  drawn without rendering any (AOI, direction, date pair) twice. A scene may be
  shifted up to +/- 5 km on each axis.

Scenes are north-up in the AOI's UTM zone: 20 km east-west by 30 km north-south.

Run ``python -m hrdps_synthetic.aois --help`` for the KMZ export and the ocean
check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence
from xml.sax.saxutils import escape

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform as shp_transform


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_API_DUMP = PACKAGE_DIR / 'volcano_aois.geojson'  # default AOI file (GeoJSON or VRRC API dump)
DEFAULT_KMZ = PACKAGE_DIR / 'synthetic_aois.kmz'

SCENE_WIDTH_M = 20_000.0  # east-west
SCENE_LENGTH_M = 30_000.0  # north-south
POINT_OFFSET_M = 5_000.0  # max scene-centre shift for point AOIs
WIDE_AOI_OFFSET_M = 1_000.0  # max shift on an axis where a polygon is larger than the scene
WEATHER_MARGIN_KM = 5.0  # weather crop margin around every possible scene position
RAW_MARGIN_DEG = 0.12  # extra margin of the raw (regridded) weather crop

REGIONS = (
    'Vancouver Island, Haida Gwaii',
    'South Coast',
    'Southern Interior',
    'Kootenays, Rockies',
    'Central, Coast Mountains',
    'Northwest',
    'Northeast',
)

# Volcano targets used from the API dump, with the region used to stratify splits.
VOLCANO_AOIS = {
    'A4201_Volcano_Meager': 'South Coast',
    'A4202_Volcano_Garbld': 'South Coast',
    'A4203_Volcano_Cayley': 'South Coast',
    'A4204_Volcano_Edz_N': 'Northwest',
    'A4205_Volcano_Edz_S': 'Northwest',
    'A4207_Volcano_Nazko': 'Central, Coast Mountains',
    'A4208_Volcano_Hoodoo': 'Northwest',
    'A4209_Volcano_LavaFk': 'Northwest',
    'Edgecumbe': 'Northwest',
}

# Extra BC centroids (name, region, lat, lon). Chosen to span BC's terrain and
# climate regimes with no possible scene overlap with each other or with the
# volcano AOIs. Validate with ``--check-land`` before relying on new entries.
NEW_AOIS = (
    ('BC_Strathcona', 'Vancouver Island, Haida Gwaii', 49.70, -125.60),
    ('BC_Arrowsmith', 'Vancouver Island, Haida Gwaii', 49.25, -124.60),
    ('BC_Nimpkish', 'Vancouver Island, Haida Gwaii', 50.20, -126.60),
    ('BC_HaidaGwaii', 'Vancouver Island, Haida Gwaii', 53.45, -132.20),
    ('BC_FraserValley', 'South Coast', 49.22, -121.80),
    ('BC_Manning', 'South Coast', 49.18, -120.85),
    ('BC_Lillooet', 'Southern Interior', 50.70, -121.93),
    ('BC_Kamloops', 'Southern Interior', 50.68, -120.34),
    ('BC_Okanagan', 'Southern Interior', 49.90, -119.55),
    ('BC_Boundary', 'Southern Interior', 49.20, -118.45),
    ('BC_Monashee', 'Southern Interior', 50.40, -118.50),
    ('BC_WellsGray', 'Southern Interior', 51.95, -120.10),
    ('BC_RogersPass', 'Kootenays, Rockies', 51.30, -117.55),
    ('BC_Nelson', 'Kootenays, Rockies', 49.50, -117.30),
    ('BC_Purcells', 'Kootenays, Rockies', 50.20, -116.60),
    ('BC_Cranbrook', 'Kootenays, Rockies', 49.50, -115.77),
    ('BC_ElkValley', 'Kootenays, Rockies', 49.50, -115.05),
    ('BC_KootenayNP', 'Kootenays, Rockies', 50.95, -116.05),
    ('BC_Robson', 'Kootenays, Rockies', 53.10, -119.15),
    ('BC_WilliamsLake', 'Central, Coast Mountains', 52.13, -122.14),
    ('BC_Bowron', 'Central, Coast Mountains', 53.05, -121.25),
    ('BC_PrinceGeorge', 'Central, Coast Mountains', 53.90, -122.75),
    ('BC_AnahimLake', 'Central, Coast Mountains', 52.45, -125.30),
    ('BC_Waddington', 'Central, Coast Mountains', 51.37, -125.26),
    ('BC_BellaCoola', 'Central, Coast Mountains', 52.35, -126.30),
    ('BC_BurnsLake', 'Central, Coast Mountains', 54.23, -125.76),
    ('BC_Smithers', 'Northwest', 54.78, -127.17),
    ('BC_Terrace', 'Northwest', 54.52, -128.60),
    ('BC_DeaseLake', 'Northwest', 58.43, -130.00),
    ('BC_Atlin', 'Northwest', 59.60, -133.40),
    ('BC_Cassiar', 'Northwest', 59.25, -129.30),
    ('BC_NorthRockies', 'Northeast', 58.95, -125.75),
    ('BC_FortNelson', 'Northeast', 58.80, -122.70),
    ('BC_FortStJohn', 'Northeast', 56.25, -120.85),
    ('BC_TumblerRidge', 'Northeast', 55.10, -121.00),
    ('BC_Omineca', 'Northeast', 56.00, -125.00),
)


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=None)
def _transformer(src: int, dst: int) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def to_utm(epsg: int, lon, lat):
    return _transformer(4326, epsg).transform(lon, lat)


def to_lonlat(epsg: int, x, y):
    return _transformer(epsg, 4326).transform(x, y)


@dataclass
class AOI:
    name: str
    region: str
    kind: str  # 'volcano' or 'point'
    lat: float  # centroid
    lon: float
    epsg: int
    cx: float  # centroid, UTM metres
    cy: float
    dx_range: tuple[float, float]  # allowed scene-centre offsets from the centroid (m)
    dy_range: tuple[float, float]
    polygon_lonlat: Optional[Polygon] = None

    # -- footprints ---------------------------------------------------------
    def footprint_utm(self, dx: float = 0.0, dy: float = 0.0,
                      width: float = SCENE_WIDTH_M, length: float = SCENE_LENGTH_M) -> tuple:
        x, y = self.cx + dx, self.cy + dy
        return (x - width / 2, y - length / 2, x + width / 2, y + length / 2)

    def reach_utm(self, width: float = SCENE_WIDTH_M, length: float = SCENE_LENGTH_M) -> tuple:
        """Rectangle covering every possible scene position for this AOI."""
        return (self.cx + self.dx_range[0] - width / 2, self.cy + self.dy_range[0] - length / 2,
                self.cx + self.dx_range[1] + width / 2, self.cy + self.dy_range[1] + length / 2)

    def utm_rect_to_lonlat(self, rect: Sequence[float], n: int = 8) -> Polygon:
        xmin, ymin, xmax, ymax = rect
        t = np.linspace(0.0, 1.0, n + 1)
        xs = np.concatenate([xmin + (xmax - xmin) * t, np.full(n + 1, xmax),
                             xmax - (xmax - xmin) * t, np.full(n + 1, xmin)])
        ys = np.concatenate([np.full(n + 1, ymin), ymin + (ymax - ymin) * t,
                             np.full(n + 1, ymax), ymax - (ymax - ymin) * t])
        lon, lat = to_lonlat(self.epsg, xs, ys)
        return Polygon(zip(lon, lat))

    def footprint_lonlat(self, dx: float = 0.0, dy: float = 0.0, **kw) -> Polygon:
        return self.utm_rect_to_lonlat(self.footprint_utm(dx, dy, **kw))

    def reach_lonlat(self, **kw) -> Polygon:
        return self.utm_rect_to_lonlat(self.reach_utm(**kw))

    # -- weather crops ------------------------------------------------------
    def weather_bounds(self, **kw) -> tuple[float, float, float, float]:
        """SNWE bounds passed to RAiDER's prepareWeatherModel (all scene positions + margin)."""
        W, S, E, N = self.reach_lonlat(**kw).bounds
        dlat = WEATHER_MARGIN_KM / 111.2
        dlon = WEATHER_MARGIN_KM / (111.32 * math.cos(math.radians(self.lat)))
        return (S - dlat, N + dlat, W - dlon, E + dlon)

    def raw_bounds(self, **kw) -> tuple[float, float, float, float]:
        """SNWE bounds of the regridded raw HRDPS file (weather bounds + extra margin)."""
        S, N, W, E = self.weather_bounds(**kw)
        dlon = RAW_MARGIN_DEG / math.cos(math.radians(self.lat))
        return (S - RAW_MARGIN_DEG, N + RAW_MARGIN_DEG, W - dlon, E + dlon)


def _offset_range(pmin: float, pmax: float, c: float, size: float) -> tuple[float, float]:
    """Scene-centre offsets (relative to c) that keep [pmin, pmax] inside a scene of `size`."""
    if pmax - pmin <= size:
        return (pmax - size / 2 - c, pmin + size / 2 - c)
    return (-WIDE_AOI_OFFSET_M, WIDE_AOI_OFFSET_M)


def volcano_aoi(label: str, region: str, polygon: Polygon) -> AOI:
    lon0, lat0 = polygon.centroid.x, polygon.centroid.y
    epsg = utm_epsg(lon0, lat0)
    poly_utm = shp_transform(_transformer(4326, epsg).transform, polygon)
    c = poly_utm.centroid
    minx, miny, maxx, maxy = poly_utm.bounds
    lon, lat = to_lonlat(epsg, c.x, c.y)
    return AOI(
        name=label, region=region, kind='volcano', lat=float(lat), lon=float(lon), epsg=epsg,
        cx=float(c.x), cy=float(c.y),
        dx_range=_offset_range(minx, maxx, c.x, SCENE_WIDTH_M),
        dy_range=_offset_range(miny, maxy, c.y, SCENE_LENGTH_M),
        polygon_lonlat=polygon,
    )


def point_aoi(name: str, region: str, lat: float, lon: float) -> AOI:
    epsg = utm_epsg(lon, lat)
    x, y = to_utm(epsg, lon, lat)
    return AOI(
        name=name, region=region, kind='point', lat=lat, lon=lon, epsg=epsg,
        cx=float(x), cy=float(y),
        dx_range=(-POINT_OFFSET_M, POINT_OFFSET_M), dy_range=(-POINT_OFFSET_M, POINT_OFFSET_M),
    )


def load_volcano_aois(api_dump: Path = DEFAULT_API_DUMP) -> list[AOI]:
    """Volcano polygons from a GeoJSON FeatureCollection (``properties.label``) or a VRRC API dump."""
    with open(api_dump) as f:
        doc = json.load(f)
    if doc.get('type') == 'FeatureCollection':
        by_label = {ft['properties']['label']: ft for ft in doc['features']}
    else:
        by_label = {t['label']: t for t in doc['targets']['body']}
    missing = [label for label in VOLCANO_AOIS if label not in by_label]
    if missing:
        raise KeyError(f'Volcano targets missing from {api_dump}: {missing}')
    return [volcano_aoi(label, region, shape(by_label[label]['geometry']))
            for label, region in VOLCANO_AOIS.items()]


def load_aois(api_dump: Path = DEFAULT_API_DUMP, include_new: bool = True) -> list[AOI]:
    aois = load_volcano_aois(api_dump)
    if include_new:
        aois += [point_aoi(*row) for row in NEW_AOIS]
    return aois


def select_aois(aois: Sequence[AOI], names: Optional[Sequence[str]]) -> list[AOI]:
    """Subset by name; `None` or ['all'] keeps everything."""
    if not names or list(names) == ['all']:
        return list(aois)
    lookup = {a.name: a for a in aois}
    unknown = [n for n in names if n not in lookup]
    if unknown:
        raise KeyError(f'Unknown AOI(s): {unknown}. Known: {sorted(lookup)}')
    return [lookup[n] for n in names]


def overlap_groups(aois: Sequence[AOI]) -> list[list[str]]:
    """Group AOIs whose possible scene footprints can overlap (union-find).

    Grouped AOIs always go into the same train/validation split, so no
    atmosphere over a shared patch of ground leaks between splits.
    """
    parent = {a.name: a.name for a in aois}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    reach = {a.name: a.reach_lonlat() for a in aois}
    for i, a in enumerate(aois):
        for b in aois[i + 1:]:
            if reach[a.name].intersects(reach[b.name]):
                parent[find(a.name)] = find(b.name)
    groups: dict[str, list[str]] = {}
    for a in aois:
        groups.setdefault(find(a.name), []).append(a.name)
    return sorted(groups.values(), key=lambda g: g[0])


# ---------------------------------------------------------------------------
# KMZ export
# ---------------------------------------------------------------------------
def _kml_ring(poly: Polygon) -> str:
    coords = ' '.join(f'{x:.6f},{y:.6f},0' for x, y in poly.exterior.coords)
    return f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>'


def _kml_placemark(aoi: AOI, style: str, split: Optional[str]) -> str:
    rows = [
        ('Region', aoi.region),
        ('Kind', aoi.kind),
        ('Centroid lat, lon', f'{aoi.lat:.5f}, {aoi.lon:.5f}'),
        ('UTM EPSG', str(aoi.epsg)),
        ('Scene', f'{SCENE_WIDTH_M / 1000:.0f} km E-W x {SCENE_LENGTH_M / 1000:.0f} km N-S, north-up (footprint shown centred)'),
        ('Centre shift E-W (km)', f'{aoi.dx_range[0] / 1000:+.1f} to {aoi.dx_range[1] / 1000:+.1f}'),
        ('Centre shift N-S (km)', f'{aoi.dy_range[0] / 1000:+.1f} to {aoi.dy_range[1] / 1000:+.1f}'),
    ]
    if split is not None:
        rows.append(('Planned split', split))
    desc = '<table>' + ''.join(f'<tr><td><b>{escape(k)}</b></td><td>{escape(v)}</td></tr>' for k, v in rows) + '</table>'
    geoms = [f'<Point><coordinates>{aoi.lon:.6f},{aoi.lat:.6f},0</coordinates></Point>',
             _kml_ring(aoi.footprint_lonlat())]
    if aoi.polygon_lonlat is not None:
        geoms.append(_kml_ring(aoi.polygon_lonlat))
    return (f'<Placemark><name>{escape(aoi.name)}</name><styleUrl>#{style}</styleUrl>'
            f'<description><![CDATA[{desc}]]></description>'
            f'<MultiGeometry>{"".join(geoms)}</MultiGeometry></Placemark>')


def kml_document(aois: Sequence[AOI], splits: Optional[dict] = None, note: str = '') -> str:
    splits = splits or {}
    styles = (
        '<Style id="volcano"><IconStyle><color>ff0000ff</color><scale>1.1</scale></IconStyle>'
        '<LineStyle><color>ff0000ff</color><width>2</width></LineStyle><PolyStyle><color>400000ff</color></PolyStyle></Style>'
        '<Style id="point"><IconStyle><color>ff00ffff</color><scale>1.0</scale></IconStyle>'
        '<LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><color>4000ffff</color></PolyStyle></Style>'
    )
    folders = []
    for kind, title in (('volcano', 'Volcano AOIs'), ('point', 'New BC centroids')):
        members = [a for a in aois if a.kind == kind]
        marks = ''.join(_kml_placemark(a, kind, splits.get(a.name)) for a in members)
        folders.append(f'<Folder><name>{title} ({len(members)})</name>{marks}</Folder>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f'<name>HRDPS synthetic interferogram AOIs</name><description>{escape(note)}</description>'
            f'{styles}{"".join(folders)}</Document></kml>')


def write_kmz(aois: Sequence[AOI], path: Path, splits: Optional[dict] = None, note: str = '') -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('doc.kml', kml_document(aois, splits, note))
    return path


# ---------------------------------------------------------------------------
# Ocean check
# ---------------------------------------------------------------------------
def ocean_report(aois: Sequence[AOI], max_frac: float = 0.10, search_km: float = 20.0,
                 step_km: float = 2.5, grid_m: float = 50.0) -> list[dict]:
    """Fraction of each centred footprint that is ocean (GLO-30 orthometric height <= 0.5 m).

    For point AOIs above `max_frac`, the smallest centroid shift (on a `step_km`
    grid within `search_km`) that brings the footprint under the limit is reported.
    """
    import rasterio
    from dem_stitcher.stitcher import stitch_dem
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    out = []
    pad = search_km * 1000 if search_km > 0 else 0.0
    for a in aois:
        fx0, fy0, fx1, fy1 = a.footprint_utm()
        ext = (fx0 - pad, fy0 - pad, fx1 + pad, fy1 + pad)
        W, S, E, N = a.utm_rect_to_lonlat(ext).bounds
        for attempt in range(1, 4):
            try:
                dem, prof = stitch_dem([W - 0.01, S - 0.01, E + 0.01, N + 0.01], dem_name='glo_30',
                                       dst_ellipsoidal_height=False, dst_area_or_point='Area')
                break
            except Exception as e:  # network reads of remote COG tiles
                if attempt == 3:
                    out.append({'name': a.name, 'kind': a.kind, 'ocean_frac': float('nan'), 'error': str(e),
                                'suggest_lat': None, 'suggest_lon': None, 'suggest_frac': None})
                    dem = None
                    break
                time.sleep(30 * attempt)
        if dem is None:
            continue
        nx = int(round((ext[2] - ext[0]) / grid_m))
        ny = int(round((ext[3] - ext[1]) / grid_m))
        dst = np.full((ny, nx), np.nan, dtype=np.float32)
        reproject(dem.astype(np.float32), dst, src_transform=prof['transform'], src_crs=prof['crs'],
                  dst_transform=from_origin(ext[0], ext[3], grid_m, grid_m), dst_crs=f'EPSG:{a.epsg}',
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        ocean = (~np.isfinite(dst)) | (dst <= 0.5)
        integral = np.pad(ocean.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        fw, fl = int(round(SCENE_WIDTH_M / grid_m)), int(round(SCENE_LENGTH_M / grid_m))

        def frac(shift_x_m, shift_y_m):
            c0 = int(round((pad + shift_x_m) / grid_m))
            r0 = int(round((pad - shift_y_m) / grid_m))
            s = integral[r0 + fl, c0 + fw] - integral[r0, c0 + fw] - integral[r0 + fl, c0] + integral[r0, c0]
            return float(s) / (fw * fl)

        f0 = frac(0.0, 0.0)
        row = {'name': a.name, 'kind': a.kind, 'ocean_frac': f0, 'suggest_lat': None, 'suggest_lon': None,
               'suggest_frac': None}
        if f0 > max_frac and a.kind == 'point' and pad > 0:
            steps = np.arange(-search_km, search_km + 1e-9, step_km) * 1000
            cands = sorted(((math.hypot(sx, sy), sx, sy) for sx in steps for sy in steps), key=lambda t: t[0])
            for _, sx, sy in cands:
                f = frac(sx, sy)
                if f <= max_frac:
                    lon, lat = to_lonlat(a.epsg, a.cx + sx, a.cy + sy)
                    row.update(suggest_lat=round(float(lat), 3), suggest_lon=round(float(lon), 3), suggest_frac=f)
                    break
        out.append(row)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='python -m hrdps_synthetic.aois', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--aoi-file', type=Path, default=DEFAULT_API_DUMP,
                   help='volcano polygons: GeoJSON (default, shipped) or a VRRC API dump')
    p.add_argument('--kmz', type=Path, nargs='?', const=DEFAULT_KMZ, default=None,
                   help=f'write a KMZ of all AOIs (default path {DEFAULT_KMZ})')
    p.add_argument('--assume-days', type=int, default=29,
                   help='usable days assumed when labelling the planned train/val split in the KMZ')
    p.add_argument('--seed', type=int, default=0, help='seed for the planned split shown in the KMZ')
    p.add_argument('--check-land', action='store_true', help='report the ocean fraction of every footprint (downloads GLO-30)')
    p.add_argument('--max-ocean', type=float, default=0.10)
    p.add_argument('--aois', nargs='+', default=None, help='restrict --check-land/--list to these AOI names')
    p.add_argument('--list', action='store_true', help='print the AOI table')
    args = p.parse_args(argv)

    all_aois = load_aois(args.aoi_file)
    aois = all_aois
    groups = [g for g in overlap_groups(all_aois) if len(g) > 1]

    if args.list or not (args.kmz or args.check_land):
        print(f'{"name":24s} {"kind":8s} {"region":30s} {"lat":>8s} {"lon":>10s} {"epsg":>6s}  shift E-W / N-S (km)')
        for a in aois:
            print(f'{a.name:24s} {a.kind:8s} {a.region:30s} {a.lat:8.3f} {a.lon:10.3f} {a.epsg:6d}  '
                  f'{a.dx_range[0] / 1e3:+5.1f}..{a.dx_range[1] / 1e3:+5.1f} / {a.dy_range[0] / 1e3:+5.1f}..{a.dy_range[1] / 1e3:+5.1f}')
        print(f'\n{len(aois)} AOIs; overlap groups kept in one split: {groups}')

    if args.kmz:
        from hrdps_synthetic.sampling import planned_aoi_splits

        splits, design = planned_aoi_splits(all_aois, args.assume_days, seed=args.seed)
        note = (f'{len(all_aois)} AOIs. Planned split assumes {args.assume_days} usable days: '
                f'{design.val_dates} validation dates / {design.train_dates} training dates; '
                f'{sum(v == "val" for v in splits.values())} validation AOIs. '
                f'Overlap groups kept together: {groups}')
        path = write_kmz(all_aois, args.kmz, splits, note)
        print(f'Wrote {path}')

    if args.check_land:
        rows = ocean_report(select_aois(all_aois, args.aois), max_frac=args.max_ocean)
        print(f'\n{"name":24s} {"ocean %":>8s}  suggestion')
        bad = 0
        for r in rows:
            sug = ''
            if r.get('error'):
                bad += 1
                sug = f'ERROR {r["error"][:80]}'
            elif r['ocean_frac'] > args.max_ocean:
                bad += 1
                sug = (f'move to {r["suggest_lat"]}, {r["suggest_lon"]} ({100 * r["suggest_frac"]:.1f}% ocean)'
                       if r['suggest_lat'] is not None else 'over limit')
            print(f'{r["name"]:24s} {100 * r["ocean_frac"]:8.1f}  {sug}')
        print(f'\n{bad} AOI(s) above {100 * args.max_ocean:.0f}% ocean or failed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
