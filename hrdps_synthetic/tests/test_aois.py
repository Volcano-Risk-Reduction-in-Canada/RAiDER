import json
import zipfile
from xml.etree import ElementTree as ET

from hrdps_synthetic.aois import (DEFAULT_API_DUMP, NEW_AOIS, REPO_ROOT, SCENE_LENGTH_M, SCENE_WIDTH_M,
                                  VOLCANO_AOIS, load_aois, overlap_groups, write_kmz)


def test_registry(all_aois):
    assert len(all_aois) == len(VOLCANO_AOIS) + len(NEW_AOIS) == 45
    assert len({a.name for a in all_aois}) == 45
    for a in all_aois:
        assert a.dx_range[0] <= a.dx_range[1] and a.dy_range[0] <= a.dy_range[1]
        S, N, W, E = a.weather_bounds()
        assert S < a.lat < N and W < a.lon < E


def test_volcano_polygons_fit_or_are_capped(all_aois):
    """Scene shifts keep each volcano polygon inside, except along axes where it is too big."""
    from shapely.geometry import box
    from shapely.ops import transform as shp_transform
    from hrdps_synthetic.aois import _transformer

    for a in all_aois:
        if a.kind != 'volcano':
            continue
        poly = shp_transform(_transformer(4326, a.epsg).transform, a.polygon_lonlat)
        minx, miny, maxx, maxy = poly.bounds
        for dx in a.dx_range:
            for dy in a.dy_range:
                x0, y0, x1, y1 = a.footprint_utm(dx, dy)
                if maxx - minx <= SCENE_WIDTH_M:
                    assert x0 <= minx + 1e-6 and x1 >= maxx - 1e-6, a.name
                if maxy - miny <= SCENE_LENGTH_M:
                    assert y0 <= miny + 1e-6 and y1 >= maxy - 1e-6, a.name


def test_overlap_groups(all_aois):
    groups = [sorted(g) for g in overlap_groups(all_aois) if len(g) > 1]
    assert ['A4202_Volcano_Garbld', 'A4203_Volcano_Cayley'] in groups
    assert ['A4204_Volcano_Edz_N', 'A4205_Volcano_Edz_S'] in groups
    assert all(not n.startswith('BC_') for g in groups for n in g), 'new centroids must not overlap anything'


def test_geojson_matches_api_dump():
    dump = REPO_ROOT / 'test' / 'HRDPS_testing_August2026' / 'api_dump.json'
    if not dump.exists():
        import pytest
        pytest.skip('API dump not present')
    a = {x.name: x for x in load_aois(DEFAULT_API_DUMP, include_new=False)}
    b = {x.name: x for x in load_aois(dump, include_new=False)}
    for k in a:
        assert a[k].polygon_lonlat.equals(b[k].polygon_lonlat)


def test_kmz_roundtrip(tmp_path, all_aois):
    path = write_kmz(all_aois, tmp_path / 'aois.kmz', {a.name: 'train' for a in all_aois}, 'note')
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read('doc.kml'))
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    marks = root.findall('.//k:Placemark', ns)
    assert len(marks) == 45
    coords = {}
    for m in marks:
        lon, lat, _ = map(float, m.find('.//k:Point/k:coordinates', ns).text.split(','))
        coords[m.find('k:name', ns).text] = (lat, lon)
    for name, _, lat, lon in NEW_AOIS:
        assert abs(coords[name][0] - lat) < 1e-6 and abs(coords[name][1] - lon) < 1e-6
