import numpy as np
import pytest

from hrdps_synthetic.generate import los_ground_to_satellite
from hrdps_synthetic.geometry import (ground_range_from_incidence, incidence_field, incidence_from_ground_range,
                                      look_azimuth_deg, scene_for)


def test_incidence_round_trip():
    th = np.linspace(15, 60, 10)
    assert np.allclose(incidence_from_ground_range(ground_range_from_incidence(th)), th, atol=1e-9)


def test_scene_grid(all_aois):
    a = all_aois[0]
    s = scene_for(a, 1234.5, -987.6)
    assert s.shape == (3000, 2000) and s.spacing == 10
    assert s.x0 % 10 == 0 and s.y0 % 10 == 0
    cx, cy = s.center
    assert abs(cx - (a.cx + 1234.5)) <= 5 and abs(cy - (a.cy - 987.6)) <= 5


@pytest.mark.parametrize('heading,far_side', [(348.0, 'east'), (192.0, 'west')])
@pytest.mark.parametrize('theta_c', [20.0, 55.0])
def test_incidence_ramp(all_aois, heading, far_side, theta_c):
    s = scene_for(all_aois[0], 0, 0)
    th = incidence_field(s, theta_c, heading)
    assert abs(float(th[1500, 1000]) - theta_c) < 0.02
    west, east = float(th[1500, 0]), float(th[1500, -1])
    assert (east > west) if far_side == 'east' else (west > east)
    ramp = abs(east - west)
    assert 0.3 < ramp < 3.0, ramp  # ~1-2 deg across 20 km for RCM


def test_los_vector():
    from RAiDER.losreader import inc_hd_to_enu

    e, n, u = los_ground_to_satellite(35.0, 348.0)
    assert abs(e * e + n * n + u * u - 1) < 1e-12
    assert e < 0  # ascending, right-looking: satellite to the west
    assert abs(u - float(inc_hd_to_enu(np.array(35.0), np.array(348.0))[..., 2])) < 1e-12
    assert look_azimuth_deg(192.0) == 282.0
    e, n, u = los_ground_to_satellite(35.0, 192.0)
    assert e > 0  # descending: satellite to the east
