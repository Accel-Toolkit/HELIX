"""Tests for Lorentz boost between lab and beam rest frame."""
import numpy as np
import pytest
from linac_gen.pic.lorentz_boost import boost_to_rest_frame, boost_to_lab_frame


@pytest.fixture
def coords():
    """Sample spatial coordinates (N, 3)."""
    return np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [-1.5, 0.5, 10.0],
        [2.0, -3.0, -5.0],
    ])


def test_rest_frame_z_scaled_by_gamma(coords):
    """z_rest = gamma * z_lab."""
    gamma = 1.05
    rest = boost_to_rest_frame(coords, gamma)
    np.testing.assert_allclose(rest[:, 2], gamma * coords[:, 2], atol=1e-14)


def test_transverse_unchanged_in_boost(coords):
    """x and y must not change under longitudinal boost."""
    gamma = 2.0
    rest = boost_to_rest_frame(coords, gamma)
    np.testing.assert_array_equal(rest[:, 0], coords[:, 0])
    np.testing.assert_array_equal(rest[:, 1], coords[:, 1])


def test_lab_frame_z_divided_by_gamma(coords):
    """z_lab = z_rest / gamma (inverse boost)."""
    gamma = 1.5
    lab = boost_to_lab_frame(coords, gamma)
    np.testing.assert_allclose(lab[:, 2], coords[:, 2] / gamma, atol=1e-14)


def test_round_trip_lab_rest_lab(coords):
    """lab -> rest -> lab must preserve all coordinates."""
    gamma = 1.2
    rest = boost_to_rest_frame(coords, gamma)
    back = boost_to_lab_frame(rest, gamma)
    np.testing.assert_allclose(back, coords, atol=1e-14)


def test_round_trip_rest_lab_rest(coords):
    """rest -> lab -> rest must preserve all coordinates."""
    gamma = 3.0
    lab = boost_to_lab_frame(coords, gamma)
    back = boost_to_rest_frame(lab, gamma)
    np.testing.assert_allclose(back, coords, atol=1e-14)


def test_gamma_one_no_change(coords):
    """gamma=1 (non-relativistic limit) should leave coordinates unchanged."""
    rest = boost_to_rest_frame(coords, 1.0)
    np.testing.assert_array_equal(rest, coords)


def test_boost_does_not_modify_input(coords):
    """Boost functions must not mutate the input array."""
    original = coords.copy()
    _ = boost_to_rest_frame(coords, 2.0)
    np.testing.assert_array_equal(coords, original)
    _ = boost_to_lab_frame(coords, 2.0)
    np.testing.assert_array_equal(coords, original)


def test_known_value():
    """Specific numeric check: z=5.0 mm, gamma=10 -> z_rest=50.0 mm."""
    coords = np.array([[0.0, 0.0, 5.0]])
    rest = boost_to_rest_frame(coords, 10.0)
    assert abs(rest[0, 2] - 50.0) < 1e-14


def test_output_shape(coords):
    """Output shape must match input shape."""
    rest = boost_to_rest_frame(coords, 1.5)
    assert rest.shape == coords.shape
    lab = boost_to_lab_frame(coords, 1.5)
    assert lab.shape == coords.shape
