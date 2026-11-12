import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tasteq.bench import project_gains
from tasteq.space import EQSpace

RAW = EQSpace("raw")
DCT = EQSpace("dct", n_components=5)


@pytest.mark.parametrize("space", [RAW, DCT])
def test_center_of_cube_is_flat(space: EQSpace) -> None:
    np.testing.assert_allclose(space.to_gains(space.flat), 0.0, atol=1e-12)


def test_raw_corners_hit_gain_limits() -> None:
    np.testing.assert_allclose(RAW.to_gains(np.zeros(10)), -10.0)
    np.testing.assert_allclose(RAW.to_gains(np.ones(10)), 10.0)


@given(arrays(np.float64, (4, 5), elements=st.floats(0, 1)))
def test_dct_gains_stay_in_range(points: np.ndarray) -> None:
    gains = DCT.to_gains(points)
    assert gains.shape == (4, 10)
    assert gains.min() >= -10.0
    assert gains.max() <= 10.0


def test_dct_first_component_is_a_uniform_shift() -> None:
    point = DCT.flat.copy()
    point[0] = 0.6  # +2 dB
    np.testing.assert_allclose(DCT.to_gains(point), 2.0)


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        DCT.to_gains(np.zeros(10))
    with pytest.raises(ValueError, match="n_components"):
        EQSpace("dct", n_components=11)


@given(arrays(np.float64, 10, elements=st.floats(0, 1)))
def test_projection_inverts_raw_mapping(point: np.ndarray) -> None:
    np.testing.assert_allclose(project_gains(RAW, RAW.to_gains(point)), point, atol=1e-9)


def test_projection_recovers_smooth_curves() -> None:
    point = np.array([0.55, 0.7, 0.4, 0.5, 0.52])
    gains = DCT.to_gains(point)
    np.testing.assert_allclose(DCT.to_gains(project_gains(DCT, gains)), gains, atol=1e-9)
