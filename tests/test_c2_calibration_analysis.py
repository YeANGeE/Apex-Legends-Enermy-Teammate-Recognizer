import numpy as np
import pytest
from analyze_c2_offline_calibration import backward_resample


def test_calibration_grid_is_causal_and_preserves_latest_tie():
    times = np.array([0,2,2,5])
    axes = np.array([[0,0],[.1,.1],[.2,.2],[.5,.5]])
    result = backward_resample(times,axes,np.array([0,1,2,4,5]))
    assert np.allclose(result,[[0,0],[0,0],[.2,.2],[.2,.2],[.5,.5]])
    with pytest.raises(ValueError):backward_resample(times,axes,np.array([-1]))
