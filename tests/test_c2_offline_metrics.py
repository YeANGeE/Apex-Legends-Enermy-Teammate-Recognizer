import numpy as np
from evaluate_c2_joint_offline import metrics


def test_button_errors_are_not_hidden_by_overall_accuracy():
    target = np.array([[.3, 0, 1, 0], [-.4, 0, 0, 1], [0, 0, 0, 0]])
    prediction = np.array([[.2, 0, .1, .1], [.2, 0, .9, .9], [0, 0, .1, .1]])
    result = metrics(prediction, target)
    assert result['buttons']['rb']['fp'] == 1
    assert result['buttons']['rb']['fn'] == 1
    assert result['buttons']['lb']['recall'] == 1
    assert result['axes']['rx']['same_direction_fraction'] == .5
    assert result['axes']['ry']['same_direction_fraction'] is None
