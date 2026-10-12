import numpy as np
import torch
from c2_policy import backward_index,history_indices,detection_features,CombatPolicy
from c2_policy import action_values,ACTION_ORDER
import pytest


def test_backward_alignment_never_selects_future_and_uses_latest_tie():
    times=[100,200,200,300]
    assert backward_index(times,99)==-1
    assert backward_index(times,200)==2
    assert backward_index(times,299)==2


def test_user_fire_and_aim_are_shoulder_buttons_not_triggers():
    row=dict(rx='.3',ry='-.4',rb='1',lb='0',rt='0',lt='1')
    assert ACTION_ORDER==['rx','ry','rb','lb']
    assert action_values(row)==[.3,-.4,1.,0.]
    row.update(rb='0',lb='1',rt='1',lt='0')
    assert action_values(row)==[.3,-.4,0.,1.]


def test_nonbinary_button_label_is_rejected():
    with pytest.raises(ValueError):action_values(dict(rx='0',ry='0',rb='.4',lb='1'))


def test_history_uses_native_time_and_rejects_gaps():
    times=[0,50_000_000,100_000_000,160_000_000]
    assert history_indices(times,2)==[0,1,2]
    assert history_indices(times,3)==[1,2,3]
    assert history_indices([0,500_000_000],1) is None


def test_detection_features_keep_class_hypotheses_and_padding():
    features=detection_features([dict(box=[0,0,1920,1080],score=.9,**{'class':1})]).reshape(8,7)
    assert np.allclose(features[0],[.5,.5,1,1,.9,0,1])
    assert np.count_nonzero(features[1:])==0


def test_policy_output_matches_controller_ranges_and_backpropagates():
    model=CombatPolicy()
    result=model(torch.zeros(2,9,160,288),torch.zeros(2,58))
    assert result.shape==(2,4)
    assert torch.all(result[:,:2].abs()<=1)
    assert torch.all((result[:,2:]>=0)&(result[:,2:]<=1))
    result.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
