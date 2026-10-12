import numpy as np
from prepare_c2_target_review import longest_candidate_run


def test_target_proposals_break_on_multiple_enemies_and_time_gaps():
    features = np.zeros((5,8,7),np.float32)
    features[:,0] = [.5,.5,.1,.2,.9,1,0]
    features[2,1] = [.7,.5,.1,.2,.8,1,0]
    times = np.array([0,50,100,150,450],np.int64)*1_000_000
    assert longest_candidate_run(features,times) == [(0,0),(1,0)]


def test_target_jump_is_not_silently_assigned_same_identity():
    features = np.zeros((3,8,7),np.float32)
    features[:,0] = [.2,.5,.1,.2,.9,1,0]
    features[2,0,0] = .8
    assert longest_candidate_run(features,np.array([0,50,100])*1_000_000) == [(0,0),(1,0)]
