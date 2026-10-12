import unittest
import numpy as np
from c1_tracking_advanced import AdvancedScreenTracker, group_duplicate_candidates


def detection(box, cls=0, score=.8):
    return dict(box=box, score=score, **{'class':cls})


class DuplicateTests(unittest.TestCase):
    def test_full_partial_uses_full_box_and_preserves_raw(self):
        raw=[detection([100,100,180,210],score=.9),detection([100,100,180,300],score=.2)]
        candidates,_,audit=group_duplicate_candidates(raw,[[1,0],[1,0]])
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]['box'],raw[1]['box'])
        self.assertEqual(audit[0]['member_indices'],[0,1])
        self.assertNotIn('_class_conflict',raw[0])

    def test_separate_people_and_disagreeing_features_not_grouped(self):
        raw=[detection([100,100,180,300]),detection([200,100,280,300])]
        self.assertEqual(len(group_duplicate_candidates(raw,[[1,0],[1,0]])[0]),2)
        raw[1]=detection([100,100,180,300])
        self.assertEqual(len(group_duplicate_candidates(raw,[[1,0],[0,1]])[0]),2)

    def test_distinct_confirmed_tracks_protected(self):
        raw=[detection([100,100,180,300]),detection([100,100,180,300])]
        self.assertEqual(len(group_duplicate_candidates(raw,[[1,0],[1,0]],{0:10,1:11})[0]),2)

    def test_no_transitive_chain(self):
        raw=[detection([100,100,200,300]),detection([104,100,204,300]),detection([108,100,208,300])]
        # adjacent pairs IoU>0.90; extremes fail, so not all three in one group
        self.assertEqual(len(group_duplicate_candidates(raw,[None,None,None])[0]),2)

    def test_class_conflict_keeps_id_but_not_class_or_gallery(self):
        tracker=AdvancedScreenTracker(confirm_hits=3,group_duplicates=True)
        frame=np.zeros((360,640,3),np.uint8)
        clean=detection([100,100,180,300])
        for i in range(3):rows=tracker.update_frame(i*20_000_000,[clean],frame,[[1,0]])
        self.assertEqual(rows[0]['class_id'],0)
        gallery_size=len(tracker.gallery[1])
        rows=tracker.update_frame(60_000_000,[clean,detection([100,100,180,300],1,.2)],frame,[[1,0],[1,0]])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['track_id'],1)
        self.assertTrue(rows[0]['position_valid'])
        self.assertIsNone(rows[0]['class_id'])
        self.assertEqual(len(tracker.gallery[1]),gallery_size)
        self.assertEqual(tracker.diagnostics['association_candidate_count'],1)
        for i in range(4,7):rows=tracker.update_frame(i*20_000_000,[clean],frame,[[1,0]])
        self.assertEqual(rows[0]['class_id'],0)

    def test_disabled_profile_keeps_original_two_candidates(self):
        tracker=AdvancedScreenTracker(group_duplicates=False)
        raw=[detection([100,100,180,300]),detection([100,100,180,300],1,.2)]
        tracker.update_frame(0,raw,np.zeros((360,640,3),np.uint8),[[1,0],[1,0]])
        self.assertEqual(tracker.diagnostics['association_candidate_count'],2)


if __name__=='__main__':unittest.main()
