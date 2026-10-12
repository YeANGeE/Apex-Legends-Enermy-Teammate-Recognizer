"""Bounded synthetic tracking tests, no services or pretrained downloads."""
import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from c1_tracking_advanced import AdvancedScreenTracker, CameraCompensator, global_assignment, normalize, CAMERA_PROFILES
from c1_tracking import Track


def detection(x=100,cls=0,score=.8):
    return dict(box=[x,100,x+40,200],score=score,**{'class':cls})


class TrackerTests(unittest.TestCase):
    def setUp(self):self.frame=np.zeros((360,640,3),np.uint8)

    def update(self,tr,t,detections,features=None):
        return tr.update_frame(t,detections,self.frame,features)

    def test_global_assignment_not_greedy(self):
        matrix=[[.90,.80],[.85,None]]
        self.assertEqual(set(global_assignment(matrix)),{(0,1),(1,0)})
        self.assertEqual(global_assignment([[None,None],[None,None]]),[])
        self.assertEqual(global_assignment([]),[])
        self.assertEqual(global_assignment([[]]),[])

    def test_assignment_matches_small_brute_force(self):
        rng=np.random.default_rng(5)
        for _ in range(30):
            matrix=rng.random((3,3)).tolist();matrix[0][1]=None
            best=0.
            for choices in itertools.product(range(-1,3),repeat=3):
                columns=[c for c in choices if c>=0]
                if len(set(columns))!=len(columns):continue
                if any(c>=0 and matrix[r][c] is None for r,c in enumerate(choices)):continue
                best=max(best,sum(matrix[r][c] for r,c in enumerate(choices) if c>=0))
            actual=sum(matrix[r][c] for r,c in global_assignment(matrix))
            self.assertAlmostEqual(actual,best)

    def test_appearance_prevents_position_id_swap(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection(100),detection(150)],[[1,0],[0,1]])
        rows=self.update(tr,20_000_000,[detection(110),detection(140)],[[0,1],[1,0]])
        seen={r['track_id']:r['box'][0] for r in rows if r['position_valid']}
        self.assertEqual(seen,{1:140,2:110})

    def test_camera_transport_preserves_id_and_velocity(self):
        tr=AdvancedScreenTracker(confirm_hits=1,appearance=False)
        self.update(tr,0,[detection()])
        tr.compensate(np.array([[1.,0.,180.],[0.,1.,0.]]))
        rows=self.update(tr,20_000_000,[detection(280)])
        self.assertEqual(rows[0]['track_id'],1)
        self.assertEqual(tr.tracks[1].velocity,[0.]*4)
        self.assertEqual(rows[0]['last_observed_box'],detection(280)['box'])

    def test_memory_reconnect_requires_fresh_class_evidence(self):
        tr=AdvancedScreenTracker()
        for i in range(3):self.update(tr,i*20_000_000,[detection()],[[1,0]])
        rows=self.update(tr,300_000_000,[])
        self.assertEqual(rows[0]['state'],'remembered')
        self.assertIsNone(rows[0]['box'])
        rows=self.update(tr,400_000_000,[detection(170)],[[1,0]])
        self.assertEqual(rows[0]['track_id'],1)
        self.assertIsNone(rows[0]['class_id'])
        self.assertIn('memory_reconnected',tr.events)
        self.update(tr,420_000_000,[detection(170)],[[1,0]])
        rows=self.update(tr,440_000_000,[detection(170)],[[1,0]])
        self.assertEqual(rows[0]['class_id'],0)

    def test_identical_ambiguous_people_not_forced(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection(100),detection(102)],[[1,0],[1,0]])
        rows=self.update(tr,20_000_000,[detection(101)],[[1,0]])
        self.assertIn('ambiguous_assignment',tr.events)
        self.assertFalse(next(r for r in rows if r['track_id']==1)['position_valid'])
        self.assertFalse(next(r for r in rows if r['track_id']==2)['position_valid'])

    def test_opposite_appearance_cannot_steal_memory_id(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection()],[[1,0]])
        rows=self.update(tr,300_000_000,[detection()],[[0,1]])
        self.assertEqual([r['track_id'] for r in rows if r['position_valid']],[2])

    def test_no_appearance_memory_does_not_claim_reid(self):
        tr=AdvancedScreenTracker(confirm_hits=1,appearance=False)
        self.update(tr,0,[detection()])
        rows=self.update(tr,300_000_000,[detection()])
        self.assertEqual([r['track_id'] for r in rows if r['position_valid']],[2])

    def test_native_timestamp_expiry_and_duplicate_rejection(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,123,[detection()],[[1,0]])
        with self.assertRaises(ValueError):self.update(tr,123,[])
        self.assertEqual(tr.timestamp,123)
        rows=self.update(tr,900_000_123,[])
        self.assertEqual(rows,[]);self.assertEqual(tr.gallery,{})

    def test_low_confidence_not_silently_dropped_and_class_conflict(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection(score=.06)],[[1,0]])
        rows=self.update(tr,20_000_000,[detection(cls=1,score=.06)],[[1,0]])
        self.assertEqual(rows[0]['track_id'],1)
        self.assertIsNone(rows[0]['class_id'])
        self.assertTrue(rows[0]['position_valid'])

    def test_flash_does_not_reset_track(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection()],[[1,0]])
        rows=tr.update_frame(20_000_000,[detection()],np.full_like(self.frame,255),[[1,0]])
        self.assertEqual(rows[0]['track_id'],1)
        self.assertNotIn('scene_resets',tr.events)

    def test_zero_or_invalid_features(self):
        self.assertIsNone(normalize([0,0]));self.assertIsNone(normalize([float('nan'),1]))

    def test_tentative_ghost_cannot_veto_confirmed_person(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection()],[[1,0]])
        # Reproduce a provisional duplicate left behind by an earlier frame.
        tr.tracks[2]=Track(2,detection(102)['box'],0)
        tr.gallery[2]=[np.array([1.,0.],np.float32)];tr.next_id=3
        rows=self.update(tr,20_000_000,[detection(101)],[[1,0]])
        seen=[r for r in rows if r['position_valid']]
        self.assertEqual([r['track_id'] for r in seen],[1])
        self.assertNotIn('ambiguous_assignment',tr.events)

    def test_ambiguity_does_not_create_id_storm_and_remains_visible(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection(100),detection(102)],[[1,0],[1,0]])
        for i in range(1,9):
            rows=self.update(tr,i*20_000_000,[detection(101)],[[1,0]])
            seen=[r for r in rows if r['position_valid']]
            self.assertEqual(len(seen),1)
            self.assertIsNone(seen[0]['track_id'])
            self.assertIsNone(seen[0]['class_id'])
            self.assertFalse(seen[0]['association_valid'])
            self.assertIn('birth_deferred',tr.events)
            self.assertNotIn('created',tr.events)
        self.assertEqual(tr.next_id,3)

    def test_partial_duplicate_does_not_spawn_competing_person(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection()],[[1,0]])
        partial=dict(box=[103,105,137,175],score=.15,**{'class':0})
        rows=self.update(tr,20_000_000,[detection(),partial],[[1,0],[1,0]])
        self.assertEqual(tr.next_id,2)
        self.assertEqual(len([r for r in rows if r['position_valid']]),2)
        self.assertEqual([r['track_id'] for r in rows if r['state']=='observed'],[1])
        self.assertEqual([r['track_id'] for r in rows if r['state']=='unassociated'],[None])

    def test_feature_glitch_near_old_person_waits_not_new_id(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection()],[[1,0]])
        rows=self.update(tr,20_000_000,[detection()],[[0,1]])
        seen=[r for r in rows if r['position_valid']]
        self.assertEqual([r['track_id'] for r in seen],[None])
        self.assertEqual(tr.next_id,2)
        rows=self.update(tr,40_000_000,[detection()],[[1,0]])
        self.assertEqual([r['track_id'] for r in rows if r['position_valid']],[1])
        self.assertEqual(tr.next_id,2)

    def test_distinct_new_person_still_gets_track(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection()],[[1,0]])
        rows=self.update(tr,20_000_000,[detection(),detection(400)],[[1,0],[0,1]])
        self.assertEqual([r['track_id'] for r in rows if r['position_valid']],[1,2])

    def test_class_conflict_does_not_poison_gallery(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection()],[[1,0]])
        self.update(tr,20_000_000,[detection(cls=1)],[[.99,.1]])
        self.assertEqual(len(tr.gallery[1]),1)

    def test_missing_raw_detection_is_not_invented(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,0,[detection()],[[1,0]])
        rows=self.update(tr,20_000_000,[])
        self.assertFalse(any(r['position_valid'] for r in rows))
        self.assertEqual(tr.unassociated,[])

    def test_expiry_logs_real_native_age(self):
        tr=AdvancedScreenTracker(confirm_hits=1)
        self.update(tr,10,[detection()],[[1,0]])
        self.update(tr,900_000_010,[])
        self.assertEqual(tr.diagnostics['expired_tracks'],[dict(track_id=1,age_ms=900.,confirmed=True)])


class CameraTests(unittest.TestCase):
    def test_profile_validation_and_phase_timings(self):
        for profile in CAMERA_PROFILES.values():
            camera=CameraCompensator(profile['width'],profile['max_corners'])
            _,info=camera.update(np.zeros((360,640,3),np.uint8),[],0)
            self.assertEqual(info['work_width'],profile['width'])
            self.assertEqual(info['max_corners'],profile['max_corners'])
            self.assertGreaterEqual(info['timings_ms']['total'],info['timings_ms']['prepare'])
            self.assertEqual(info['timings_ms']['flow_forward'],0.)
        with self.assertRaises(ValueError):CameraCompensator(width=10)

    def test_fast_camera_translation_original_coordinates(self):
        rng=np.random.default_rng(11)
        gray=cv2.GaussianBlur(rng.integers(0,256,(540,960),dtype=np.uint8),(3,3),0)
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        moved=cv2.warpAffine(rgb,np.array([[1.,0.,16.],[0.,1.,8.]]),(960,540))
        camera=CameraCompensator(width=480,max_corners=200)
        camera.update(rgb,[],0);matrix,info=camera.update(moved,[],20_000_000)
        self.assertEqual(info['status'],'accepted')
        self.assertAlmostEqual(matrix[0,2],16,delta=1.)
        self.assertAlmostEqual(matrix[1,2],8,delta=1.)

    def test_background_translation(self):
        rng=np.random.default_rng(7)
        gray=rng.integers(0,256,(360,640),dtype=np.uint8)
        gray=cv2.GaussianBlur(gray,(3,3),0)
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        moved=cv2.warpAffine(rgb,np.array([[1.,0.,12.],[0.,1.,6.]]),(640,360))
        camera=CameraCompensator(width=320)
        camera.update(rgb,[],0)
        matrix,info=camera.update(moved,[],20_000_000)
        self.assertEqual(info['status'],'accepted')
        self.assertAlmostEqual(matrix[0,2],12,delta=1.)
        self.assertAlmostEqual(matrix[1,2],6,delta=1.)

    def test_textureless_and_time_gap_are_exposed(self):
        camera=CameraCompensator()
        frame=np.zeros((360,640,3),np.uint8)
        camera.update(frame,[],0)
        matrix,info=camera.update(frame,[],20_000_000)
        self.assertIsNone(matrix);self.assertEqual(info['status'],'rejected')
        matrix,info=camera.update(frame,[],200_000_000)
        self.assertIsNone(matrix);self.assertEqual(info['status'],'timestamp_gap')


if __name__=='__main__':unittest.main()
