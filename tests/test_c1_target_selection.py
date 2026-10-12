import json
import unittest
from c1_target_selection import EnemyTargetSelector


def track(i, score=.5, box=(0, 0, 20, 20), **overrides):
    return dict(track_id=i, class_id=0, state='observed', confirmed=True,
                position_valid=True, association_valid=True, score=score,
                last_observed_box=box, **overrides)


class TargetSelectionTests(unittest.TestCase):
    def test_four_tiers(self):
        result=EnemyTargetSelector().update([
            track(4,.2,(0,0,5,5)), track(3,.2),
            track(2,.9,(0,0,5,5)),track(1,.3)],1,100,100)
        self.assertEqual([c['track_id'] for c in result['ranked_candidates']],[1,2,3,4])
        self.assertEqual(result['target_id'],1)
        json.dumps(result)

    def test_hold_switch_and_clear(self):
        s=EnemyTargetSelector()
        self.assertEqual(s.update([track(1,.1)],1,100,100)['reason'],'acquire')
        r=s.update([track(1,.1),track(2,.9)],2,100,100)
        self.assertEqual((r['target_id'],r['reason']),(1,'hold'))
        r=s.update([track(2,.9)],3,100,100)
        self.assertEqual((r['target_id'],r['reason']),(2,'switch'))
        r=s.update([],4,100,100)
        self.assertEqual((r['target_id'],r['reason']),(None,'lost'))
        self.assertEqual(s.update([],5,100,100)['reason'],'idle')

    def test_excludes_invalid_candidates(self):
        bad=[]
        for field,value in [('class_id',1),('class_id',None),('state','predicted'),
                            ('state','tentative'),('confirmed',False),
                            ('position_valid',False),('association_valid',False),
                            ('track_id',None),('score',.049),('score',float('nan'))]:
            t=track(1);t[field]=value;bad.append(t)
        self.assertEqual(EnemyTargetSelector().update(bad,1,100,100)['candidate_count'],0)

    def test_area_clipping_and_tie(self):
        r=EnemyTargetSelector().update([track(2),track(1),track(3,.1,(-100,-100,10,10))],1,100,100)
        self.assertEqual(r['target_id'],1)
        self.assertEqual(r['ranked_candidates'][-1]['area_pixels'],100)

    def test_causal_timestamp(self):
        s=EnemyTargetSelector();s.update([],100,100,100)
        for timestamp in (100,99):
            with self.assertRaises(ValueError):s.update([],timestamp,100,100)


if __name__=='__main__':unittest.main()
