import unittest
from c1_tracking import Track
from c1_tracking_advanced import AdvancedScreenTracker


class GeometryTests(unittest.TestCase):
    def track(self, confirmed=True):
        return Track(1, [100.,100.,200.,300.], 0, velocity=[300.,300.,-300.,-300.], confirmed=confirmed)

    def test_prediction_shrink_does_not_reject_plausible_measurement(self):
        detection = {'box': [80.,70.,220.,330.]}
        old = AdvancedScreenTracker(appearance=False)
        candidate = AdvancedScreenTracker(appearance=False, stable_geometry=True)
        self.assertEqual(old.evaluate_pair(self.track(), detection, 100_000_000, False)[1], 'area_gate')
        self.assertIsNotNone(candidate.evaluate_pair(self.track(), detection, 100_000_000, False)[0])

    def test_tiny_fragment_still_rejected(self):
        tracker = AdvancedScreenTracker(appearance=False, stable_geometry=True)
        self.assertEqual(tracker.evaluate_pair(self.track(), {'box':[110.,110.,140.,170.]},
                                              100_000_000, False)[1], 'area_gate')

    def test_dormant_and_tentative_keep_original_gates(self):
        detection = {'box': [80.,70.,220.,330.]}
        old = AdvancedScreenTracker(appearance=False)
        candidate = AdvancedScreenTracker(appearance=False, stable_geometry=True)
        for confirmed, dormant in [(False, False), (True, True)]:
            self.assertEqual(old.evaluate_pair(self.track(confirmed), detection, 100_000_000, dormant),
                             candidate.evaluate_pair(self.track(confirmed), detection, 100_000_000, dormant))

    def test_position_gate_unchanged(self):
        tracker = AdvancedScreenTracker(appearance=False, stable_geometry=True)
        self.assertEqual(tracker.evaluate_pair(self.track(), {'box':[800.,800.,900.,1000.]},
                                              100_000_000, False)[1], 'position_gate')


if __name__ == '__main__': unittest.main()
