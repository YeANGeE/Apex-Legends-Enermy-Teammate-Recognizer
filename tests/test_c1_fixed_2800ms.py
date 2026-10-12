import unittest
from prepare_c1_day3_fixed_2800ms import choose_frame


def frame(index,time,duplicate=0):
    return dict(frame_id=index,source_present_ns=time,is_duplicate=duplicate)


class FixedSamplingTests(unittest.TestCase):
    def test_native_time_and_duplicate(self):
        rows=[frame(0,10_000_000_000),frame(100,12_790_000_000),
              frame(101,12_800_000_000,1),frame(102,12_820_000_000),frame(200,14_000_000_000)]
        self.assertEqual(choose_frame(rows)['frame_id'],100)

    def test_short_clip_not_substituted(self):
        self.assertIsNone(choose_frame([frame(0,100),frame(60,2_700_000_100)]))

    def test_gap_rejected(self):
        with self.assertRaises(ValueError):choose_frame([frame(0,0),frame(100,4_000_000_000)])


if __name__=='__main__':unittest.main()
