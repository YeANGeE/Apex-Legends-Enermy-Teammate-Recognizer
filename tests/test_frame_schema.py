import csv

from fps_recorder.writer import write_frames_csv


def test_frames_csv_keeps_source_time_separate_from_video_time(tmp_path):
    path = tmp_path / "frames.csv"
    source_ns = 12_345_678_901
    write_frames_csv(path, [
        {"frame_id": 0, "source_present_ns": source_ns, "video_pts_ns": 0,
         "is_duplicate": 0, "dxgi_present_ticks": 123, "global_frame_id": 10},
        {"frame_id": 1, "source_present_ns": source_ns, "video_pts_ns": 8_333_333,
         "is_duplicate": 1, "dxgi_present_ticks": 123, "global_frame_id": 11},
    ])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_present_ns"] == rows[1]["source_present_ns"]
    assert rows[1]["video_pts_ns"] == "8333333"
    assert rows[1]["is_duplicate"] == "1"
