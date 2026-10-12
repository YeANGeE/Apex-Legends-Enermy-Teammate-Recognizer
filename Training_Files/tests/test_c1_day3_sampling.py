import csv
import json
from prepare_c1_day3_recordings import selected_frames


def test_sampling_uses_native_time_not_row_position(tmp_path):
    frames=[{"frame_id":i,"source_present_ns":t,"is_duplicate":duplicate} for i,t,duplicate in
            [(0,0,0),(1,100_000_000,0),(2,1_000_000_000,1),(3,1_100_000_000,0),(4,2_000_000_000,0)]]
    with (tmp_path/'frames.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(frames[0]))
        writer.writeheader();writer.writerows(frames)
    (tmp_path/'metadata.json').write_text(json.dumps({'rb_down_timestamps_ns':[500_000_000],'rb_up_timestamps_ns':[1_500_000_000]}))
    selected=selected_frames(tmp_path,{'preview_source_present_ns':1_000_000_000})
    assert all(not int(f['is_duplicate']) for f in selected)
    assert len(selected)==2
    assert abs(int(selected[0]['source_present_ns'])-int(selected[1]['source_present_ns']))>=750_000_000
    assert 3 in [int(f['frame_id']) for f in selected]
