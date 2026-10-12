import pytest
from normalize_c1_day3_export import parse_annotation


def task(quality="usable"):
    return {"predictions":[{"result":[{"value":{"rectanglelabels":["enemy"]}}]}],
            "annotations":[{"id":1,"result":[
                {"from_name":"frame_quality","type":"choices","value":{"choices":[quality]}},
                {"from_name":"scene_tags","type":"choices","value":{"choices":["occlusion","ads"]}},
                {"from_name":"label","to_name":"image","type":"rectanglelabels","original_width":1920,"original_height":1080,
                 "value":{"rectanglelabels":["teammate"],"x":10,"y":20,"width":30,"height":40}}]}]}


def test_human_identity_overrides_predictions_and_multiselect_is_retained():
    quality,tags,boxes,_=parse_annotation(task(),{"image_name":"a.png","width":1920,"height":1080})
    assert quality=="usable" and tags==["ads","occlusion"]
    assert boxes[0]["class_id"]==1
    assert boxes[0]["xywh"]==pytest.approx([.25,.4,.3,.4])


def test_discard_empty_is_not_promoted_to_usable():
    t=task("discard")
    t['annotations'][0]['result']=t['annotations'][0]['result'][:2]
    quality,_,boxes,_=parse_annotation(t,{"image_name":"a.png","width":1920,"height":1080})
    assert quality=="discard" and boxes==[]


def test_missing_quality_and_bad_geometry_fail():
    t=task();t['annotations'][0]['result'].pop(0)
    with pytest.raises(ValueError,match="Missing frame quality"):
        parse_annotation(t,{"image_name":"a.png","width":1920,"height":1080})
    t=task();t['annotations'][0]['result'][-1]['value']['width']=95
    with pytest.raises(ValueError,match="coordinates"):
        parse_annotation(t,{"image_name":"a.png","width":1920,"height":1080})
