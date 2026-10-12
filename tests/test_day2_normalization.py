import pytest
from normalize_c1_day2_export import parse_annotation


def task(quality="usable",scene="real_match",label="teammate"):
    return {"id":1,"annotations":[{"result":[
        {"type":"choices","from_name":"frame_quality","value":{"choices":[quality]}},
        {"type":"choices","from_name":"scene_type","value":{"choices":[scene]}},
        {"type":"rectanglelabels","original_width":1920,"original_height":1080,
         "value":{"rectanglelabels":[label],"x":10,"y":20,"width":20,"height":40}}]}]}


def test_day2_box_geometry_and_identity():
    q,s,boxes,_=parse_annotation(task())
    assert (q,s)==("usable","real_match")
    assert boxes==[("teammate",(.2,.4,.2,.4))]


@pytest.mark.parametrize("quality",["uncertain","discard"])
def test_excluded_status_is_preserved_not_negative(quality):
    q,s,boxes,_=parse_annotation(task(quality))
    assert q==quality and boxes


def test_reject_scene_as_detection_class():
    with pytest.raises(ValueError):
        parse_annotation(task(label="training_range"))


def test_usable_requires_known_scene():
    with pytest.raises(ValueError):
        parse_annotation(task(scene="uncertain"))
