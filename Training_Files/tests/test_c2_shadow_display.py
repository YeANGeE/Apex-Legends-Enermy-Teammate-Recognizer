from c2_shadow_display import ShadowDisplay


def sample(ms, fire):
    return dict(source_present_ns=int(ms*1e6),action=dict(rx=.2,ry=-.1,rb=fire,lb=.2))


def test_display_deduplicates_repeated_suggestions_and_tracks_raw_switches():
    display = ShadowDisplay()
    for item in [sample(0,.8),sample(0,.8),sample(50,.9),sample(100,.2),sample(150,.8)]:
        display.observe(item)
    assert len(display.samples) == 4
    assert display.fire_since_ns == 150_000_000
    assert '切换 2次' in display.text()


def test_source_gap_cannot_count_as_continuous_fire():
    display = ShadowDisplay()
    display.observe(sample(0,.8))
    display.observe(sample(500,.8))
    assert len(display.samples) == 1
    assert display.fire_since_ns == 500_000_000
