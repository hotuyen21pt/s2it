from asqp_s2it.timing import format_duration


def test_format_duration_as_hours_minutes_seconds():
    assert format_duration(3661.4) == "01:01:01"
    assert format_duration(59.8) == "00:01:00"
    assert format_duration(-1) == "00:00:00"
