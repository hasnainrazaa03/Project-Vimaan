"""Tests for vimaan_nlu.validators (Phase 6A) — declarative slot validation.

Pure-Python, no torch. Also guards that the postprocessor's bounds stay in
sync with the validator definitions (single source of truth).
"""

from vimaan_nlu.validators import (
    SLOT_VALIDATORS,
    Enum,
    Range,
    in_range,
    slot_validator,
    validate_slot,
)


class TestRange:
    def test_contains_integer_bounds(self):
        r = Range(1000, 50000)
        assert r.contains("1000")
        assert r.contains("50000")
        assert r.contains("15000")
        assert not r.contains("999")
        assert not r.contains("50001")

    def test_contains_matches_int_truncation(self):
        # Historical behaviour: integer ranges compared as int(float(num)).
        assert Range(1000, 50000).contains("50000.9")  # int -> 50000, in range
        assert not Range(1000, 50000).contains("50001.0")  # int -> 50001, out
        assert Range(0, 360).contains("360.9")  # int -> 360, in range

    def test_non_numeric_is_not_contained(self):
        assert not Range(0, 360).contains("abc")
        assert not Range(0, 360).contains(None)

    def test_validate_in_range(self):
        value, ok, _ = Range(1000, 50000).validate("12000")
        assert ok and value == "12000"

    def test_validate_reject_out_of_range(self):
        value, ok, reason = Range(1000, 50000).validate("80000")
        assert not ok and value is None and "out of range" in reason

    def test_validate_wrap(self):
        value, ok, reason = Range(0, 360, wrap=True).validate("450")
        assert ok and value == "90" and "wrapped" in reason

    def test_float_range_keeps_decimals(self):
        r = Range(118.0, 137.0, integer=False)
        assert r.contains("121.75")
        value, ok, _ = r.validate("121.75")
        assert ok and value == "121.75"
        assert not r.contains("140.0")


class TestEnum:
    def test_contains_case_insensitive(self):
        e = Enum(("on", "off"))
        assert e.contains("ON")
        assert e.contains(" off ")
        assert not e.contains("maybe")

    def test_validate(self):
        value, ok, _ = Enum(("up", "down")).validate("UP")
        assert ok and value == "up"
        value, ok, reason = Enum(("up", "down")).validate("sideways")
        assert not ok and value is None


class TestSlotRegistry:
    def test_validate_slot_known(self):
        value, ok, _ = validate_slot("altitude", "12000")
        assert ok and value == "12000"

    def test_validate_slot_unknown_passes_through(self):
        value, ok, reason = validate_slot("mystery", "whatever")
        assert ok and value == "whatever" and reason == "no validator"

    def test_in_range_helper(self):
        assert in_range("flight_level", "350")
        assert not in_range("flight_level", "9")
        assert not in_range("flight_level", "431")

    def test_heading_wraps_via_validate_but_contains_is_strict(self):
        # contains() (used by selection) is strict; validate() wraps.
        assert not in_range("degrees", "450")
        value, ok, _ = validate_slot("degrees", "450")
        assert ok and value == "90"

    def test_expected_slots_present(self):
        for name in ("altitude", "degrees", "flight_level", "frequency", "com_port", "state"):
            assert slot_validator(name) is not None


class TestBoundsMatchReadme:
    """The README documents these numeric guards; pin them so a drift is loud."""

    def test_documented_bounds(self):
        assert (SLOT_VALIDATORS["altitude"].lo, SLOT_VALIDATORS["altitude"].hi) == (1000, 50000)
        assert (SLOT_VALIDATORS["degrees"].lo, SLOT_VALIDATORS["degrees"].hi) == (0, 360)
        assert (SLOT_VALIDATORS["flight_level"].lo, SLOT_VALIDATORS["flight_level"].hi) == (10, 430)
        assert (SLOT_VALIDATORS["frequency"].lo, SLOT_VALIDATORS["frequency"].hi) == (118.0, 137.0)
