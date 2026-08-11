"""Tests for utils.dataset_filters.instance_number_ok (PR-D / audit T1.3)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))

from utils.dataset_filters import instance_number_ok  # noqa: E402


class TestInstanceNumberOk:
    def test_keeps_row_with_matching_digit(self):
        assert instance_number_ok("toggle_engine_1", "start engine 1")
        assert instance_number_ok("toggle_engine_2", "engine 2 off")

    def test_keeps_row_with_spelled_number(self):
        assert instance_number_ok("toggle_autopilot_1", "engage autopilot one")
        assert instance_number_ok("toggle_flight_director_2", "flight director two on")

    def test_drops_row_that_lost_its_number(self):
        # the augmentation defect: number dropped -> ambiguous label
        assert not instance_number_ok("toggle_engine_1", "start the engine")
        assert not instance_number_ok("toggle_autopilot_2", "engage the autopilot")

    def test_drops_row_with_wrong_number(self):
        assert not instance_number_ok("toggle_engine_1", "shut down engine 2")

    def test_digit_boundary(self):
        # "1" inside "118" must not count as the instance
        assert not instance_number_ok("toggle_engine_1", "engine tuned to 118")

    def test_non_numbered_intents_always_kept(self):
        assert instance_number_ok("toggle_landing_gear", "gear up")
        assert instance_number_ok("set_autopilot_heading", "heading 270")
        assert instance_number_ok("None", "what time is it")
        assert instance_number_ok(None, "anything")
