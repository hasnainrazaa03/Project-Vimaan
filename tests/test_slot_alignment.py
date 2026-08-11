"""Tests for utils.slot_alignment (PR-D / audit T1.1, T1.2)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))

from utils.slot_alignment import align_offsets_to_labels, find_slot_char_spans  # noqa: E402


class TestFindSlotCharSpans:
    def test_word_boundary_avoids_substring_on_in_one(self):
        # THE bug: "on" must not match inside "one".
        spans = find_slot_char_spans("ap one on", {"state": "on"})
        assert len(spans) == 1
        start, end, name = spans[0]
        assert name == "state"
        assert "ap one on"[start:end] == "on"
        assert start == 7  # the standalone "on", not the "on" at index 3 in "one"

    def test_digit_boundary_avoids_inside_number(self):
        text = "frequency 128.50 on com 1"
        spans = find_slot_char_spans(text, {"com_port": "1", "frequency": "128.50"})
        by = {n: (s, e) for s, e, n in spans}
        cs, ce = by["com_port"]
        assert text[cs:ce] == "1" and cs == 24  # the trailing "1", not inside 128
        fs, fe = by["frequency"]
        assert text[fs:fe] == "128.50"

    def test_missing_value_skipped(self):
        assert find_slot_char_spans("gear up", {"state": "left"}) == []

    def test_empty_or_none(self):
        assert find_slot_char_spans("gear up", {}) == []
        assert find_slot_char_spans("gear up", None) == []


class TestAlignOffsetsToLabels:
    slot_map = {"O": 0, "B-state": 1, "I-state": 2, "B-degrees": 3, "I-degrees": 4}

    def test_special_tokens_ignored(self):
        offsets = [(0, 0), (0, 4), (5, 7), (0, 0)]  # CLS, gear, on, SEP
        labels = align_offsets_to_labels(offsets, [(5, 7, "state")], self.slot_map)
        assert labels == [-100, 0, 1, -100]

    def test_bio_within_multi_token_span(self):
        # "270" tokenized as 2/7/0 over chars 0-3 -> B, I, I
        offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (0, 0)]
        labels = align_offsets_to_labels(offsets, [(0, 3, "degrees")], self.slot_map)
        assert labels == [-100, 3, 4, 4, -100]

    def test_token_overrunning_span_is_O(self):
        # a token (0,5) not fully inside span (0,3) -> O, not a slot
        labels = align_offsets_to_labels([(0, 5)], [(0, 3, "degrees")], self.slot_map)
        assert labels == [0]

    def test_two_separate_slots(self):
        # "com 1 ... 118" -> com_port then frequency, each its own B-
        offsets = [(0, 3), (4, 5), (6, 9)]
        smap = {"O": 0, "B-com_port": 1, "I-com_port": 2, "B-frequency": 3, "I-frequency": 4}
        spans = [(4, 5, "com_port"), (6, 9, "frequency")]
        labels = align_offsets_to_labels(offsets, spans, smap)
        assert labels == [0, 1, 3]  # "com"=O, "1"=B-com_port, "118"=B-frequency
