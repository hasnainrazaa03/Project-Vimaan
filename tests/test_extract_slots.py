"""Tests for the BIO slot extractor in vimaan_nlu.inference.

The slot extractor itself is pure-Python (lives in inference.py but only
indexes into a dict). It needs `torch` available because inference.py
imports torch at module load — so we gate on the `requires_torch` marker.
"""

import pytest


@pytest.fixture
def slot_map_rev():
    # tiny BIO label set: O + B-/I- for degrees and altitude
    return {
        0: "O",
        1: "B-degrees",
        2: "I-degrees",
        3: "B-altitude",
        4: "I-altitude",
    }


@pytest.mark.requires_torch
class TestReconstructSlotValue:
    def test_empty_tokens(self):
        from vimaan_nlu.inference import reconstruct_slot_value

        assert reconstruct_slot_value([]) == ""

    def test_single_token(self):
        from vimaan_nlu.inference import reconstruct_slot_value

        assert reconstruct_slot_value(["270"]) == "270"

    def test_subword_merging(self):
        from vimaan_nlu.inference import reconstruct_slot_value

        # WordPiece subwords (`##xxx`) must concatenate without spaces.
        out = reconstruct_slot_value(["one", "##two", "##three"])
        assert out == "onetwothree"

    def test_decimal_token_attaches_without_space(self):
        from vimaan_nlu.inference import reconstruct_slot_value

        # A decimal point joins both sides ("0. 78" failed every Mach number
        # and altimeter setting in inches in v18's first gate).
        assert reconstruct_slot_value(["121", ".", "9"]) == "121.9"
        assert reconstruct_slot_value(["0", ".", "78"]) == "0.78"
        assert reconstruct_slot_value(["28", ".", "42"]) == "28.42"
        assert reconstruct_slot_value(["one", "two"]) == "one two"


@pytest.mark.requires_torch
class TestExtractSlots:
    def test_special_tokens_skipped(self, slot_map_rev):
        from vimaan_nlu.inference import extract_slots

        tokens = ["[CLS]", "set", "heading", "270", "[SEP]"]
        preds = [0, 0, 0, 1, 0]
        out = extract_slots(preds, tokens, slot_map_rev)
        assert out == {"degrees": "270"}

    def test_b_then_i_concatenates(self, slot_map_rev):
        from vimaan_nlu.inference import extract_slots

        tokens = ["[CLS]", "climb", "to", "15", "000", "[SEP]"]
        preds = [0, 0, 0, 3, 4, 0]
        out = extract_slots(preds, tokens, slot_map_rev)
        assert out["altitude"] in {"15 000", "15000"}

    def test_two_consecutive_b_tags_split_spans(self, slot_map_rev):
        from vimaan_nlu.inference import extract_slots

        # B-degrees, then a fresh B-altitude resets the span.
        tokens = ["[CLS]", "270", "15000", "[SEP]"]
        preds = [0, 1, 3, 0]
        out = extract_slots(preds, tokens, slot_map_rev)
        assert out == {"degrees": "270", "altitude": "15000"}

    def test_orphan_i_tag_without_b_ignored(self, slot_map_rev):
        from vimaan_nlu.inference import extract_slots

        # I- without preceding B- must NOT crash; should produce empty dict.
        tokens = ["[CLS]", "noise", "[SEP]"]
        preds = [0, 2, 0]
        out = extract_slots(preds, tokens, slot_map_rev)
        assert out == {}

    def test_mismatched_i_tag_skipped(self, slot_map_rev):
        from vimaan_nlu.inference import extract_slots

        # B-degrees followed by I-altitude (wrong slot) — the I should be dropped.
        tokens = ["[CLS]", "270", "300", "[SEP]"]
        preds = [0, 1, 4, 0]
        out = extract_slots(preds, tokens, slot_map_rev)
        assert out == {"degrees": "270"}


@pytest.mark.requires_torch
class TestWholeWords:
    """A value ends at a word's end (phase L.1: v27's first candidate tagged
    "02" of "024" and "turn left zero two four" became a 2-degree turn)."""

    def test_a_piece_continuing_the_value_joins_it(self, slot_map_rev):
        from vimaan_nlu.inference import extract_slots

        tokens = ["[CLS]", "turn", "left", "02", "##4", "[SEP]"]
        for tag in (0, 1, 2, 3):  # whatever the piece's own tag
            out = extract_slots([0, 0, 0, 1, tag, 0], tokens, slot_map_rev)
            assert out == {"degrees": "024"}

    def test_a_value_starting_mid_word_is_unchanged(self, slot_map_rev):
        from vimaan_nlu.inference import extract_slots

        # "ap2": "ap" untagged, "##2" the value.
        out = extract_slots([0, 0, 1, 0, 0], ["[CLS]", "ap", "##2", "on", "[SEP]"], slot_map_rev)
        assert out == {"degrees": "2"}

    def test_a_piece_after_a_skipped_token_does_not_join(self, slot_map_rev):
        from vimaan_nlu.inference import extract_slots

        # B-degrees, then I-altitude (skipped): that word's piece isn't degrees.
        tokens = ["[CLS]", "270", "30", "##0", "[SEP]"]
        out = extract_slots([0, 1, 4, 0, 0], tokens, slot_map_rev)
        assert out == {"degrees": "270"}


@pytest.mark.requires_torch
class TestTurnOrHeading:
    def test_only_for_a_model_that_knows_turns(self):
        from vimaan_nlu.inference import turn_or_heading

        slots = {"degrees": "90"}
        assert (
            turn_or_heading(
                "set_autopilot_heading", slots, "turn right 90", ["set_autopilot_heading"]
            )
            is None
        )
        assert turn_or_heading(
            "set_autopilot_heading", slots, "turn right 90", ["set_autopilot_heading", "turn_by"]
        ) == ("turn_by", {"turn": "right", "turn_degrees": "90"})
