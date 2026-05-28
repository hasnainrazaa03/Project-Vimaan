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

        out = reconstruct_slot_value(["121", ".", "9"])
        # "." is special-cased to attach directly.
        assert "121.9" in out or out == "121. 9"


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
