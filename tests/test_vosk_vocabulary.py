"""Every word in the recogniser's grammar must exist in the Vosk model: Vosk
drops a missing word with only a log warning, so a command using it could
never be heard (Tier 1 step B3)."""

import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.path.join(ROOT, "ML", "models", "vosk", "vosk-model-small-en-us-0.15")
KNOWN_MISSING = {"vimaan"}  # the product name; never needed to act

vosk = pytest.importorskip("vosk")


@pytest.mark.skipif(not os.path.isdir(MODEL), reason="Vosk model not downloaded")
def test_every_grammar_word_is_in_the_vosk_model(capfd):
    from vimaan_nlu.stt_vocabulary import load_vocabulary

    words = load_vocabulary()
    vosk.KaldiRecognizer(vosk.Model(MODEL), 16000, json.dumps([*words, "[unk]"]))
    err = capfd.readouterr().err
    missing = {
        line.rsplit("'", 2)[-2] for line in err.splitlines() if "missing in vocabulary" in line
    }
    assert missing <= KNOWN_MISSING, sorted(missing - KNOWN_MISSING)
