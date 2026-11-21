import torch

from vimaan_nlu import normalize_aviation_input, postprocess_slots
from vimaan_nlu.commands import LEGACY_INTENTS
from vimaan_nlu.relative import resolve_turn


def family_of(intent: str) -> str:
    """The command a model's intent name means (models from before command
    families say "toggle_engine_2" for what is now "toggle_engine")."""
    legacy = LEGACY_INTENTS.get(intent)
    return legacy[0] if legacy else intent


# Aviation commands are short (the longest of the 15,305 texts the model gate
# scores is 21 word-pieces). A model's own training length comes from its
# manifest (ModelLoader sets tokenizer.model_max_length); this is the fallback
# for a checkpoint without one.
DEFAULT_MAX_LENGTH = 32


def reconstruct_slot_value(tokens):
    if not tokens:
        return ""

    reconstructed = ""
    previous = ""

    for i, token in enumerate(tokens):
        clean_token = token.replace("##", "")

        # A decimal point joins the tokens on both sides: "0", ".", "78" is
        # 0.78. Joining only the left side gave "0. 78", which failed every
        # Mach number and altimeter setting in inches (COM frequencies were
        # rebuilt from the text by the postprocessor, which hid it).
        if clean_token == "." or token.startswith("##") or previous == ".":
            reconstructed += clean_token
        elif i == 0:
            reconstructed = clean_token
        else:
            reconstructed += " " + clean_token
        previous = clean_token

    return reconstructed.strip()


def extract_slots(slot_pred_indices, tokens, slot_map_rev):
    extracted_slots = {}
    current_slot_name = None
    current_slot_tokens = []
    in_value = False  # the last token went into the open value

    for token, slot_idx in zip(tokens, slot_pred_indices):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # A value ends at a word's end: a piece that continues a word in the
        # value joins it, whatever its own tag. v27's first candidate tagged
        # "02" of "024" ("02", "##4") and not the rest, so "turn left zero two
        # four" was a turn of 2 degrees. (A value that starts mid-word, as
        # "2" of "ap2", is left as it was.)
        if in_value and token.startswith("##"):
            current_slot_tokens.append(token)
            continue
        in_value = False

        slot_name_bio = slot_map_rev.get(int(slot_idx), "O")

        if slot_name_bio.startswith("B-"):
            if current_slot_name and current_slot_tokens:
                extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)

            current_slot_name = slot_name_bio[2:]
            current_slot_tokens = [token]
            in_value = True

        elif slot_name_bio.startswith("I-") and current_slot_name:
            slot_name_from_tag = slot_name_bio[2:]
            if slot_name_from_tag == current_slot_name:
                current_slot_tokens.append(token)
                in_value = True

        else:
            if current_slot_name and current_slot_tokens:
                extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)
                current_slot_name = None
                current_slot_tokens = []

    if current_slot_name and current_slot_tokens:
        extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)

    return extracted_slots


TURN_INTENTS = ("turn_by", "set_autopilot_heading")


def turn_or_heading(intent, slots, text, intent_names):
    """Phase L.1: the words, not the model, say whether a number after "left"
    or "right" is a turn or a heading (``vimaan_nlu.relative.resolve_turn``).
    Only for a model that knows turns, so v26 is unchanged. ``text`` is the
    normalized text. Returns (intent, slots), or None to keep the model's."""
    if "turn_by" not in set(intent_names):
        return None
    return resolve_turn(intent, slots, text)


def is_wordpiece(tokenizer) -> bool:
    """WordPiece (DistilBERT: "##" pieces, offsets without spaces), or not
    (Ettin / ModernBERT: byte-level BPE). A tokenizer with no fast backend is
    taken as WordPiece, as every model before the question model's round 4."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    return model is None or type(model).__name__ == "WordPiece"


def extract_slots_by_offsets(slot_pred_indices, offsets, text, slot_map_rev):
    """Like extract_slots, but each value is the text its tokens cover
    (from the first token's start to the last's end): for tokenizers whose
    pieces don't spell the words back (byte-level BPE's "Ġvertical")."""
    from utils.slot_alignment import trim_offsets

    extracted: dict[str, str] = {}
    current, start, end = None, 0, 0

    def close():
        if current is not None:
            extracted[current] = text[start:end].strip()

    for (s, e), slot_idx in zip(trim_offsets(text, offsets), slot_pred_indices):
        if s == e:  # special token
            continue
        tag = slot_map_rev.get(int(slot_idx), "O")
        if tag.startswith("B-"):
            close()
            current, start, end = tag[2:], s, e
        elif tag.startswith("I-") and current == tag[2:]:
            end = e
        elif not tag.startswith("I-"):
            close()
            current = None
    close()
    return extracted


def predict(
    text,
    model,
    tokenizer,
    device,
    intent_map_rev,
    slot_map_rev,
    do_postprocess=True,
    max_length=None,
    pad_to_max_length=False,
):
    """Intent, slots and confidence for one utterance.

    The input is truncated at ``max_length`` (default: the length the model
    was trained at, ``tokenizer.model_max_length`` as set by ModelLoader) and,
    unless ``pad_to_max_length``, not padded: one utterance needs no padding,
    and padding it to a fixed length only adds compute (and, for the int8
    model, shifts the per-tensor quantization range).
    """
    text_normalized = normalize_aviation_input(text)
    if max_length is None:
        limit = getattr(tokenizer, "model_max_length", None)
        max_length = limit if isinstance(limit, int) and 0 < limit <= 512 else DEFAULT_MAX_LENGTH

    wordpiece = is_wordpiece(tokenizer)
    encoding = tokenizer(
        text_normalized,
        padding="max_length" if pad_to_max_length else False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        **({} if wordpiece else {"return_offsets_mapping": True}),
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        _, intent_logits, slot_logits = model(input_ids, attention_mask)

    intent_pred_idx = int(torch.argmax(intent_logits, dim=1).item())
    intent_raw = intent_map_rev[intent_pred_idx]
    probs = torch.softmax(intent_logits, dim=1)[0]
    # Score the command, not the model's label: an older model splits "engage
    # the autopilot" 50/50 between toggle_autopilot_1 and _2; both mean the
    # autopilot (the instance is resolved from the words), so sum them.
    intent_pred = family_of(intent_raw)
    intent_confidence = sum(
        probs[idx].item() for idx, name in intent_map_rev.items() if family_of(name) == intent_pred
    )

    slot_pred_indices = torch.argmax(slot_logits, dim=2)[0].cpu().numpy()
    if wordpiece:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        extracted_slots = extract_slots(slot_pred_indices, tokens, slot_map_rev)
    else:
        offsets = encoding["offset_mapping"][0].tolist()
        extracted_slots = extract_slots_by_offsets(
            slot_pred_indices, offsets, text_normalized, slot_map_rev
        )
    for old, new in (LEGACY_INTENTS[intent_raw][2] if intent_raw in LEGACY_INTENTS else {}).items():
        if old in extracted_slots:
            extracted_slots[new] = extracted_slots.pop(old)

    resolved = turn_or_heading(
        intent_pred, extracted_slots, text_normalized, intent_map_rev.values()
    )
    if resolved is not None:
        intent_pred, extracted_slots = resolved
        # The words chose between the two: the model's confidence is in either.
        intent_confidence = sum(
            probs[idx].item()
            for idx, name in intent_map_rev.items()
            if family_of(name) in TURN_INTENTS
        )

    if do_postprocess:
        extracted_slots = postprocess_slots(extracted_slots, text_normalized, intent_pred)

    return {
        "intent": intent_pred,
        "slots": extracted_slots,
        "confidence": intent_confidence,
        "original_text": text,
        "normalized_text": text_normalized,
    }
