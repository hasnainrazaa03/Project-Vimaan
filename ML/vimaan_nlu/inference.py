import torch

from vimaan_nlu import correct_numbered_intent, normalize_aviation_input, postprocess_slots
from vimaan_nlu.postprocessor import NUMBERED_INTENT_RE

# Aviation commands are short (the longest in the training set is well under
# 32 word-piece tokens). Padding/truncating to 32 instead of 64 roughly halves
# the per-token compute on CPU with no change in output for real commands,
# since no genuine tokens are ever truncated. Override per call if needed.
DEFAULT_MAX_LENGTH = 32


def reconstruct_slot_value(tokens):
    if not tokens:
        return ""

    reconstructed = ""

    for i, token in enumerate(tokens):
        clean_token = token.replace("##", "")

        if clean_token == ".":
            reconstructed += clean_token
        elif token.startswith("##"):
            reconstructed += clean_token
        else:
            if i == 0:
                reconstructed = clean_token
            else:
                reconstructed += " " + clean_token

    return reconstructed.strip()


def extract_slots(slot_pred_indices, tokens, slot_map_rev):
    extracted_slots = {}
    current_slot_name = None
    current_slot_tokens = []

    for token, slot_idx in zip(tokens, slot_pred_indices):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        slot_name_bio = slot_map_rev.get(int(slot_idx), "O")

        if slot_name_bio.startswith("B-"):
            if current_slot_name and current_slot_tokens:
                extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)

            current_slot_name = slot_name_bio[2:]
            current_slot_tokens = [token]

        elif slot_name_bio.startswith("I-") and current_slot_name:
            slot_name_from_tag = slot_name_bio[2:]
            if slot_name_from_tag == current_slot_name:
                current_slot_tokens.append(token)

        else:
            if current_slot_name and current_slot_tokens:
                extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)
                current_slot_name = None
                current_slot_tokens = []

    if current_slot_name and current_slot_tokens:
        extracted_slots[current_slot_name] = reconstruct_slot_value(current_slot_tokens)

    return extracted_slots


def predict(
    text,
    model,
    tokenizer,
    device,
    intent_map_rev,
    slot_map_rev,
    do_postprocess=True,
    max_length=DEFAULT_MAX_LENGTH,
):
    text_normalized = normalize_aviation_input(text)

    encoding = tokenizer(
        text_normalized,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        _, intent_logits, slot_logits = model(input_ids, attention_mask)

    intent_pred_idx = int(torch.argmax(intent_logits, dim=1).item())
    intent_pred = intent_map_rev[intent_pred_idx]
    probs = torch.softmax(intent_logits, dim=1)[0]
    intent_confidence = probs[intent_pred_idx].item()

    # Trust the spoken instance number over the model's variant choice (fixes
    # engine/AP/FD 1<->2 confusion; defaults to 1 when none is named).
    intent_pred = correct_numbered_intent(intent_pred, text_normalized)

    # The instance is now decided by rule, so the model's job was only to pick
    # the family ("an autopilot command"). Score that: the summed probability
    # of every variant. Otherwise "engage the autopilot" — certain it's the
    # autopilot, split 50/50 on which — falls under the confidence floor.
    family = NUMBERED_INTENT_RE.match(intent_pred or "")
    if family:
        intent_confidence = sum(
            probs[idx].item()
            for idx, name in intent_map_rev.items()
            if (m := NUMBERED_INTENT_RE.match(name)) and m.group(1) == family.group(1)
        )

    slot_pred_indices = torch.argmax(slot_logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

    extracted_slots = extract_slots(slot_pred_indices, tokens, slot_map_rev)

    if do_postprocess:
        extracted_slots = postprocess_slots(extracted_slots, text_normalized, intent_pred)

    return {
        "intent": intent_pred,
        "slots": extracted_slots,
        "confidence": intent_confidence,
        "original_text": text,
        "normalized_text": text_normalized,
    }
