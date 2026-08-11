"""Public facade for the joint NLU package.

Lightweight: this module deliberately avoids eager-importing torch.

- `normalization.*` and `postprocessor.*` are pure-Python and imported eagerly.
- `JointIntentAndSlotModel`, `ModelLoader`, and `predict` pull torch /
  transformers and are exposed via PEP 562 lazy attribute resolution.
  Code can still write `from vimaan_nlu import JointIntentAndSlotModel`;
  torch is loaded only on first access of one of those names.

This lets tests, dataset scripts, and the plugin's normalization-only paths
load without pulling a ~600 MB torch wheel.
"""

from .manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    load_manifest,
    write_manifest,
)
from .normalization import (
    PHONETIC_MAP,
    normalize_aviation_input,
    normalize_dataset,
    normalize_dataset_item,
    normalize_slot_value,
)
from .postprocessor import (
    ACTION_STATE_MAP,
    add_implicit_state,
    correct_numbered_intent,
    extract_digit_sequence_frequency,
    extract_numbers_from_text,
    postprocess_slots,
)
from .readback import spell_digits
from .validators import (
    SLOT_VALIDATORS,
    Enum,
    Range,
    in_range,
    slot_validator,
    validate_slot,
)

__all__ = [
    "normalize_aviation_input",
    "normalize_slot_value",
    "normalize_dataset_item",
    "normalize_dataset",
    "PHONETIC_MAP",
    "ACTION_STATE_MAP",
    "postprocess_slots",
    "add_implicit_state",
    "correct_numbered_intent",
    "extract_digit_sequence_frequency",
    "extract_numbers_from_text",
    "SLOT_VALIDATORS",
    "Range",
    "Enum",
    "in_range",
    "slot_validator",
    "validate_slot",
    "spell_digits",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    # The following are lazy-loaded via __getattr__ below — they pull torch.
    "JointIntentAndSlotModel",
    "ModelLoader",
    "predict",
]


_LAZY = {
    "JointIntentAndSlotModel": (".model", "JointIntentAndSlotModel"),
    "ModelLoader": (".model_loader", "ModelLoader"),
    "predict": (".inference", "predict"),
}


def __getattr__(name):
    """PEP 562 lazy attribute resolution for torch-bound symbols."""
    if name in _LAZY:
        import importlib

        module_path, attr = _LAZY[name]
        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr)
        globals()[name] = value  # cache: __getattr__ runs at most once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))
