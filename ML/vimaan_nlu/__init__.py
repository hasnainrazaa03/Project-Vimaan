from .normalization import (
    normalize_aviation_input,
    normalize_slot_value,
    normalize_dataset_item,
    normalize_dataset,
    PHONETIC_MAP
)
from .postprocessor import (
    postprocess_slots,
    add_implicit_state,
    extract_digit_sequence_frequency,
    extract_numbers_from_text,
    ACTION_STATE_MAP
)
from .model import JointIntentAndSlotModel

__all__ = [
    'normalize_aviation_input',
    'normalize_slot_value',
    'normalize_dataset_item',
    'normalize_dataset',
    'PHONETIC_MAP',
    'ACTION_STATE_MAP',
    
    'postprocess_slots',
    'add_implicit_state',
    'extract_digit_sequence_frequency',
    'extract_numbers_from_text',
    'JointIntentAndSlotModel',
]

# Note: ModelLoader / predict are NOT re-exported here. Importing them
# eagerly would pull torch + transformers on every `import vimaan_nlu`, which
# is unnecessary for lightweight consumers (normalization-only callers, tests).
# Import them explicitly from their submodules when needed:
#   from vimaan_nlu.model_loader import ModelLoader
#   from vimaan_nlu.inference import predict
