"""The training generator's schema, derived from the command table.

Commands are defined once, in ``ML/vimaan_nlu/commands.py``; edit them there.
This module keeps the old ``SCHEMA`` shape (intent -> templates + slots) for
the data scripts that import it.
"""

import os
import sys

_ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ML not in sys.path:
    sys.path.insert(0, _ML)

from vimaan_nlu.commands import training_schema  # noqa: E402

SCHEMA = training_schema()
