# Fake a preview.previewbrowser module with a single member TextLabel for
# back-compatibility. TextLabel was unfortunately imported from add-ons.

import sys
import types
import warnings

from Orange.widgets.utils.label import TextLabel as _TextLabel

warnings.warn(
    "'Orange.canvas.preview' was removed",
    DeprecationWarning,
    stacklevel=2
)

previewbrowser = types.ModuleType("previewbrowser")
previewbrowser.__package__ = "Orange.canvas.preview"
previewbrowser.TextLabel = _TextLabel

sys.modules["Orange.canvas.preview.previewbrowser"] = previewbrowser
