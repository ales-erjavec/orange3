# Back compatibility stub module.
import warnings

from Orange.widgets.widget import InputSignal, OutputSignal

warnings.warn(
    f"{__name__} is deprecated and will be removed.",
    DeprecationWarning, stacklevel=2
)
