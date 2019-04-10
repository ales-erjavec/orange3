# backwards compatibility stub, this was never
import os
import warnings

warnings.warn(
    "'Orange.canvas.resources' module is deprecated and will be removed "
    "in the future.",
    DeprecationWarning, stacklevel=2
)


# this was never part of exposed api but is imported and used anyway.
def package_dirname(package):
    """Return the directory path where package is located.
    """
    warnings.warn(
        "'package_dirname' is deprecated and will be removed in the future.",
        DeprecationWarning, stacklevel=2,
    )
    if isinstance(package, str):
        package = __import__(package, fromlist=[""])
    filename = package.__file__
    dirname = os.path.dirname(filename)
    return dirname

