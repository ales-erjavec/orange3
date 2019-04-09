"""
"""

# If Qt is available (GUI) and Qt5, install backport for PyQt4 imports
try:
    import AnyQt.importhooks
except ImportError:
    pass
else:
    if AnyQt.USED_API == "pyqt5":
        # Make the chosen PyQt version pinned
        from AnyQt.QtCore import QObject
        del QObject

        import pyqtgraph  # import pyqtgraph first so that it can detect Qt5
        del pyqtgraph

        AnyQt.importhooks.install_backport_hook('pyqt4')
    del AnyQt


# A hack that prevents segmentation fault with Nvidia drives on Linux if Qt's
# browser window is shown (seen in https://github.com/spyder-ide/spyder/pull/7029/files)

try:
    import ctypes
    ctypes.CDLL("libGL.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:  # pylint: disable=bare-except
    pass
finally:
    del ctypes
