#! /usr/bin/env python3

import os
import sys
import subprocess
from setuptools import setup, find_packages, Command

NAME = 'orange-widget-base'

VERSION = '3.21.0'
ISRELEASED = False
# full version identifier including a git revision identifier for development
# build/releases (this is filled/updated in `write_version_py`)
FULLVERSION = VERSION

DESCRIPTION = 'Base Widget for Orange Canvas'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
LONG_DESCRIPTION = """
This project implements the base OWWidget class and utilities for use in
Orange Canvas workflows.

Provides:

    * `OWWidget` class
    * `gui` module for building GUI
    * `OWWidgetsScheme` the workflow execution model/bridge.
    ...
  
"""
AUTHOR = 'Bioinformatics Laboratory, FRI UL'
AUTHOR_EMAIL = 'info@biolab.si'
URL = 'http://orange.biolab.si/'
LICENSE = 'GPLv3+'

KEYWORDS = (
    'workflow',
)

CLASSIFIERS = (
    'Development Status :: 4 - Beta',
    'Environment :: X11 Applications :: Qt',
    'Environment :: Console',
    'Environment :: Plugins',
    'Programming Language :: Python',
    'License :: OSI Approved :: '
    'GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
)

INSTALL_REQUIRES = [
    "AnyQt",
    "orange-canvas-core>=0.1.*,<0.2a",
]

EXTRAS_REQUIRE = {
}

ENTRY_POINTS = {
}

DATA_FILES = []


# Return the git revision as a string
def git_version():
    """Return the git revision as a string.

    Copied from numpy setup.py
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, env=env)
        return out.stdout

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION


PACKAGES = find_packages()

# Extra non .py, .{so,pyd} files that are installed within the package dir
# hierarchy
PACKAGE_DATA = {
    "orangewidget": ["icons/*.png", "icons/*.svg"],
    "orangewidget.report": ["icons/*.svg", "*.html"],
    "orangewidget.utils": ["_webview/*.js"],
}


class LintCommand(Command):
    """A setup.py lint subcommand developers can run locally."""
    description = "run code linter(s)"
    user_options = []
    initialize_options = finalize_options = lambda self: None

    def run(self):
        """Lint current branch compared to a reasonable master branch"""
        sys.exit(subprocess.call(r'''
        set -eu
        upstream="$(git remote -v |
                    awk '/[@\/]github.com[:\/]biolab\/orange3[\. ]/{ print $1; exit }')"
        git fetch -q $upstream master
        best_ancestor=$(git merge-base HEAD refs/remotes/$upstream/master)
        .travis/check_pylint_diff $best_ancestor
        ''', shell=True, cwd=os.path.dirname(os.path.abspath(__file__))))


class CoverageCommand(Command):
    """A setup.py coverage subcommand developers can run locally."""
    description = "run code coverage"
    user_options = []
    initialize_options = finalize_options = lambda self: None

    def run(self):
        """Check coverage on current workdir"""
        sys.exit(subprocess.call(r'''
        coverage run --source=Orange -m unittest -v Orange.tests
        echo; echo
        coverage combine
        coverage report
        coverage html &&
            { echo; echo "See also: file://$(pwd)/htmlcov/index.html"; echo; }
        ''', shell=True, cwd=os.path.dirname(os.path.abspath(__file__))))


def setup_package():
    setup(
        name=NAME,
        version=FULLVERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        entry_points=ENTRY_POINTS,
        python_requires=">=3.6",
        zip_safe=False,
    )


if __name__ == '__main__':
    setup_package()
