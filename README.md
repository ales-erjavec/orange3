Orange
======

[![Join the chat at https://gitter.im/biolab/orange3](https://badges.gitter.im/biolab/orange3.svg)](https://gitter.im/biolab/orange3?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![build: passing](https://img.shields.io/travis/biolab/orange3.svg)](https://travis-ci.org/biolab/orange3)
[![codecov](https://codecov.io/gh/biolab/orange3/branch/master/graph/badge.svg)](https://codecov.io/gh/biolab/orange3)

[Orange] is a component-based data mining software. It includes a range of data
visualization, exploration, preprocessing and modeling techniques. It can be
used through a nice and intuitive user interface or, for more advanced users,
as a module for the Python programming language.

This is the latest version of Orange (for Python 3). The deprecated version of Orange 2.7 (for Python 2.7) is still available ([binaries] and [sources]).

[Orange]: https://orange.biolab.si/
[binaries]: https://orange.biolab.si/orange2/
[sources]: https://github.com/biolab/orange


Installing with Miniconda / Anaconda
------------------------------------

Orange requires Python 3.6 or newer.

First, install [Miniconda] for your OS. Create virtual environment for Orange:

    conda create python=3 --name orange

In your Anaconda Prompt add conda-forge to your channels:

    conda config --add channels conda-forge

This will enable access to the latest Orange release. Then install Orange:

    conda install orange

[Miniconda]: https://docs.conda.io/en/latest/miniconda.html

To install the add-ons, follow a similar recipe:

    conda install orange-<addon name>

See specific add-on repositories for details.

Installing with pip
-------------------

To install Orange with pip, run the following.

    # Install some build requirements via your system's package manager
    sudo apt install virtualenv build-essential python3-dev

    # Create a separate Python environment for Orange and its dependencies ...
    virtualenv --python=python3 --system-site-packages orangevenv
    # ... and make it the active one
    source orangevenv/bin/activate

    # Install Qt dependencies for the GUI
    pip install PyQt5 PyQtWebEngine

    # Install Orange
    pip install "orange>=4"

#### Upgrading from Orange3 to Orange

If upgrading an exiting environment with orange3; uninstall Orange3 first

    pip uninstall orange3
    pip install "orange>=4"

Starting Orange GUI
-------------------

To start Orange GUI from the command line, run:

    orange-canvas
    # or
    python -m Orange.canvas

Append `--help` for a list of program options.
