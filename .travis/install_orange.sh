foldable pip install -U setuptools pip codecov

pip install numba==0.41.0 llvmlite==0.26.0

# PyQt >= 5.12 distributes WebEngine separately
pip install pyqtwebengine

# Install dependencies sequentially
cat requirements-core.txt \
    requirements-gui.txt \
    requirements-dev.txt \
    requirements-opt.txt \
    requirements-doc.txt |
    while read dep; do
        dep="${dep%%#*}"  # Strip the comment
        [ "$dep" ] &&
            foldable pip install $dep
    done

# Create a source tarball from the git checkout
foldable python setup.py sdist
# Create a binary wheel from the packed source
foldable pip wheel --no-deps -w dist dist/Orange-*.tar.gz
# Install into a testing folder
ORANGE_DIR="$(pwd)"/build/travis-test
mkdir -p "$ORANGE_DIR"
pip install --no-deps --target "$ORANGE_DIR"  dist/Orange-*.whl

cd $TRAVIS_BUILD_DIR
