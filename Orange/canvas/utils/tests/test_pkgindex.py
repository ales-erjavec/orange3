import unittest
import unittest.mock

import io
import textwrap
from urllib.parse import urljoin

from .. import pypiquery as pkgidx
simple_index_html = """
<html>
<head></head>
<body>
<a href="../pkg/foo-42.42.42-py2.py3-none-any.whl">foo-42.42.42-py2.py3-none-any.whl</a>
<a href="https://example.org/pkgs/bar-1.1.1-py35-cp35m-win32.whl">bar-1.1.1-py35-cp35m-win32.whl</a>
<a href="pkg/baz-2.2.tar.gz">baz-2.2.tar.gz</a>
</body>
</html>
"""

simple_index_links = [
    ("../pkg/foo-42.42.42-py2.py3-none-any.whl",
     "foo-42.42.42-py2.py3-none-any.whl"),
    ("https://example.org/pkgs/bar-1.1.1-py35-cp35m-win32.whl",
     "bar-1.1.1-py35-cp35m-win32.whl"),
    ("pkg/baz-2.2.tar.gz", "baz-2.2.tar.gz")
]

flatindex_html = """
<html>
<body>
<a href="pkg/baz-2.2.tar.gz">baz-2.2.tar.gz<a/>
</body>
</html>
"""

import pip.index


class Test(unittest.TestCase):
    def test_anchor_parse(self):
        data = io.StringIO(simple_index_html)
        links = pkgidx.parse_html_anchor_links(data)
        self.assertSequenceEqual(links, simple_index_links)

        html_base = simple_index_html.replace(
            '<head></head>',
            '<head><base href="http://a.com/i/" /></head>')
        data = io.StringIO(html_base)
        links = pkgidx.parse_html_anchor_links(data)
        expected = [(urljoin("http://a.com/i/", link), text)
                    for link, text in simple_index_links]
        self.assertSequenceEqual(links, expected)

    def test_simple_index_parse(self):
        data = io.StringIO(simple_index_html)
        pkgs = pkgidx.parse_simple_index_html(data)
        self.assertSequenceEqual(
            pkgs,
            [pkgidx.ReleaseUrl(
                filename="foo-42.42.42-py2.py3-none-any.whl",
                url="../pkg/foo-42.42.42-py2.py3-none-any.whl"),
             pkgidx.ReleaseUrl(
                 filename="bar-1.1.1-py35-cp35m-win32.whl",
                 url="https://example.org/pkgs/bar-1.1.1-py35-cp35m-win32.whl"
             ),
             pkgidx.ReleaseUrl(
                 filename="baz-2.2.tar.gz",
                 url="pkg/baz-2.2.tar.gz"
             )]
        )