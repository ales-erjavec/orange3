import sys
import socket
import xmlrpc.client
import itertools
import operator
import re
import io
import hashlib
import html.parser
import urllib.parse
import json
import warnings

from types import SimpleNamespace as namespace
from collections import namedtuple

import pkg_resources
import requests

parse_version = pkg_resources.parse_version

Project = namedtuple(
    "Project",
    ["name",  #: The project name
     "meta"]  #: All project metadata
)


class ProjectMetaData(namespace):
    def __init__(self, name="", version=None, summary=None, description=None,
                 homepage=None, license=None, classifiers=[], keywords=None,
                 requires=[]):
        super(ProjectMetaData, self).__init__(
            name=name, version=version, summary=summary,
            description=description, homepage=homepage, license=license,
            classifiers=classifiers, keywords=keywords, requires=requires
        )

Release = namedtuple(
    "Release",
    ["name",     # type: str  # project name
     "version",  # type: str  # release version
     "urls"]     # type: List[ReleaseUrl]  # list of installable packages
)

# An installable package (distribution) for a particular project's release
ReleaseUrl_ = namedtuple(
    "ReleaseUrl",
    ["filename",  # type: str # filename
     "url",       # type: str # full url
     "digests",    # type: Dict[str, str]  # a hashname -> digestvalue mapping
     # "meta"  # type: namespace  # Any extra metadata (subsume digests)?
    ]
)


class ReleaseUrl(ReleaseUrl_):
    def __new__(cls, filename, url, digests={}):
        return super(ReleaseUrl, cls).__new__(cls, filename, url, digests)

    @property
    def package_type(self):
        if self.filename.endswith(WHEEL_EXT):
            return "bdist_wheel"
        elif self.filename.endswith(SDIST_EXT):
            return "sdist"
        elif self.filename.endswith(EGG_EXT):
            return "bdist_egg"
        elif self.filename.endswith(WININST_EXT):
            return "bdist_wininst"
        else:
            return "S.N.A.F.U"


class SafeTransport(xmlrpc.client.SafeTransport):
    """
    `xmlrpc.client.SafeTransport` with configurable timeout.
    """
    def __init__(self, use_datetime=0, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        super(SafeTransport, self).__init__(use_datetime)
        self._timeout = timeout

    def make_connection(self, *args, **kwargs):
        conn = super(SafeTransport, self).make_connection(*args, **kwargs)
        # assert isinstance(conn, http.client.HTTPSConnection)
        conn.timeout = self._timeout
        return conn


# Legacy pypi json/xmlrpc entry point
PYPI_API = "https://pypi.python.org/pypi"
# Warehouse xmlrpc entry point (presumably) (does not support package_releases
# https://github.com/pypa/pypi-legacy/issues/418; -> only use XMLRPC for
# search and JSONRPC for metadata query)
PYPI_WAREHOUSE_API = "https://pypi.io/pypi"


def pypi_search(searchspec, pypi_client=None):
    """
    Search package distributions available on PyPi using `PyPiXMLRPC`_.

    Parameters
    ----------
    searchspec : dict
        A search spec for the search method `PyPIXmlRpc`_
    pypi_client : Optional[xmlrpc.client.ServerProxy]
        An xmlrpc client. If None the `default_pypi_client()` will be used.

    Returns
    -------
    projects : List[SimpleNamespace[name: str, version: str]]

    .. _`PyPIXmlRpc`: https://wiki.python.org/moin/PyPIXmlRpc
    """
    if pypi_client is None:
        pypi_client = default_pypi_client()

    results = pypi_client.search(searchspec)
    # collect all reported projects and their latest version.
    pacakges = []
    key = operator.itemgetter("name")
    for name, pkgs in itertools.groupby(sorted(results, key=key), key=key):
        # TODO: filter out non PEP 440 versions here?
        pkg = max(pkgs, key=lambda p: parse_version(p["version"]))
        pacakges.append(namespace(**pkg))

    return pacakges


def pypi_query_release_meta(projects, pypi_client=None):
    """
    Query PyPI for project meta data using PyPIXMLRPC

    .. warning::
        deprecated, please do not use

    Parameters
    ----------
    projects : List[Release]
    pypi_client : xmlrpc.client.ServerProxy

    Returns
    -------
    rval : List[Tuple[ProjectMetaData, Release]

    """
    warnings.warn("Deprecated", DeprecationWarning, stacklevel=2)
    if pypi_client is None:
        pypi_client = default_pypi_client()

    multicall = xmlrpc.client.MultiCall(pypi_client)

    for projspec in projects:
        multicall.release_data(projspec.name, projspec.version)
        multicall.release_urls(projspec.name, projspec.version)

    results = list(multicall())
    release_data = results[::2]
    release_urls = results[1::2]

    releases = []
    for release, urls in zip(release_data, release_urls):
        if release and urls:
            # ignore releases without actual source/wheel/egg files,
            # or with empty metadata (deleted from PyPi?).
            urls = [ReleaseUrl(url["filename"], url["url"],
                               digests={"md5": url["md5_digest"]})
                    for url in urls]
            release_meta = namespace(
                name=release["name"],
                version=release["version"],
                summary=release["summary"],
                description=release["description"],
                package_url=release["package_url"],
                home_page=release["home_page"],
                author=release["author"],
                license=release["license"],
                keywords=release["keywords"],
                classifiers=release["classifiers"],
                requires_dist=release.get("requires_dist", []),
                bugtrack_url=release.get("bugtrack_url")
            )
            releases.append(
                (release_meta,
                 Release(release["name"], release["version"], urls))
            )
    return releases


def pypi_json_query_project_meta(projects, pypi_url=PYPI_API, session=None):
    """
    Parameters
    ----------
    projects : List[str]
        List of project names to query
    pypi_url : str
        A PyPiJSONRPC compatible entry point
    session : requests.Session

    Returns
    -------
    res : List[Tuple[ProjectMetaData, List[Release]]

    ..
        List[Future[Release]]
        A list of Futures yielding the result (one for each project)

    """
    if session is None:
        session = requests.Session()

    if not pypi_url.endswith("/"):
        pypi_url = pypi_url + "/"

    rval = []
    for name in projects:
        r = session.get(pypi_url + name + "/json",
                        headers={"Accept": "text/plain",
                                 "Cache-Control": "max-age=600"})
        if r.status_code == 404:
            rval.append(None)
        elif 200 <= r.status_code < 300:
            try:
                meta = r.json()
            except json.JSONDecodeError:
                rval.append(None)
            else:
                projectmeta, releases = pypi_from_json_response(meta)
                rval.append((projectmeta, releases))
    return rval


def pypi_from_json_response(meta):
    """
    Extract relevant project meta data from a PyPiJSONRPC response
    Parameters
    ----------
    meta : dict
        JSON response decoded into python native strucures.

    Returns
    -------
    metadata : Tuple[ProjectMetaData, List[Release]]
    """
    info = meta["info"]
    projmeta = namespace(
        name=info.get("name"),
        version=info.get("version"),
        summary=info.get("summary"),
        description=info.get("description"),
        home_page=info.get("home_page"),
        keywords=info.get("keywords"),
        licence=info.get("licence"),
        author=info.get("author"),
        author_email=info.get("author_mail"),
        classifiers=info.get("classifiers", []),
        bugtrack_url=info.get("bugtrack_url"),
        release_url=info.get("release_url"),
        package_url=info.get("package_url"),
        requires_dist=info.get("requires_dist")
    )

    releases = []
    for version, dists in meta.get("releases", {}).items():
        urls = []
        for dist in dists:
            digests = dist.get("digests", {})
            if not digests and "md5_digest" in dist:
                digests = {"md5": dist["md5_digest"]}
            urls.append(
                ReleaseUrl(dist["filename"], dist["url"], digests=digests))
        releases.append(Release(info["name"], version, urls))
    return projmeta, releases


def default_pypi_client(timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
    return xmlrpc.client.ServerProxy(
        PYPI_API,
        transport=SafeTransport(timeout=timeout)
    )

PYPI_INDEX_URL = "https://pypi.python.org/simple/"
PYPI_WAREHOUSE_INDEX_URL = "https://pypi.io/simple/"


def simple_index_query(project, index_url=PYPI_INDEX_URL, session=None):
    """
    Query a simple package repository (PEP 503) for project releases.

    Parameters
    ----------
    project : str
        Project name to query
    index_url : str
        Base url of the simple index repository
    session : Optional[requests.Session]

    Return
    ------
    result : List[Release] or None
        A list of releases available from the index or None if the index does
        not list the project (i.e. the server responds with a 404 error code)
    """
    if session is None:
        session = requests.Session()

    def normalize(name):
        # pep-503 normalized name
        return re.sub(r"[-_.]+", "-", name).lower()

    if not index_url.endswith("/"):
        index_url = index_url + "/"

    query_url = urllib.parse.urljoin(index_url, normalize(project) + "/")

    resp = session.get(query_url, headers={"Accept": "text/html",
                                           "Cache-Control": "max-age=600"})
    if resp.status_code == 404:
        return None

    resp.raise_for_status()

    urls = parse_simple_index_html(io.StringIO(resp.text), base=index_url)
    # filter out unrecognized filenames
    def isvalid(releaseurl):
        try:
            parse_fname(releaseurl.filename)
            return True
        except ValueError:
            return False
    urls = list(filter(isvalid, urls))

    releases = []

    verkey = lambda url: parse_fname(url.filename).version
    for ver, urls in itertools.groupby(sorted(urls, key=verkey), key=verkey):
        releases.append(Release(project, str(ver), list(urls)))
    return releases


_HASH_PREFIX = tuple("{}=".format(a) for a in hashlib.algorithms_guaranteed)


def release_url_from_link(link):
    parsed = urllib.parse.urlparse(link)
    filename = parsed.path.rsplit("/", 1)[-1]
    # unquote percent encoded special values
    filename = urllib.parse.unquote(filename)

    if not filename.endswith(WHEEL_EXT + SDIST_EXT):
        return None

    if parsed.fragment and parsed.fragment.startswith(_HASH_PREFIX):
        digests = dict([parsed.fragment.split("=", 1)])
    else:
        digests = {}
    return ReleaseUrl(filename, link, digests)


class _HTMLLinkParser(html.parser.HTMLParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.links = []
        self.base = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        elif tag == "base" and "href" in attrs:
            assert self.base is None
            self.base = attrs["base"]


def parse_html_anchor_links(stream):
    """
    Parse and return all links in a HTML5 text stream

    Parameters
    ----------
    stream : file-like

    Returns
    -------
    urls : List[Tuple[str, str]]
    """

    parser = _HTMLLinkParser()
    for chunk in stream:
        parser.feed(chunk)

    base = parser.base
    links = [(lnk, "") for lnk in parser.links]

    if base:
        return [(urllib.parse.urljoin(base[0].attrib["href"], link), text)
                for link, text in links]
    else:
        return links


def parse_simple_index_html(stream, base=None):
    """
    Parse a simple html index (PEP 503) for release links.

    Parameters
    ----------
    stream : file-like
        Simple index html stream
    base : Optional[str]
        Optional base url. If supplied then any relative links are resolved
        relative to base.

    Returns
    -------
    urls : List[ReleaseUrl]

    """
    links = parse_html_anchor_links(stream)
    if base is not None:
        links = [(urllib.parse.urljoin(base, link), text)
                 for link, text in links]
    return [release_url_from_link(link) for link, _ in links
            if release_url_from_link(link) is not None]


def simple_index_group_by_version(urls, project_names=None):
    """
    Group a list of ReleaseUrl's into Release(project, version, urls)

    The project names are not fully identifiable from simple index repo api
    due to wheel filename normalization. This function tries to match
    wheel canonized names to source distribution's (sdist) urls if available
    in the same `urls` list. An extra list of official full project names can
    also be supplied.

    Parameters
    ----------
    urls : List[ReleaseUrls]
        ReleaseUrls to group
    project_names : Optional[List[str]]
        An optional list of full project names.

    Returns
    -------
    releases : List[Release]
        Input `urls` grouped by project names
    """
    def isvalid(releaseurl):
        try:
            parse_fname(releaseurl.filename)
            return True
        except ValueError:
            return False

    urls = list(filter(isvalid, urls))

    releases = []
    # First try collect all projnames from the sdist, and create a mapping
    # to normalized wheel names
    project_names = list(project_names)

    for url in urls:
        parsed = parse_fname(url.filename)
        if isinstance(parsed, SDIST_PARTS) and \
                parsed.name not in project_names:
            project_names.append(parsed.name)

    def pep427_escape(name):
        return re.sub("[^\w\d.]+", "_", name, re.UNICODE)

    whlname_to_proj = {pep427_escape(name): name
                       for name in project_names}

    def match_project_name(releaseurl):
        parsed = parse_fname(releaseurl.filename)
        if isinstance(parsed, SDIST_PARTS):
            return parsed.name
        elif isinstance(parsed, WHEEL_FNAME_PARTS):
            return whlname_to_proj.get(parsed.normalized_name,
                                       parsed.normalized_name)
        else:
            raise TypeError

    def releasekey(releaseurl):
        return (match_project_name(releaseurl),
                parse_version(parse_fname(releaseurl.filename).version))
    for (name, ver), urls in itertools.groupby(sorted(urls, key=releasekey),
                                               key=releasekey):
        releases.append(Release(name, str(ver), list(urls)))
    return releases


SDIST_EXT = (".zip", ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.Z", ".tar")
WHEEL_EXT = (".whl",)
EGG_EXT = (".egg",)
WININST_EXT = (".exe",)


def parse_fname(name):
    if name.endswith(WHEEL_EXT):
        return parse_whl_name(name)
    elif name.endswith(SDIST_EXT):
        return parse_sdist_name(name)
    elif name.endswith(EGG_EXT + WININST_EXT):
        raise ValueError("You really should not be here.")
    else:
        raise ValueError("Unrecognized filename extension:' {}'".format(name))


#: WHEEL_FNAME_PARTS(normalized_name:str, version:str, build:Union[int, None]
#:                   plattag:str)
WHEEL_FNAME_PARTS = namedtuple(
    "WHEEL_NAME_PARTS",
    ["normalized_name",
     "version",
     "build",
     "plattag"]
)
#: SDIST_PARTS(name:str, version:str)
SDIST_PARTS = namedtuple(
    "SDIST_NAME_PARTS",
    ["name",
     "version"]
)


def parse_whl_name(wheelname):
    """
    Parse a wheel filename into parts

    Parameters
    ----------
    wheelname : str
        Wheel filename

    Returns
    -------
    parts : WHEEL_FNAME_PARTS
    """
    assert wheelname.endswith(WHEEL_EXT)
    # Strip extension
    rest = wheelname[:-len(".whl")]
    normalized_distname, _, rest = rest.partition("-")
    if not rest:
        raise ValueError(wheelname)
    version, _, rest = rest.partition("-")
    if not rest:
        raise ValueError(wheelname)

    if rest[0].isdigit():
        buildnum, _, rest = rest.partition("-")
    else:
        buildnum = None
    plattag = rest
    return WHEEL_FNAME_PARTS(normalized_distname, version, buildnum, plattag)


def parse_sdist_name(sdistname):
    """
    Parse an sdist filename into parts

    Parameters
    ----------
    sdistname : str
        A sdist archive filename (as created by sdist distutils command)

    Returns
    -------
    parts : SDIST_PARTS
    """
    assert sdistname.endswith(SDIST_EXT)
    # Strip extension
    for ext in SDIST_EXT:
        if sdistname.endswith(ext):
            rest = sdistname[:-len(ext)]
            break
    else:
        assert False
    # make no effort to detect no pep-0440 compliant version (in particular
    # versions containing '-')
    name, _, version = rest.rpartition("-")
    return SDIST_PARTS(name, version)


class SimpleIndex(object):
    """
    PEP 503 compatible simple package index

    Attributes
    ----------
    name : str
        Human readable name (e.g PyPI, MyPI)
    index_url : str
        The base index url
    session : Optional[requests.Session]
        Optional session instance to use for network access
    """
    def __init__(self, name, index_url, session=None):
        self.name = name
        self.index_url = index_url
        self.session = session

    def query_projects(self, projnames):
        """
        Parameters
        ----------
        projnames :  List[str]

        Returns
        -------
        releases : List[Release]
        """
        if self.session is not None:
            session = self.session
        else:
            session = requests.Session()

        res = []
        for name in projnames:
            try:
                meta, releases = simple_index_query(name, self.index_url, session)
            except requests.HTTPError as err:
                if err.response.status_code == 404:
                    releases = []
                else:
                    raise
            except requests.ConnectionError:
                # TODO: return partial result?
                raise

            res.extend(releases)

        return res

_FlatIndex = namedtuple("FlatIndex", ["name", "index_url"])


class FlatIndex(object):
    """
    A 'flat' package index (i.e. pip's --find-links ${url})

    Attributes
    ----------
    name : str
        A human readable index name/identifier
    index_url : str
        Index url
    """
    def __init__(self, name, index_url, session=None):
        self.name = name
        self.index_url = index_url
        self.session = session

    @property
    def is_local(self):
        return urllib.parse.urlparse(self.index_url).scheme == "file"

    def query_projects(self, projnames):
        """
        Parameters
        ----------
        projnames

        Returns
        -------
        releases : List[Releases]
        """
        if self.session is not None:
            session = self.session
        else:
            session = requests.Session()

        index_url = self.index_url
        if not index_url.endswith("/"):
            index_url += "/"

        res = session.get(index_url,
                          headers={"Cache-Control": "max-age=600"})
        if res.status_code == 404:
            return []
        else:
            res.raise_for_status()

        content = res.text
        urls = parse_simple_index_html(io.StringIO(content), base=index_url)
        urls = [url for url in urls
                if url.package_type in {"sdist", "bdist_wheel"}]
        releases = simple_index_group_by_version(urls, projnames)
        return [r for r in releases if r.name in projnames]


def _conda_plat():
    if sys.platform == "win32":
        ostag = "win"
    elif sys.platform == "darwin":
        ostag = "osx"
    elif sys.platform.startswith("linux"):
        ostag = "linux"
    else:
        raise RuntimeError

    if sys.maxsize == (2 ** 31) - 1:
        btag = "32"
    elif sys.maxsize == (2 ** 63) - 1:
        btag = "64"
    else:
        raise RuntimeError

    return "{}-{}".format(ostag, btag)


class CondaChannel(object):
    def __init__(self, name, channel_url, session=None):
        self.name = name
        self.channel_url = channel_url
        self.session = session

    def query_projects(self, projectnames, conda_plat=None):
        url = self.channel_url
        if conda_plat is None:
            conda_plat = _conda_plat()
        base = self.channel_url
        if not base.endswith("/"):
            base = base + "/"

        url = urllib.parse.urljoin(
            self.channel_url, conda_plat + "/" + "repodata.json.bz2")
        resp = requests.get(url)


