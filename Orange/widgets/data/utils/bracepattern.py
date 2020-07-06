"""
A simplified brace pattern like expansion.
"""
import itertools
import math
import re
from typing import NamedTuple, List, Union, Sequence, Iterable, Optional


class String(NamedTuple):
    string: str

    def expand(self):
        return [self.string]


class CommaPattern(NamedTuple):
    string: str

    def expand(self):
        return split_escaped(self.string, ",")


class IntRangePattern(NamedTuple):
    string: str

    def expand(self):
        return list(parse_range_expand(self.string))


PatternExp = Union[String, CommaPattern, IntRangePattern]


class BracePatternError(ValueError): ...


class BracePatternSyntaxError(BracePatternError): ...


def split_escaped(string: str, split: str) -> Sequence[str]:
    r"""
    Split `string` on `split` substring ignoring '\\' escaped

    >>> split_escaped(r"a,b\,c", ",")
    ['a', 'b\\,c']
    >>> split_escaped(r"a,b\\,c", ",")
    ['a', 'b', 'c']
    """
    return re.split(r"(?<!\\)(?:\\\\)*" + re.escape(split), string)


def parse_range_expand(string: str) -> List[str]:
    """
    >>> parse_range_expand("1..3")
    ['1', '2', '3']
    >>> parse_range_expand("01..03")  # zero padding
    ['01', '02', '03']
    """
    range_re = re.compile(r"^(?P<start>\d+)\.\.(?P<end>\d+)$")
    match = range_re.match(string)
    if match is None:
        raise BracePatternError("")
    s_start = match.group("start")
    s_end = match.group("end")
    start, end = int(s_start), int(s_end)
    step = -1 if end < start else 1
    padd = s_start != "0" and s_start.startswith("0") \
           or s_end != "0" and s_end.startswith("0")

    if padd:
        places = math.ceil(math.log10(max(abs(start), abs(end)) + 1))
        places = max(places, len(s_start.lstrip("-")), len(s_end.lstrip("-")))
        fmt = f"{{:0{places}}}"
    else:
        fmt = "{}"
    return list(map(fmt.format, range(start, end + step, step)))


def parse_brace_pattern(pattern: str) -> Sequence[PatternExp]:
    """
    Parse a brace expansion pattern returning a sequence of pattern expressions

    >>> parse_brace_pattern("a{b,c}{1..2}")
    [String(string='a'), CommaPattern(string='b,c'), IntRangePattern(string='1..2')]
    """
    pos = 0
    open_re = re.compile(r"(?<!\\)(?:\\\\)*\{")
    closing_re = re.compile(r"(?<!\\)(?:\\\\)*}")

    seq: List[PatternExp] = []
    while pos < len(pattern):
        match_start = open_re.search(pattern, pos)
        if match_start is not None:
            span_start = match_start.span()
            if span_start[0] != pos:
                seq.append(String(pattern[pos: span_start[0]]))

            match_end = closing_re.search(pattern, span_start[1])
            if match_end is not None:
                span_end = match_end.span()
                expand_pattern = pattern[span_start[1]: span_end[0]]
                pos = span_end[1]
                try:
                    parse_range_expand(expand_pattern)
                    seq.append(IntRangePattern(expand_pattern))
                except BracePatternError:
                    seq.append(CommaPattern(expand_pattern))
            else:
                raise BracePatternSyntaxError(
                    f"Unmatched braces in {pattern!r}")
        else:
            seq.append(String(pattern[pos:]))
            pos = len(pattern)
    return seq


def expand_pattern(
        pattern: str, count: int, *, fill_string=None, trim_to_count=False
) -> Sequence[str]:
    """
    Expand the `pattern` string to `count` strings.

    If `fill_string` is not None.
    """
    tokens = parse_brace_pattern(pattern)
    tokens_ = []
    for t in tokens:
        if isinstance(t, String):
            tokens_.append(t.expand() * count)
        else:
            strings = t.expand()
            if len(strings) < count:
                if fill_string is None:
                    raise ValueError(f"Expansion of {t} produced fewer then "
                                     f"{count} values.")
                else:
                    strings = strings + ([fill_string] * (count - len(strings)))
            elif len(strings) > count:
                # ??? Raise an error or trim strings to count length
                if trim_to_count:
                    strings = strings[:count]
                else:
                    raise ValueError(f"Expansion of {t} produced more then "
                                     f"{count} values.")
            tokens_.append(strings)
    return ["".join(s) for s in zip(*tokens_)]


def infer_pattern(strings: Iterable[str]) -> str:
    """
    Infer a brace pattern such that `expand_pattern(..., len(sequence))` can
    round trip.

    Parameters
    ----------
    strings: Iterable[str]

    Returns
    -------
    pattern: str

    Examples
    --------
    >>> infer_pattern(["a b", ["a c"]])
    "a {b,c}"
    >>> infer_pattern(["a 1", ["a 2"]])
    "a {1..2}"
    """
    parts = list(map(lambda n: re.split(r"(\W)", n), strings))
    N = max(map(len, parts), default=0)
    # pad to equal len
    parts = [part + ([""] * (N - len(part))) for part in parts]
    pattern = []
    for els in zip(*parts):
        if len(set(els)) == 1:
            pattern.append(els[0])
        else:
            int_range = maybe_int_range(els)
            if int_range:
                pattern.append(int_range)
            else:
                pattern.append("{" + (",".join(p.replace(",", r"\,") for p in els)) + "}")
    return "".join(pattern)


def maybe_int_range(strings: Sequence[str]) -> Optional[str]:
    """
    Given a sequence of strings infer a equivalent int brace expansion pattern.

    >>> maybe_int_range(["1", "3"])
    '{1..2}'
    >>> maybe_int_range(["8", "9", "10"])
    '{8..10}'
    >>> maybe_int_range(["08", "09", "10"])  # zero padding
    '{08..10}'
    """
    try:
        values = [int(s) for s in strings]
    except ValueError:
        return None
    if len(strings) == 1:
        return None
    it1, it2 = itertools.tee(values)
    diffs = set()
    lens = set()
    next(it2)
    for v1, v2 in zip(it1, it2):
        diffs.add(int(v1) - int(v2))
    if len(diffs) != 1:
        return None
    step = diffs.pop()
    if step not in (-1, 1):
        return None
    padded = False
    for v in strings:
        if v != "0" and v.startswith("0"):
            padded = True
            lens.add(len(v))
    if padded and len(lens) != 1:
        return None

    if padded:
        padd = lens.pop()
        return f"{{{values[0]:0{padd}}...{values[-1]:0{padd}}}}"
    else:
        return f"{{{values[0]}..{values[-1]}}}"
