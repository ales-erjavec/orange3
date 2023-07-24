import re


YearLong = re.compile("")

def strftimepattern(pattern: str) -> str:
    AMPM = "ap" in pattern or "AP" in pattern
    H = "%I" if AMPM else "%H"
    return (pattern
            .replace("ddd", "%a")
            .replace("dddd", "%A")
            .replace("dd", "%d")
            .replace("d", "%d")
            .replace("MMMM", "%B")
            .replace("MMM", "%b")
            .replace("MM", "%m")
            .replace("M", "%m")
            .replace("YYYY", "%Y")
            .replace("YY", "%y")
            .replace("hh", H)
            .replace("h", H)
            .replace("HH", "%H")
            .replace("H", "%H")
            .replace("mm", "%M")
            .replace("m", "%M")
            .replace("ss", "%S")
            .replace("s", "%S")
            )

