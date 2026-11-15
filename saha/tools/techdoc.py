"""Assemble the technical document, SAHA.md, from docs/tech/.

    python -m saha.tools.techdoc                 # writes SAHA.md at the repository root
    python -m saha.tools.techdoc --full          # the same with every package's whole section
    python -m saha.tools.techdoc --out <path>    # writes it somewhere else

SAHA.md is docs/tech/saha_main.md (the cross-package sections that WS1 writes), followed by the
appendices. By default they are the short ones of docs/tech/appendix_summaries.md, one per
package, each linking to its full section: the form asks for a short document (team leader, 14
November). With --full they are the package sections themselves, docs/tech/ws1_*.md to ws5_*.md
in that order. A package's letter is fixed (WS1 is Appendix A, ..., WS5 is Appendix E), so
saha_main.md can refer to it even if another section is missing; a second section of the same
package gets "B2" and so on. Each full appendix keeps its owner's text; only its headings move one
level down, so that it nests under "Appendices". Code blocks are left untouched.
"""

import argparse
import re
import sys
from pathlib import Path

from saha.common import paths

TECH_DIR = paths.REPO_ROOT / "docs" / "tech"
MAIN = "saha_main.md"
SUMMARIES = "appendix_summaries.md"
APPENDICES = "## Appendices"
SECTION = re.compile(r"^ws([1-5])_[a-z0-9_]+\.md$")
FENCE = re.compile(r"^\s*(```|~~~)")


def demote(text):
    """The text with every Markdown heading one level deeper, outside code blocks."""
    out, fenced = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced and line.startswith("#"):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def sections(directory):
    """[(letter, path)]: the package sections in WS order. WS n is letter n of the alphabet."""
    found = []
    for path in directory.glob("ws*.md"):
        match = SECTION.match(path.name)
        if match:
            found.append((int(match.group(1)), path.name, path))
    out, seen = [], {}
    for ws, _, path in sorted(found):
        seen[ws] = seen.get(ws, 0) + 1
        letter = chr(ord("A") + ws - 1)
        out.append((letter if seen[ws] == 1 else "{}{}".format(letter, seen[ws]), path))
    return out


def assemble(directory=TECH_DIR, full=False):
    """The text of SAHA.md: the short appendices, or with `full` every package's whole section."""
    main = (directory / MAIN).read_text(encoding="utf-8").rstrip() + "\n"
    if not full:
        # The summaries bring their own "Appendices" heading and introduction.
        lines = main.splitlines()
        if APPENDICES in lines:
            main = "\n".join(lines[:lines.index(APPENDICES)]).rstrip() + "\n"
        summaries = (directory / SUMMARIES).read_text(encoding="utf-8").strip()
        return main + "\n" + summaries + "\n"
    parts = [main]
    for letter, path in sections(directory):
        text = path.read_text(encoding="utf-8").strip()
        title, _, body = text.partition("\n")
        if title.startswith("# "):
            heading = "## Appendix {}: {}".format(letter, title[2:].strip())
        else:
            heading, body = "## Appendix {}: {}".format(letter, path.stem), text
        parts.append(heading + "\n" + demote(body).rstrip() + "\n")
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(paths.REPO_ROOT / "SAHA.md"))
    parser.add_argument("--full", action="store_true",
                        help="every package's whole section instead of the short appendices")
    args = parser.parse_args(argv)
    text = assemble(full=args.full)
    Path(args.out).write_text(text, encoding="utf-8")
    if args.full:
        appendices = ", ".join(path.name for _, path in sections(TECH_DIR)) or "none"
    else:
        appendices = SUMMARIES
    print("wrote {}: {} words, appendices from {}".format(args.out, len(text.split()), appendices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
