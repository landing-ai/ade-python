#!/usr/bin/env python3
"""Generate a markdown changelog section for the commits since the previous tag.

Usage: generate_release_notes.py <owner/repo> <prev_tag> <next_version>

Reads `git log <prev_tag>..HEAD` and groups Conventional Commit subjects into
sections (same section layout the repo's CHANGELOG.md has always used). The
Release workflow prepends the output to CHANGELOG.md and uses it as the GitHub
Release notes.
"""

from __future__ import annotations

import re
import sys
import subprocess
from datetime import date

SECTIONS = [
    ("feat", "Features"),
    ("fix", "Bug Fixes"),
    ("perf", "Performance Improvements"),
    ("revert", "Reverts"),
    ("chore", "Chores"),
    ("docs", "Documentation"),
    ("style", "Styles"),
    ("refactor", "Refactors"),
    ("build", "Build System"),
    ("other", "Other Changes"),
]
SECTION_KEYS = {key for key, _ in SECTIONS}
HIDDEN = {"test", "ci", "release"}
CONVENTIONAL = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?!?:\s*(?P<desc>.+)$")


def main() -> None:
    repo, prev_tag, next_version = sys.argv[1:4]
    log = subprocess.run(
        ["git", "log", "--no-merges", "--pretty=format:%h\t%H\t%s", f"{prev_tag}..HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    buckets: dict[str, list[str]] = {}
    for line in log.splitlines():
        if not line.strip():
            continue
        short, full, subject = line.split("\t", 2)
        match = CONVENTIONAL.match(subject)
        if match:
            kind = match.group("type")
            if kind in HIDDEN:
                continue
            if kind not in SECTION_KEYS:
                kind = "other"
            scope, desc = match.group("scope"), match.group("desc")
            text = f"**{scope}:** {desc}" if scope else desc
        else:
            kind, text = "other", subject
        entry = f"* {text} ([{short}](https://github.com/{repo}/commit/{full}))"
        buckets.setdefault(kind, []).append(entry)

    out = [
        f"## {next_version} ({date.today().isoformat()})",
        "",
        f"Full Changelog: [{prev_tag}...v{next_version}](https://github.com/{repo}/compare/{prev_tag}...v{next_version})",
    ]
    for kind, title in SECTIONS:
        entries = buckets.get(kind, [])
        if entries:
            out += ["", f"### {title}", ""] + entries
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
