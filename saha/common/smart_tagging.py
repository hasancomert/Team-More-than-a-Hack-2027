"""Smart Tagging (SportsCode XML) as a table on the video-frame axis.

    from saha.common import smart_tagging
    tags = smart_tagging.load()          # one row per <instance>
    restarts = smart_tagging.restarts(tags)

Columns: id, code (upper case), start_s, end_s, start_frame, end_frame (half-open: the first
frame after the phase, like starter/load_smart_tagging.py), half (1/2/None), team ('H', 'A' or
None), team_raw, type (set-piece Type or None), and one column per remaining tag group.

The XML names clubs ("Barcelona", "Atletic Club" -- sic) and 'N/A'. They are mapped to the
submission's sides through TEAM_MAP; an unknown value raises instead of falling through, because a
silently unmapped team is exactly the kind of bug that costs a week. Every throw-in is 'N/A'
(DATA.md) -- handle team None explicitly.

It is automatic, not ground truth: tags are evidence with false positives and negatives.
"""

import xml.etree.ElementTree as ET

import pandas as pd

from saha.common import alignment, paths

TEAM_MAP = {"barcelona": "H", "fc barcelona": "H",
            "atletic club": "A", "athletic club": "A", "athletic club bilbao": "A",
            "n/a": None, "": None}
HALF_MAP = {"1st half": 1, "2nd half": 2}

SET_PIECES = "SET PIECES"
RESTART_TYPES = ("throw-in", "corner kick", "goal kick", "free kick",
                 "kick off (start)", "kick off (after goal)")


def _team(value):
    key = (value or "").strip().lower()
    if key not in TEAM_MAP:
        raise ValueError("unknown Smart Tagging team {!r}: add it to TEAM_MAP".format(value))
    return TEAM_MAP[key]


def load(data=None, path=None):
    """All instances, in file order, with frames computed by saha.common.alignment."""
    path = path or paths.require_assets(["smart_tagging"], data)[0]
    rows = []
    for instance in ET.parse(str(path)).getroot().iter("instance"):
        tags = {}
        for label in instance.findall("label"):
            group = (label.findtext("group") or "").strip()
            if group:
                tags[group] = (label.findtext("text") or "").strip()
        start_s, end_s = float(instance.findtext("start")), float(instance.findtext("end"))
        rows.append({
            "id": (instance.findtext("ID") or "").strip(),
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": alignment.video_frame_of_seconds(start_s),
            "end_frame": alignment.video_frame_of_seconds(end_s),
            "half": HALF_MAP.get(tags.get("Half", "").strip().lower()),
            "team": _team(tags.get("Team")),
            "team_raw": tags.get("Team"),
            "type": tags.get("Type") or None,
            **{"tag_" + k: v for k, v in tags.items() if k not in ("Team", "Half", "Type")},
        })
    return pd.DataFrame(rows)


def restarts(tags):
    """The SET PIECES rows whose Type is a restart, with a lower-case `restart` column."""
    rows = tags[(tags["code"] == SET_PIECES) & tags["type"].notna()].copy()
    rows["restart"] = rows["type"].str.strip().str.lower()
    return rows[rows["restart"].isin(RESTART_TYPES)].reset_index(drop=True)
