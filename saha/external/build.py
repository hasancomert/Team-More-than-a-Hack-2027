"""WS5: build the external validation sets.

    python -m saha.external.build          -> cache/external/...

STUB written by WS1 in Phase 0. Starting plan (TEAM.md, not binding):
    1. Metrica sample data (github.com/metrica-sports/sample-data): same 0-1 coordinates; game 3
       is in the same EPTS format, so saha.common.epts reads it.
    2. SkillCorner Open Data (github.com/SkillCorner/opendata): 10 fps, possession per frame.
    3. DFL/Sportec open data (kloppy sportec.load_open_tracking_data): first check that the
       BallPossession and BallStatus fields are actually filled in the open files.
Degrade each to our conditions, record licences, and check against the public 5 minutes that
their notion of possession matches ours before trusting it.
"""


def main(argv=None):
    print("saha.external.build is a stub; see TEAM.md (WS5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
