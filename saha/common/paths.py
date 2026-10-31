"""Every path the team uses, built in one place with pathlib.

    data         MTAH_DATA_DIR if set, else <repo>/data     (the official variable, see games.py)
    cache        SAHA_CACHE_DIR if set, else <repo>/cache    (frame table, evidence; gitignored)
    submissions  <repo>/submissions                          (gitignored)

Run commands from the repository root. games.py lives there, and this module puts the root on
sys.path so that `import games` works from any entry point.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import games  # noqa: E402  (needs the path above)

DATA_ENV = "MTAH_DATA_DIR"
CACHE_ENV = "SAHA_CACHE_DIR"
SOURCE = "tactical"  # the recording a submission is indexed to; broadcast is opt-in everywhere

EVALUATION_DIR = REPO_ROOT / "evaluation"
PUBLIC_GT = EVALUATION_DIR / "gt" / "BAR-ATH_possession_gt_public.csv"
VALIDATOR = EVALUATION_DIR / "validate_submission.py"
DOCS_DIR = REPO_ROOT / "docs"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
CONFIGS_DIR = REPO_ROOT / "configs"


def data_dir(override=None):
    """The data bundle root: an explicit override, else MTAH_DATA_DIR, else <repo>/data."""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get(DATA_ENV)
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


def use_data_dir(path):
    """Point this process, and every subprocess it starts, at a data bundle (the --data flag)."""
    resolved = data_dir(path)
    os.environ[DATA_ENV] = str(resolved)
    return resolved


def cache_dir():
    env = os.environ.get(CACHE_ENV)
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "cache"


def evidence_dir():
    return cache_dir() / "evidence"


def evidence_path(name):
    return evidence_dir() / "{}.parquet".format(name)


def submissions_dir():
    return REPO_ROOT / "submissions"


def asset_path(name, data=None, source=SOURCE):
    """Path of one provided asset ('atd_raw', 'atd_meta', 'smart_tagging', 'homography', ...).

    games.asset_paths() is the only place a filename is spelled out; this wraps it in pathlib.
    """
    return Path(games.asset_paths(games.GAME, str(data_dir(data)), source)[name])


def require_assets(names, data=None, source=SOURCE):
    """Paths of the named assets, or games.require()'s readable error naming every missing file."""
    return [Path(p) for p in games.require(list(names), games.GAME, str(data_dir(data)), source)]
