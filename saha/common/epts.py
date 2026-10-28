"""Read Metrica's FIFA/EPTS tracking export (metadata XML + raw text) without kloppy.

The layout of a raw line is not a convention, it is declared in the metadata: each
<DataFormatSpecification> lists, in order, which player channel or ball channel every value is.
starter/baseline.py assumes "x,y,speed per slot, slots in <Player> order"; this reader follows the
declaration instead, and data_checks verifies that the two agree on the delivered file.

Two parsing paths, same result:

  fast   separators become commas and pandas' C parser reads the whole file in one pass, with
         float_precision="round_trip" so every value is bit-identical to Python's float(), which
         is what the official baseline uses. Taken when every line has exactly the declared number
         of values.
  exact  a recursive split that follows the declared register tree, line by line. Slower, and
         tolerant of trailing separators and string registers. The fallback.

Error messages name line numbers and counts, never line contents: data must not leak into logs
(see CLAUDE.md).
"""

import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Values the raw file uses for a missing observation.
MISSING_TOKENS = ("NaN", "nan", "")

# Channel names that would mean the export carries ball state or ball ownership. The challenge
# says the ATD names a team even for dead balls (CHALLENGE.md); data_checks reports any match.
STATE_PATTERN = re.compile(r"state|status|alive|dead|possess|own|team|event|phase", re.I)


class EptsError(ValueError):
    pass


# -- metadata ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Channel:
    sensor_id: str
    sensor_name: str
    channel_id: str
    name: str
    unit: str


@dataclass(frozen=True)
class Team:
    team_id: str
    name: str
    params: tuple  # ((name, value), ...)


@dataclass(frozen=True)
class Player:
    player_id: str
    team_id: str
    name: str
    shirt_number: str
    params: tuple


@dataclass(frozen=True)
class Field:
    """One value of a raw line. kind is 'frame', 'string', 'player', 'ball' or 'skip'."""
    kind: str
    name: str = ""
    player_id: str = ""
    channel_id: str = ""


@dataclass(frozen=True)
class Register:
    """A SplitRegister: children joined by one separator. Children are Field or Register."""
    separator: str
    children: tuple


@dataclass(frozen=True)
class DataSpec:
    start_frame: int
    end_frame: int
    root: Register

    @property
    def fields(self):
        return tuple(_flatten(self.root))

    @property
    def separators(self):
        out = []
        _collect_separators(self.root, out)
        return tuple(out)

    def player_order(self):
        """Player ids in the order their values appear on a line."""
        order = []
        for field in self.fields:
            if field.kind == "player" and field.player_id not in order:
                order.append(field.player_id)
        return order


@dataclass
class EptsMetadata:
    frame_rate: int
    field_size: tuple       # (length, width) in metres, or None
    provider: str
    global_params: dict
    session_params: dict
    teams: list             # [Team], file order
    home_team_id: str
    players: list           # [Player], file order
    channels: list          # [Channel], every sensor
    player_channels: dict   # player channel id -> (player_id, channel_id)
    specs: list             # [DataSpec]

    @property
    def away_team_id(self):
        return next(t.team_id for t in self.teams if t.team_id != self.home_team_id)

    def team_name(self, team_id):
        return next(t.name for t in self.teams if t.team_id == team_id)

    def player_ids(self):
        return [p.player_id for p in self.players]


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _child(elem, name):
    if elem is None:
        return None
    return next((c for c in elem if _local(c.tag) == name), None)


def _children(elem, name):
    return [] if elem is None else [c for c in elem if _local(c.tag) == name]


def _text(elem, name, default=None):
    found = _child(elem, name)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def _params(elem):
    """ProviderParameter name -> value, empty values dropped (as kloppy does)."""
    out = {}
    for param in _children(elem, "ProviderParameter"):
        name, value = _text(param, "Name"), _text(param, "Value")
        if name and value:
            out[name] = value
    return out


def _find(root, *path):
    node = root
    for name in path:
        node = _child(node, name)
        if node is None:
            return None
    return node


def _register(elem, player_channels, top_level):
    children = []
    for index, child in enumerate(elem):
        tag = _local(child.tag)
        if tag == "StringRegister":
            # The first value of a line is the frame counter (kloppy reads it the same way).
            kind = "frame" if top_level and index == 0 else "string"
            children.append(Field(kind, name=child.get("name", "")))
        elif tag == "PlayerChannelRef":
            ref = child.get("playerChannelId", "")
            if ref in player_channels:
                player_id, channel_id = player_channels[ref]
                children.append(Field("player", player_id=player_id, channel_id=channel_id))
            else:
                children.append(Field("skip", name=ref))
        elif tag == "BallChannelRef":
            children.append(Field("ball", channel_id=child.get("channelId", "")))
        elif tag == "SplitRegister":
            children.append(_register(child, player_channels, top_level=False))
        else:
            raise EptsError("unknown register <{}> in DataFormatSpecification".format(tag))
    return Register(elem.get("separator", ""), tuple(children))


def _flatten(register):
    for child in register.children:
        if isinstance(child, Register):
            yield from _flatten(child)
        else:
            yield child


def _collect_separators(register, out):
    if register.separator not in out:
        out.append(register.separator)
    for child in register.children:
        if isinstance(child, Register):
            _collect_separators(child, out)


def read_metadata(path):
    """Parse the EPTS metadata XML. Standard library only."""
    root = ET.parse(str(path)).getroot()
    meta = _child(root, "Metadata")
    if meta is None:
        raise EptsError("{}: no <Metadata> element".format(Path(path).name))

    config = _child(meta, "GlobalConfig")
    session = _find(meta, "Sessions", "Session")
    field_size = None
    size = _child(session, "FieldSize")
    if size is not None and _text(size, "Width") and _text(size, "Height"):
        field_size = (float(_text(size, "Width")), float(_text(size, "Height")))

    teams = [Team(t.get("id"), _text(t, "Name", ""),
                  tuple(sorted(_params(_child(t, "ProviderTeamsParameters")).items())))
             for t in _children(_child(meta, "Teams"), "Team")]
    if len(teams) != 2:
        raise EptsError("expected 2 teams in the metadata, found {}".format(len(teams)))

    # Home side: the Score element's idLocalTeam when present (kloppy's rule), else the first
    # team in the file (starter/baseline.py's rule). data_checks verifies they agree.
    score = _find(meta, "Sessions", "Session", "MatchParameters", "Score")
    home = score.get("idLocalTeam") if score is not None else None
    home_team_id = home if home in {t.team_id for t in teams} else teams[0].team_id

    players = [Player(p.get("id"), p.get("teamId"), _text(p, "Name", ""),
                      _text(p, "ShirtNumber", ""),
                      tuple(sorted(_params(_child(p, "ProviderPlayerParameters")).items())))
               for p in _children(_child(meta, "Players"), "Player")]

    channels = []
    for device in _children(_child(meta, "Devices"), "Device"):
        for sensor in _children(_child(device, "Sensors"), "Sensor"):
            for channel in _children(_child(sensor, "Channels"), "Channel"):
                channels.append(Channel(sensor.get("id", ""), _text(sensor, "Name", ""),
                                        channel.get("id", ""), _text(channel, "Name", ""),
                                        _text(channel, "Unit", "")))

    player_channels = {pc.get("id"): (pc.get("playerId"), pc.get("channelId"))
                       for pc in _children(_child(meta, "PlayerChannels"), "PlayerChannel")}

    specs_elem = _child(root, "DataFormatSpecifications")
    specs = [DataSpec(int(s.get("startFrame")), int(s.get("endFrame")),
                      _register(s, player_channels, top_level=True))
             for s in _children(specs_elem, "DataFormatSpecification")]
    if not specs:
        raise EptsError("no DataFormatSpecification in the metadata")

    return EptsMetadata(
        frame_rate=int(_text(config, "FrameRate", "0")),
        field_size=field_size,
        provider=_text(config, "ProviderName", ""),
        global_params=_params(_child(config, "ProviderGlobalParameters")),
        session_params=_params(_child(session, "ProviderSessionParameters")),
        teams=teams,
        home_team_id=home_team_id,
        players=players,
        channels=channels,
        player_channels=player_channels,
        specs=specs,
    )


def channels_report(meta):
    """Plain-text inventory of everything the metadata declares. Names only, no data."""
    lines = ["provider: {}   frame rate: {}   field size: {}".format(
        meta.provider, meta.frame_rate, meta.field_size)]
    lines.append("teams: {}".format(", ".join(
        "{}{}".format(t.name, " (home)" if t.team_id == meta.home_team_id else "")
        for t in meta.teams)))
    lines.append("players (track slots): {}".format(len(meta.players)))
    lines.append("global provider parameters: {}".format(sorted(meta.global_params) or "-"))
    lines.append("session provider parameters: {}".format(sorted(meta.session_params) or "-"))
    team_params = sorted({name for t in meta.teams for name, _ in t.params})
    player_params = sorted({name for p in meta.players for name, _ in p.params})
    lines.append("team provider parameters: {}".format(team_params or "-"))
    lines.append("player provider parameters: {}".format(player_params or "-"))
    lines.append("sensors and channels:")
    for c in meta.channels:
        lines.append("  sensor {!r} ({}) channel {!r} name {!r} unit {!r}".format(
            c.sensor_id, c.sensor_name, c.channel_id, c.name, c.unit))
    for index, spec in enumerate(meta.specs):
        fields = spec.fields
        kinds = {}
        for f in fields:
            key = (f.kind, f.channel_id if f.kind in ("player", "ball") else f.name)
            kinds[key] = kinds.get(key, 0) + 1
        lines.append("data spec {}: frames {}..{}, {} values per line, separators {}".format(
            index, spec.start_frame, spec.end_frame, len(fields), list(spec.separators)))
        for (kind, name), count in sorted(kinds.items()):
            lines.append("    {:<7} {:<12} x{}".format(kind, name, count))
    flagged = sorted({name for c in meta.channels for name in (c.channel_id, c.name)
                      if STATE_PATTERN.search(name or "")}
                     | {f.name for s in meta.specs for f in s.fields
                        if f.kind == "string" and STATE_PATTERN.search(f.name)}
                     | {name for name in list(meta.global_params) + team_params + player_params
                        if STATE_PATTERN.search(name)})
    lines.append("names suggesting ball state / ownership: {}".format(flagged or "none"))
    return "\n".join(lines)


# -- raw data ----------------------------------------------------------------------------------

@dataclass
class RawTracking:
    """The raw file as arrays, in file line order. Slots follow the metadata <Player> order."""
    frame_id: np.ndarray    # (n,) int64
    players: dict           # channel id -> (n, n_slots) float64
    ball: dict              # channel id -> (n,) float64
    slot_player_ids: list   # metadata player id of each slot
    path: str               # "fast" or "exact"


def _read_lines(path):
    lines = Path(path).read_bytes().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _frame_ids(lines, top_separator):
    sep = top_separator.encode("ascii")
    ids = np.empty(len(lines), dtype=np.int64)
    for i, line in enumerate(lines):
        cut = line.find(sep)
        try:
            ids[i] = int(line[:cut] if cut >= 0 else line)
        except ValueError:
            raise EptsError("line {}: frame counter is not an integer".format(i + 1))
    return ids


def _fast_block(lines, spec):
    """(n, n_fields) float64 via the C parser, or None if the layout does not fit."""
    fields = spec.fields
    if any(f.kind == "string" for f in fields):
        return None
    separators = spec.separators
    if any(len(s) != 1 for s in separators):
        return None
    table = bytes.maketrans(b"".join(s.encode("ascii") for s in separators),
                            b"," * len(separators))
    blob = b"\n".join(lines).translate(table)

    # The C parser pads a short line with NaN instead of failing, so count values per line first:
    # a line missing its ball group must not silently become a line without a ball.
    buffer = np.frombuffer(blob, dtype=np.uint8)
    ends = np.append(np.flatnonzero(buffer == ord("\n")), len(buffer))
    commas = np.searchsorted(np.flatnonzero(buffer == ord(",")), ends)
    if not np.all(np.diff(np.concatenate(([0], commas))) == len(fields) - 1):
        return None
    try:
        frame = pd.read_csv(io.BytesIO(blob), header=None, sep=",", dtype=np.float64,
                            na_values=list(MISSING_TOKENS), keep_default_na=False,
                            float_precision="round_trip", engine="c")
    except (pd.errors.ParserError, ValueError):
        return None
    if frame.shape != (len(lines), len(fields)):
        return None
    return frame.to_numpy(dtype=np.float64)


def _split_exact(text, register, out):
    """Append this register's values to out, following the declared tree."""
    parts = text.split(register.separator) if register.separator else [text]
    expected = len(register.children)
    # A trailing separator is legal in the EPTS grammar (kloppy's regex allows it too).
    while len(parts) > expected and parts[-1] == "":
        parts.pop()
    if len(parts) != expected:
        raise EptsError("expected {} values for separator {!r}, found {}".format(
            expected, register.separator, len(parts)))
    for part, child in zip(parts, register.children):
        if isinstance(child, Register):
            _split_exact(part, child, out)
        elif child.kind == "string":
            out.append(np.nan)  # string registers carry no numeric channel we use
        else:
            token = part.strip()
            out.append(np.nan if token in MISSING_TOKENS else float(token))


def _exact_block(lines, spec):
    rows = np.empty((len(lines), len(spec.fields)), dtype=np.float64)
    for i, line in enumerate(lines):
        values = []
        try:
            _split_exact(line.decode("ascii").strip(), spec.root, values)
        except (EptsError, ValueError, UnicodeDecodeError) as error:
            raise EptsError("raw line {} does not match the declared layout: {}".format(
                i + 1, error.__class__.__name__ if not isinstance(error, EptsError) else error))
        rows[i] = values
    return rows


def read_raw(raw_path, meta, force_exact=False):
    """Parse the raw tracking file into arrays. See the module docstring for the two paths."""
    lines = _read_lines(raw_path)
    if not lines:
        raise EptsError("{}: empty file".format(Path(raw_path).name))

    top = meta.specs[0].root.separator
    ids = _frame_ids(lines, top)
    slot_ids = meta.player_ids()
    slot_of = {player_id: slot for slot, player_id in enumerate(slot_ids)}
    n, n_slots = len(lines), len(slot_ids)

    players, ball = {}, {}
    which = np.full(n, -1, dtype=np.int64)
    for index, spec in enumerate(meta.specs):
        which[(ids >= spec.start_frame) & (ids <= spec.end_frame) & (which < 0)] = index
    if (which < 0).any():
        raise EptsError("{} raw lines have a frame counter outside every DataFormatSpecification"
                        .format(int((which < 0).sum())))

    used_path = "fast"
    for index, spec in enumerate(meta.specs):
        rows = np.flatnonzero(which == index)
        if not len(rows):
            continue
        block_lines = [lines[i] for i in rows]
        values = None if force_exact else _fast_block(block_lines, spec)
        if values is None:
            used_path = "exact"
            values = _exact_block(block_lines, spec)
        for column, field in enumerate(spec.fields):
            if field.kind == "player":
                if field.player_id not in slot_of:
                    raise EptsError("player {!r} is in the data spec but not in <Players>"
                                    .format(field.player_id))
                target = players.setdefault(
                    field.channel_id, np.full((n, n_slots), np.nan, dtype=np.float64))
                target[rows, slot_of[field.player_id]] = values[:, column]
            elif field.kind == "ball":
                target = ball.setdefault(field.channel_id, np.full(n, np.nan, dtype=np.float64))
                target[rows] = values[:, column]
            elif field.kind == "frame":
                if not np.array_equal(values[:, column], ids[rows].astype(np.float64)):
                    raise EptsError("frame counter parsed inconsistently")
    return RawTracking(ids, players, ball, slot_ids, used_path)
