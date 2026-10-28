"""A synthetic match in the exact file formats of the real bundle, for tests and CI.

The real data must never reach a cloud LLM or CI (CLAUDE.md), so the code is tested on this
instead: an EPTS metadata XML and raw text file laid out like Metrica's export (kloppy reads them),
a SportsCode Smart Tagging XML, a homography CSV, and the scripted ground truth of the match.

The possession story is scripted and easy on purpose -- the ball carrier is always the tracked
player nearest the ball -- because these files test plumbing (formats, alignment, team mapping,
the pipeline), not modelling.

    from saha.testing import synthetic
    match = synthetic.write_match(tmp_path / "data", n_frames=4000)
    match.truth          # (n_frames,) label codes, X outside the graded halves
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from saha.common.constants import A, D, FPS, GRADED_HALVES, H, N_FRAMES, X
from saha.common.paths import games

TEAM_IDS = ("FIFATMA", "FIFATMB")
TEAM_NAMES = ("Synthetic Home", "Synthetic Away")
TAG_TEAMS = {H: "Barcelona", A: "Atletic Club"}  # the spellings the real XML uses (DATA.md)
FIELD = (105, 68)
SECOND_KICKOFF = 113742  # published in FAQ.md; used only to make the synthetic match realistic


@dataclass
class SyntheticMatch:
    data_dir: Path
    truth: np.ndarray         # (N_FRAMES,) int8 label codes
    n_frames: int             # frames with an ATD record (1..n_frames)
    n_home: int
    n_away: int

    def truth_labels(self):
        return np.array(["H", "A", "D", "X"])[self.truth]


def _script(rng, n_frames):
    """Label codes per video frame: alternating spells with stoppages inside graded halves."""
    truth = np.full(N_FRAMES, X, dtype=np.int8)
    for half, first, last in GRADED_HALVES:
        if first >= n_frames:
            continue
        t, stop = first, min(last + 1, n_frames)
        if half == 2:
            # Like the real match: graded from 110808, dead until the kick-off (FAQ.md).
            t = min(SECOND_KICKOFF, stop)
            truth[first:t] = D
        team = int(rng.integers(0, 2))
        while t < stop:
            length = int(rng.integers(3 * FPS, 25 * FPS))
            truth[t:min(t + length, stop)] = team
            t += length
            if rng.random() < 0.3 and t < stop:
                length = int(rng.integers(4 * FPS, 15 * FPS))
                truth[t:min(t + length, stop)] = D
                t += length
                team = int(rng.integers(0, 2))
            else:
                team = 1 - team
    return truth


def _tracks(rng, truth, n_frames, n_home, n_away, ball_drop):
    """(players (n, S, 2) normalised, speed (n, S), ball (n, 2)), NaN where untracked."""
    n_slots = n_home + n_away
    scale = np.array(FIELD, dtype=np.float64)
    frames = np.arange(n_frames)
    label = truth[:n_frames]

    # The carrier wanders smoothly inside x 17.5-87.5 m, y 12-56 m; everyone else circles it at
    # 6 m or more, so the carrier is always the tracked player nearest the ball.
    carrier = np.stack([52.5 + 35 * np.sin(frames / 431.0),
                        34.0 + 22 * np.sin(frames / 677.0)], axis=1)
    lead = np.where(label == A, n_home, 0)
    players = np.empty((n_frames, n_slots, 2))
    speed = np.empty((n_frames, n_slots))
    for slot in range(n_slots):
        angle = 2 * np.pi * slot / n_slots + 0.01 * frames
        around = carrier + (6 + slot) * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        players[:, slot] = np.where((lead == slot)[:, None], carrier, around) / scale
        speed[:, slot] = np.where(label == D, 0.5, 1.0 + 0.1 * slot)
    players[label == X] = np.nan
    speed[label == X] = np.nan
    ball = (carrier + [0.4, 0.2]) / scale
    ball[label > A] = np.nan

    # Drop the ball in chunks on about `ball_drop` of the alive frames, as occlusions do.
    alive = np.flatnonzero(truth[:n_frames] <= A)
    dropped, target = 0, int(ball_drop * len(alive))
    while dropped < target and len(alive):
        start = int(rng.choice(alive))
        length = int(rng.integers(5, 40))
        ball[start:start + length] = np.nan
        dropped += length
    # A short slot dropout, so missing players are exercised too.
    if n_frames > 100:
        players[50:80, n_slots - 1] = np.nan
        speed[50:80, n_slots - 1] = np.nan
    return players, speed, ball


def _metadata_xml(n_home, n_away, n_frames):
    """EPTS metadata with Metrica's structure: position sensor x/y/z, speed sensor s."""
    n_slots = n_home + n_away
    players, channels, groups = [], [], []
    for slot in range(n_slots):
        number, team = slot + 1, TEAM_IDS[0 if slot < n_home else 1]
        players.append(
            '      <Player id="P{0}" teamId="{1}">\n        <Name>Track {0}</Name>\n'
            '        <ShirtNumber>{0}</ShirtNumber>\n      </Player>'.format(number, team))
        for channel in ("x", "y", "s"):
            channels.append('      <PlayerChannel channelId="{1}" id="player{0}_{1}" '
                            'playerId="P{0}"/>'.format(number, channel))
        groups.append(
            '        <SplitRegister separator=",">\n' + "\n".join(
                '          <PlayerChannelRef playerChannelId="player{}_{}"/>'.format(number, c)
                for c in ("x", "y", "s")) + "\n        </SplitRegister>")

    def team(index, direction):
        return ('      <Team id="{}">\n        <Name>{}</Name>\n        <ProviderTeamsParameters>'
                '\n          <ProviderParameter><Name>attack_direction_first_half</Name>'
                '<Value>{}</Value></ProviderParameter>\n        </ProviderTeamsParameters>\n'
                '      </Team>').format(TEAM_IDS[index], TEAM_NAMES[index], direction)

    half_ids = [(first + 1, last + 1) for _, first, last in GRADED_HALVES]
    params = "".join(
        "<ProviderParameter><Name>{}</Name><Value>{}</Value></ProviderParameter>".format(n, v)
        for n, v in (("first_half_start", half_ids[0][0]), ("first_half_end", half_ids[0][1]),
                     ("second_half_start", half_ids[1][0]), ("second_half_end", half_ids[1][1])))
    sensors = (
        '<Sensor id="position"><Name>Position</Name><Channels>'
        '<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>'
        '<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>'
        '<Channel id="z"><Name>position_z</Name><Unit>normalized</Unit></Channel>'
        '</Channels></Sensor>'
        '<Sensor id="speed"><Name>Speed</Name><Channels>'
        '<Channel id="s"><Name>speed</Name><Unit>meters per second</Unit></Channel>'
        '</Channels></Sensor>')
    return """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <ProviderName>Metrica Sports</ProviderName>
      <FrameRate>{fps}</FrameRate>
      <ProviderGlobalParameters>{params}</ProviderGlobalParameters>
    </GlobalConfig>
    <Sessions>
      <Session id="session1">
        <SessionType>Match</SessionType>
        <FieldSize><Width>{w}</Width><Height>{h}</Height></FieldSize>
      </Session>
    </Sessions>
    <Teams>
{home}
{away}
    </Teams>
    <Players>
{players}
    </Players>
    <Devices>
      <Device id="device1"><Name>Device 1</Name><Sensors>{sensors}</Sensors></Device>
    </Devices>
    <PlayerChannels>
{channels}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification endFrame="{end}" separator=":" startFrame="1">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
{groups}
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
        <BallChannelRef channelId="z"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
""".format(fps=FPS, params=params, w=FIELD[0], h=FIELD[1], home=team(0, "left_to_right"),
           away=team(1, "right_to_left"), players="\n".join(players), sensors=sensors,
           channels="\n".join(channels), groups="\n".join(groups), end=n_frames)


def _raw_lines(players, speed, ball, ball_x_from_frame_id):
    def num(value, digits):
        return "NaN" if not np.isfinite(value) else "{:.{}f}".format(value, digits)

    lines = []
    for f in range(len(ball)):
        frame_id = f + 1
        groups = ";".join("{},{},{}".format(num(players[f, s, 0], 5), num(players[f, s, 1], 5),
                                            "NaN" if not np.isfinite(speed[f, s])
                                            else repr(float(speed[f, s])))
                          for s in range(players.shape[1]))
        if ball_x_from_frame_id:
            bx, by = "{:.6f}".format(frame_id / 1e6), "0.50000"  # frame id readable from x
        else:
            bx, by = num(ball[f, 0], 5), num(ball[f, 1], 5)
        lines.append("{}:{}:{},{},NaN".format(frame_id, groups, bx, by))
    return lines


def _tagging_xml(truth, n_frames):
    """Kickoffs, a restart after every stoppage, and team phases, timed like the real tags."""
    instances, number = [], 0

    def add(code, start_frame, end_frame, labels):
        nonlocal number
        number += 1
        instances.append(
            "  <instance>\n    <ID>{}</ID>\n    <start>{:.2f}</start>\n    <end>{:.2f}</end>\n"
            "    <code>{}</code>\n{}  </instance>".format(
                number, max(start_frame, 0) / FPS, end_frame / FPS, code, "".join(
                    "    <label>\n      <group>{}</group>\n      <text>{}</text>\n"
                    "    </label>\n".format(g, t) for g, t in labels)))

    for index, (_, first, last) in enumerate(GRADED_HALVES):
        if first >= n_frames:
            continue
        half = "1st Half" if index == 0 else "2nd Half"
        stop = min(last + 1, n_frames)
        kickoff = first if index == 0 else SECOND_KICKOFF
        if kickoff < stop:
            add("SET PIECES", kickoff - 88, kickoff + 37,
                [("Team", TAG_TEAMS.get(int(truth[kickoff]), "N/A")), ("Half", half),
                 ("Type", "Kick Off (Start)")])
        window = truth[first:stop]
        bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1,
                                 [len(window)])) + first
        for start, end in zip(bounds[:-1], bounds[1:]):
            label = int(truth[start])
            if label in (H, A) and end - start > 3 * FPS:
                add("BUILD UP", start + 12, end - 12, [("Team", TAG_TEAMS[label]),
                                                       ("Half", half)])
            if end < stop and label == D and truth[end] in (H, A):
                kind = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick")[number % 4]
                team = "N/A" if kind == "Throw-in" else TAG_TEAMS[int(truth[end])]
                add("SET PIECES", end - 88, end + 37, [("Team", team), ("Half", half),
                                                       ("Type", kind)])
    return ('<?xml version="1.0" encoding="utf-8"?>\n<file>\n<ALL_INSTANCES>\n{}\n'
            '</ALL_INSTANCES>\n</file>\n').format("\n".join(instances))


def write_match(data_dir, n_frames=N_FRAMES, n_home=2, n_away=2, seed=0, ball_drop=0.3,
                ball_x_from_frame_id=False):
    """Write tactical/ files under data_dir with games.py's names. Returns a SyntheticMatch."""
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    stems = games.asset_paths(games.GAME, str(data_dir), "tactical")
    Path(stems["atd_raw"]).parent.mkdir(parents=True, exist_ok=True)

    truth = _script(rng, n_frames)
    players, speed, ball = _tracks(rng, truth, n_frames, n_home, n_away, ball_drop)

    Path(stems["atd_meta"]).write_text(_metadata_xml(n_home, n_away, n_frames), encoding="utf-8")
    with open(stems["atd_raw"], "w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(_raw_lines(players, speed, ball, ball_x_from_frame_id)) + "\n")
    Path(stems["smart_tagging"]).write_text(_tagging_xml(truth, n_frames), encoding="utf-8")

    matrix = "1822.663,-511.2005,44.48168,1.698323,169.2971,255.7303,-0.01907188,-0.5269586,1"
    with open(stems["homography"], "w", encoding="ascii", newline="\n") as handle:
        handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        handle.writelines("{},{},0.25,1\n".format(f, matrix) for f in range(N_FRAMES))
    return SyntheticMatch(data_dir, truth, n_frames, n_home, n_away)


def write_truth_csv(path, truth):
    """The synthetic ground truth as a frame,label file, scorable by possession_metrics."""
    labels = np.array(["H", "A", "D", "X"])[truth]
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, label) for f, label in enumerate(labels))
    return Path(path)
