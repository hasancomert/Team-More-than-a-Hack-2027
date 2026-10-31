"""Fusion and segmentation: exact integers, deterministic, and doing what the costs say."""

import numpy as np

from saha.combine import fuse, viterbi
from saha.combine.decode import CombineParams, decode
from saha.common import evidence
from saha.common.constants import A, D, H, N_FRAMES, X


def make(name, p_dead=None, p_home=None, weight=1.0):
    dead, home, w = evidence.empty()
    if p_dead is not None:
        dead[:] = p_dead
    if p_home is not None:
        home[:] = p_home
    w[:] = weight
    return evidence.Evidence(name, evidence.quantise(dead), evidence.quantise(home),
                             evidence.quantise(w), {})


def test_logit_table_is_odd_and_clipped():
    table = fuse.logit_table(0.01)
    assert table[5000] == 0
    assert np.array_equal(table, -table[::-1])
    assert table[0] == table[100] == -4595 and table[-1] == 4595   # logit(0.99) = 4.595 nats


def test_fusion_is_the_weighted_integer_sum():
    first = make("a", p_dead=0.9, p_home=0.8, weight=0.5)
    second = make("b", p_dead=0.2, weight=2.0)
    dead, home = fuse.fuse([(first, 1.0), (second, 0.5)], eps=0.01)
    table = fuse.logit_table(0.01)
    expected_dead = 1000 * 5000 * table[9000] + 500 * 20000 * table[2000]
    assert dead[0] == expected_dead
    assert home[0] == 1000 * 5000 * table[8000]                  # b has no p_home opinion
    assert dead.dtype == np.int64
    swapped = fuse.fuse([(second, 0.5), (first, 1.0)], eps=0.01)
    assert np.array_equal(swapped[0], dead) and np.array_equal(swapped[1], home)


def test_no_opinion_contributes_nothing():
    silent = make("silent", weight=0.0)
    dead, home = fuse.fuse([(silent, 1.0)], prior_dead=0.0)
    assert not dead.any() and not home.any()
    switched_off = make("off", p_dead=0.9, p_home=0.9)
    dead, home = fuse.fuse([(switched_off, 0.0)])
    assert not dead.any() and not home.any()


def segments(path):
    breaks = np.flatnonzero(path[1:] != path[:-1]) + 1
    return [(int(path[s]), int(e - s)) for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(path)])]


def test_viterbi_follows_strong_evidence_and_ignores_blips():
    unit = 1_000_000
    home = np.full(300, 2 * unit, dtype=np.int64)
    home[150:] = -2 * unit                       # away from frame 150
    home[60:63] = -2 * unit                      # a 3-frame blip
    dead = np.full(300, -unit, dtype=np.int64)
    scores = viterbi.emissions(dead, home)
    path = viterbi.decode(scores, 20 * unit, 20 * unit, 20 * unit)
    assert segments(path) == [(H, 150), (A, 150)]
    cheap = viterbi.decode(scores, 1, 20 * unit, 20 * unit)
    assert segments(cheap) == [(H, 60), (A, 3), (H, 87), (A, 150)]


def test_viterbi_ties_are_deterministic():
    flat = np.zeros((50, 3), dtype=np.int64)
    assert (viterbi.decode(flat, 5, 5, 5) == H).all()
    dead = np.r_[np.full(20, -10), np.full(20, 10)].astype(np.int64)
    path = viterbi.decode(viterbi.emissions(dead, np.zeros(40, dtype=np.int64)), 0, 0, 0)
    assert (path[:20] == H).all() and (path[20:] == D).all()


def test_decode_answers_x_outside_the_halves_and_is_reproducible():
    source = make("a", p_dead=0.1, p_home=0.9)
    labels = decode([(source, 1.0)], CombineParams())
    assert labels.shape == (N_FRAMES,)
    assert (labels[:11185] == X).all() and (labels[90652:110808] == X).all()
    assert (labels[190198:] == X).all()
    assert (labels[11185:90652] == H).all()
    assert np.array_equal(labels, decode([(source, 1.0)], CombineParams()))
