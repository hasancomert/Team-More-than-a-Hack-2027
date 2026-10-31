"""WS1 shared layer: paths, constants, the one loader and the one frame alignment.

Every package reads the match through this module and nothing else, so that the alignment is
defined exactly once. Import submodules explicitly (`from saha.common import frame_table`); this
file imports nothing on purpose.
"""
