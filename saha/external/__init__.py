"""WS5 -- external data and independent validation. Owner and plan: TEAM.md.

WS5 produces no evidence about our match. It builds labelled validation sets from other matches'
open data, degraded to look like ours (ball hidden on ~41% of frames, identities removed, slots
split), so that WS1 and every package can choose parameters somewhere other than the 17 public
possessions. Everything it downloads lives under cache/external/ (gitignored); licences and
attribution go to docs/tech/ws5_external.md.
"""
