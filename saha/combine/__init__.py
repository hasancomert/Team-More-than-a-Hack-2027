"""WS1 combiner: evidence files in, one H/A/D/X label per frame out.

    fuse.py     evidence -> two log-odds tracks (dead vs alive, home vs away), in exact integers
    viterbi.py  three-state segmentation (H, A, D) with switching costs, in exact integers
    decode.py   the two together, per graded half, X outside the halves

Integers end to end on purpose: the submission must be the same file, to the byte, on every
member's machine (Windows and macOS, x86 and ARM). Probabilities and weights arrive quantised to
1e-4 (saha.common.evidence), logits come from a table, sums are int64, and Viterbi ties break by a
fixed rule. No floating-point reduction decides a label.
"""
