# The Instrument Is Not the Model

Artifacts for *The Instrument Is Not the Model: Measuring How Much of an LLM
Hiring Disparity Comes from Unreported Design Choices*.

Every number in the paper is interpolated from a file in `paper-a/data/` at
typesetting time. None is typed into the prose. This repository is what that
sentence refers to.

## Layout

    paper-a/src/     experiment, analysis, audit and build scripts
    paper-a/data/    every derived artifact the paper reads
    paper-c/src/     the companion paper's build
    tests/           the test suite the claims are pinned by
    tools/fonts/     Libertinus, bundled -- see Typesetting below

## Rebuilding the paper

    python paper-a/src/build_paper_v3.py

The build reads `paper-a/data/` and `tools/fonts/`, and nothing else. It does
not call any model, so it runs offline and produces the same PDF the preprint
was made from -- 31 pages, byte-comparable text.

## Re-running the audits

    python paper-a/src/audit_hardtyped_numbers.py    # no measurement is typed
    python paper-a/src/audit_consistency.py          # the document agrees with itself
    python paper-a/src/audit_matrix_evidence.py      # every claim about another
                                                     # paper, re-checked
    python -m pytest tests/ -q

## What is not here, and why

**Literature full texts.** The survey in Section 8 was coded from the full text
of thirteen papers. Those texts are copyrighted and are not redistributed; the
matrix in `paper-a/data/reference/reporting_practice_matrix.json` carries a
verbatim quote or an explicit negative search for every cell, which is what a
reader needs to check a coding without a copy of the paper.

**Model weights.** Open-weight checkpoints are named and hash-pinned in the
artifacts; download them from their original sources.

**Raw model outputs.** The per-call JSONL is large. The derived artifacts every
claim actually reads are all here. Ask if you want the raw records.

## Typesetting

`tools/fonts/` carries Libertinus, under the SIL Open Font License 1.1
(`tools/fonts/Libertinus-7.051/OFL.txt`), which permits this redistribution.
It is bundled rather than assumed because reportlab falls back to Times-Roman
when it cannot find the faces, and Times-Roman has neither U+2009 THIN SPACE
nor U+0394 -- so the paper builds to the right page count, without warning, in
the wrong face. `tests/test_paper_fonts.py` fails if that happens.

Two directories, deliberately: matplotlib reads the OTFs for the figures,
reportlab reads the TTFs for the body.

## Citing

Please cite the paper rather than this repository.
