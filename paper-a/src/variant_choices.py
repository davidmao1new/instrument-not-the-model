r"""Which form of a doubly-reported quantity the paper uses, and why.

Several artifacts carry the same quantity twice: once as measured and once
after a correction, or once over all units and once over the subset a stated
rule admits. Reading either is defensible; reading one without recording
which produced two defects in this paper, both of which passed every gate
because the number WAS interpolated from an artifact -- just not the field
the prose described.

audit_variant_choice.py finds these families, works out which member the
builders name, and requires an entry here. An unrecorded family is a
finding; so is an entry that no longer matches anything, because a stale
record stops describing what it registers.

A REASON HERE IS NOT A RUBBER STAMP. It must say what makes the choice
right, and where a reader of the PAPER can see the choice disclosed. If the
paper does not disclose it anywhere, that is the finding, not the entry.

  key:   (artifact file name, the family stem or the restricted key)
  value: (which form the paper uses, why, where the paper says so)
"""

CHOICES = {
    # ------------------------------------------------------------------
    # The conversion factor and the ratio that divides by the effect are
    # different quantities, and the project has conflated them before:
    # CLAUDE.md records that "1.8-108x" once stood in the rules and was
    # neither, because it paired the identified minimum with an
    # unidentified model's ratio.
    # ------------------------------------------------------------------
    ("reporting_scale.json", "predicted_for_distinguishable"): (
        "unrestricted",
        "jacobian_error is 0.25 / mean_cells p(1-p) and the artifact's own "
        "_definition says it is independent of the effect size, so the "
        "separability rule that governs its sibling realised_ratio -- which "
        "does divide by the effect -- does not apply to it. All four "
        "checkpoints therefore enter the conversion-factor range. The "
        "restricted form is the PREDICTED error on the two distinguishable "
        "checkpoints and is a third quantity again, not a filtered version "
        "of what the paper prints.",
        "Section 7: the exclusion rule is scoped to headline ranges that "
        "divide by the effect, and the sentence after it says the "
        "conversion-factor range divides by no effect.",
    ),

    # ------------------------------------------------------------------
    # The noise floor. The frontier arm is reported corrected; the headline
    # dispersion range is reported published. That asymmetry is the one the
    # paper had to disclose rather than remove, because it cannot be
    # removed from the artifacts as they stand.
    # ------------------------------------------------------------------
    ("frontier_noise_floor.json", "ratio_sd_to_effect"): (
        "published for the headline range, corrected for the frontier arm",
        "Correcting the headline range would need a replicate floor for "
        "every cell in it, and two of the six are occupation cells that "
        "occupation_analysis.json carries with no noise-floor counterpart. "
        "Correcting only the cells that have a floor would be a one-sided "
        "correction -- precisely the unreported choice this paper exists to "
        "measure. So the range stays published and says so, and the "
        "frontier arm, where every cell does have a floor, is reported "
        "corrected.",
        "Section 4 marks the range as published ratios and points at "
        "Section 5; Section 5 says the range is published throughout and "
        "gives the reason.",
    ),
    ("frontier_noise_floor.json", "min_ratio"): (
        "corrected",
        "The frontier arm's reported range subtracts the API noise floor "
        "estimated from the byte-identical replicate. Every frontier cell "
        "has such a floor, so the correction is complete on this arm.",
        "Section 5, where the corrected pair is printed.",
    ),
    ("frontier_noise_floor.json", "max_ratio"): (
        "corrected",
        "As min_ratio: the frontier arm is reported after subtracting the "
        "replicate floor, which exists for every cell on this arm.",
        "Section 5, where the corrected pair is printed.",
    ),

    # ------------------------------------------------------------------
    # The corpus totals. The decision is already recorded inside the
    # artifact; this entry points at it rather than restating it, so the
    # two cannot drift.
    # ------------------------------------------------------------------
    ("corpus_size.json", "n_matched_pair_records_excluding_d9_recheck"): (
        "unrestricted",
        "The D9 recheck rows are real matched-pair measurements and belong "
        "in a corpus total, which is what the headline count is. The "
        "artifact states the rule itself under _d9_recheck_rule: the "
        "recheck IS included in the headline totals and is excluded only "
        "from the mechanism panel's CONTRASTS, by analyze_mech_panel.py, "
        "for a different reason.",
        "The abstract's corpus sentence; the excluded-from-contrasts rule "
        "is a panel matter, not a corpus one.",
    ),
    ("corpus_size.json", "n_model_calls_excluding_d9_recheck"): (
        "unrestricted",
        "As the matched-pair count above, and for the same reason recorded "
        "in the artifact's _d9_recheck_rule. The two must agree: a call "
        "total that excluded the recheck beside a pair total that included "
        "it would misstate calls per pair.",
        "The abstract's corpus sentence.",
    ),
}
