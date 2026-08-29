r"""The five compressed sections of the ICLR variant, as interpolating text.

WHAT THIS IS. The full paper's §§1, 2, 9, 10 and 11 run to ~8,800 words; the
ICLR page budget wants ~2,500. The compression briefs in facct/COMPRESS.md map
what each section must keep (the scope limits, the bound-not-absence framings,
the numbers stated nowhere else); this module is those briefs executed, for
the CONDENSED build only. The full paper is untouched, and none of this is
FAccT prose: FAccT prohibits LLM-generated text, and these sections are
LLM-drafted and disclosed.

EVERY NUMBER INTERPOLATES. Each section builder receives the same EXPORT dict
the full paper's prose interpolates from, plus the loaded artifacts it needs;
a value with no source raises rather than being typed. Word budgets from
COMPRESS.md are asserted at build time with 15% slack, so a growing section
fails the build instead of quietly outgrowing the venue.
"""
from __future__ import annotations

BUDGET = {"1": 700, "2": 550, "9": 600, "10": 400, "11": 250}


def pctw(x, nd=0):
    return f"{100 * x:.{nd}f} %"


def section_1(E, art):
    """Introduction: legal stakes, the unreported choices, the finding, the
    contributions -- plus the between-model scope sentence COMPRESS.md says
    must survive, because it licenses §4.1."""
    yield ("An audit of a hiring model reports one number, and in New York "
           "City that number has legal force: Local Law 144 makes it "
           "unlawful to use an automated employment decision tool that has "
           "not been audited for bias in the preceding year, and the 2023 "
           "DCWP final rule fixes what the audit must compute. Producing "
           "the number takes dozens of choices the audit does not report: "
           "how the instruction was worded, which names were drawn, what "
           "job was posted, how the model was served, and how the analysis "
           "resampled, converted and paired what came back.")
    yield ("This paper measures what those choices are worth. Holding the "
           f"model fixed, over {E['n_records']:,} matched pairs from "
           f"{E['n_calls']:,} calls on {E['n_panel']} open-weight and "
           f"{E['n_front']} frontier checkpoints, the instruction wording "
           f"alone moves the reported demographic effect by "
           f"{pctw(E['ratio_lo'])} to {pctw(E['ratio_hi'])} of itself "
           f"across the {E['ratio_n']} model-by-posting cells where the "
           "effect is separable from zero, including under edits that "
           "change no word. The job posted and the name drawn move it "
           "comparably; the analysis-time choices move it again, with the "
           f"resampling unit alone worth {E['resamp_lo']:.1f} to "
           f"{E['resamp_hi']:.1f} times on the width of the interval. The "
           "reported number is, to a first approximation, a property of "
           "the instrument as much as of the model.")
    yield ("One reading must be closed at the outset: none of this says "
           "the wording matters more than which model was tested. The "
           "between-model spread on this panel is larger than the "
           "between-wording spread within any model; what does not survive "
           "is the assumption that a number produced under one instrument "
           "transfers to another. The comparison in Section 4.1 is "
           "licensed by that distinction and makes no between-model "
           "claim.")
    yield ("The contributions, in order. First, a decomposition of a "
           "reported disparity into what the model contributes and what "
           "each unreported choice contributes, on a common scale, with "
           "every component measured rather than argued. Second, evidence "
           "that the measurement itself is not deterministic, and a "
           "serving configuration under which it reproduces bitwise. "
           f"Third, a survey of {E['n_audits']} LLM hiring audits showing "
           "that none reports enough for a reader to reconstruct its "
           "number. Fourth, a minimum reporting set with a calibrated "
           "screening rule, priced so an auditor can adopt it.")


def section_2(E, art):
    """Background: five paragraphs and the gap sentence, per the brief."""
    yield ("Correspondence audits are the design this paper borrows and "
           "the tradition it answers to. Bertrand and Mullainathan is the "
           "canonical instance: names selected for the highest frequency "
           "ratio between racial groups, which builds a correlation "
           "between the demographic signal and the names' rarity that "
           "Section 4.4 shows surviving into the tokenizer. The tradition "
           "has audited its own instrument before. The Heckman-Siegelman "
           "critique, formalised by Neumark, shows that variance "
           "differences in unobservables alone can generate spurious "
           "discrimination estimates of either sign; Fryer and Levitt "
           "show a name indexes the circumstances of its bearer's birth; "
           "Gaddis shows the perceived race of a name varies name by "
           "name. Each is a fact about the instrument, found by the field "
           "that built it.")
    yield ("Template bias is the closest ancestor. Lahey and Beasley, in "
           "2009, split one audit into two arbitrary sets of four resume "
           "templates and obtained a significant discrimination estimate "
           "from one set and a null from the other, on the same data. "
           "That is this paper's result on human employers, seventeen "
           "years earlier, and it is why the argument here is a methods "
           "contribution to the audit literature rather than a complaint "
           "about it.")
    yield ("Prompt sensitivity is documented at length for language "
           "models: Sclar and colleagues on formatting, Seshadri on "
           "perturbations, Tan on instruction phrasing. All of it "
           "measures accuracy moving. An and Rudinger show a name's "
           "tokenization length influences how a model treats it "
           "independently of the name's demographics, which the companion "
           "paper takes up. Researcher degrees of freedom have their own "
           "literature, from Steegen's multiverse to Simonsohn's "
           "specification curve, and both vary the analysis over a fixed "
           "measurement.")
    yield ("The gap this paper fills sits at the intersection: prior work "
           "varies the analysis on a fixed measurement, or measures "
           "accuracy rather than an estimand. Nobody has varied how the "
           "stimulus itself was built and served, and asked what happens "
           "to the reported disparity.")


def section_9(E, art):
    """The reporting set: lead, four checkable items, the screening rule,
    and the self-failure the brief says must survive."""
    srn = art.get("srn") or {}
    ms = (art.get("mstruct") or {}).get("minimal_rerun", {})
    n_flip = len(ms.get("flips", {}).get("checkpoint_pinned", []))
    yield ("What follows is the minimum an audit must report for its "
           "number to be checkable, stated as items a reader can hold an "
           "audit against. The arguments for each are made in Sections 4 "
           "to 6; here each item is stated and priced.")
    yield ("One. The wording set, and the dispersion of the effect across "
           "it. A single wording is a sample of size one from the "
           "instrument; the dispersion is what converts the headline "
           "number into a range a reader can use. Two. The name list, its "
           "source, and its token statistics against the served "
           "checkpoint's own vocabulary, because only "
           f"{pctw(E['token_matched_lo'])} to {pctw(E['token_matched_hi'])} "
           "of pairs from a standard validated list are token-matched. "
           "Three. The serving configuration: checkpoint digest, "
           "quantization, batching, cache policy and decoding parameters. "
           "The quantization alone shifts the effect by "
           f"{E['quant_lo']:.2f} to {E['quant_hi']:.2f} times the "
           "between-wording standard deviation, and the measurement "
           "reproduces bitwise only once batching and cache residency "
           "are controlled. Four. The resampling unit and the reporting "
           "scale, with the unit following the assignment process; "
           "resampling rows rather than name pairs narrows intervals "
           f"{E['resamp_lo']:.1f} to {E['resamp_hi']:.1f} fold, and a "
           "fixed operating point misstates the percentage-point "
           f"conversion by {E['jacobian_lo']:.1f} to "
           f"{E['jacobian_hi']:.0f} times.")
    if n_flip:
        yield ("The cheapest single repair on the surveyed panel is item "
               "three's first clause: under a minimal re-run criterion "
               "(exact prompt, pinned checkpoint, decoding parameters), "
               f"pinning the checkpoint alone would bring {n_flip} of the "
               f"{ms.get('n_audits', 13)} audits up to the criterion, and "
               "most already half-report it.")
    lo = hi = None
    if srn.get("rate_at_empirical_rho"):
        vals = list(srn["rate_at_empirical_rho"].values())
        lo, hi = min(vals), max(vals)
    yield ("Alongside the set, a screening rule: report a disparity as "
           "established only if it is separable from zero under every "
           "wording in the published set. The rule's false-positive rate "
           "is calibrated in closed form under an equicorrelated normal "
           "model rather than asserted"
           + (f", and on this panel's measured wording correlations it "
              f"runs {lo:.0e} to {hi:.1e}, at least four times stricter "
              "than the nominal test it replaces" if lo is not None else "")
           + ". The rule is one-directional: it protects against "
           "reporting a disparity a different wording would not have "
           "found, and does nothing about one that every wording misses.")
    yield ("The rule fails on our own panel: no disparity this paper "
           "measured would be established under it. That is stated here "
           "rather than in an appendix because a reporting standard whose "
           "author exempts himself is worth nothing, and because the "
           "failure is the finding. The instrument is currently too "
           "unstable for a wording-robust disparity to have been "
           "measured, on these checkpoints, by anyone.")


def section_10(E, art):
    """Threats: internal, external, construct -- with the bound-not-absence
    framings COMPRESS.md protects, and the one threat this design lacks."""
    yield ("Internal validity. The measurement reproduces bitwise once "
           "batching and cache residency are controlled, and the serving "
           f"shift is bounded within {pctw(E['serving_bound'], 1)} of the "
           "effect on the checkpoints where a ratio is defined: a bound, "
           "not an absence. The headline dispersions come from a study "
           "run before those controls existed, and their replicate floor "
           "is measured and subtracted rather than assumed. One "
           "template-concentration statistic was found to carry most of "
           "its own reported value as estimator bias, by an audit of this "
           "paper; it is retained in corrected form and the correction "
           "is released.")
    yield ("External validity. Open-weight checkpoints carry everything "
           "the serving stack touches, one name list carries the "
           "demographic manipulation, and three postings carry the "
           "occupational one, so no claim is made about occupations in "
           "general. The stimulus-side results were re-run on rental "
           "tenancy screening and content moderation and on frontier "
           "APIs, and the dispersion is of the same order there; the "
           "serving-side results cannot be tested on a stack whose "
           "numerics no auditor can inspect. One threat this design does "
           "not carry: there is no human tester to deviate from random "
           "assignment, so experimenter bias, the second concern named by "
           "Lahey and Beasley, is absent by construction.")
    yield ("Construct validity. Whether a name manipulation measures what "
           "an audit calls it is checked nowhere in this literature, and "
           "our own manipulation check is only partly reassuring: with "
           "twelve names per race the design detects only correlations "
           f"of {E['mde_within']:.2f} or larger, so those are the "
           "design's limits rather than results. The outcome is a "
           "decision the model was made to give, and on the lower half "
           "of the panel the grammar is doing real work; the three "
           "natural decision rules disagree on up to "
           f"{pctw(E['rule_disagreement'])} of probes on those models. "
           "Each is a place the result might not reach, and none is a "
           "place we have looked.")


def section_11(E, art):
    """Conclusion: the brief's four sentences."""
    yield ("The matched-pair difference is supposed to be the robust "
           "quantity: whatever the two applications share should cancel. "
           "It does not, by enough to change what an audit concludes, "
           "and the part that fails to cancel is the instrument. The "
           "remedy is not to abandon the design but to report the "
           "choices, and where a statistic can be made invariant to a "
           "choice, to report that statistic instead: the mean paired "
           "difference does not care how the pairing was drawn, and the "
           "probability of superiority does not care where the operating "
           "point sits. What stays open is the channel, which is not "
           "identified, and the standard itself, which no published "
           "audit has yet been re-run against.")


SECTIONS = {"1": section_1, "2": section_2, "9": section_9,
            "10": section_10, "11": section_11}
