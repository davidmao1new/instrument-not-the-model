"""Study 5 analysis: does the delimiter effect generalise, and is it specific?

THREE THINGS THIS DOES THAT analyze_mechanism.py DID NOT.

1. D4 AND D5 ARE REPORTED SEPARATELY. Study 3 labelled a contrast "one
   delimiter destroyed" and computed it from D4 alone. On Llama-3.1-8B the two
   single-delimiter conditions do not behave alike at all: destroying the FIRST
   delimiter moved the effect +2.51, destroying the SECOND moved it +0.18, a
   fourteen-fold asymmetry that the pooled label concealed and that the
   dose-response correlation absorbed as if it were noise. If the two boundaries
   differ, the mechanism is not "how many delimiters were destroyed" but "which
   one", and that is a different and more specific claim. It is reported here as
   the primary result rather than smoothed away.

2. THE QUALIFICATION CONTRAST IS COMPUTED FROM THE SAME CALLS. Every condition
   was run against the strongest and the weakest resume, so each name yields
   m(strong) - m(weak) under each condition. That is a second contrast on the
   same scale as the demographic one, and it decides whether delimiter damage
   destabilises the model's decisions generally or only its reading of a weak
   social signal. Without it the demographic framing is decoration.

3. BENJAMINI-HOCHBERG IS COMPUTED IN THIS FILE. The Study 3 artifact contained
   p_bh values that the script on disk could not produce; the values recompute
   correctly, so the numbers were right, but the analysis was not reproducible
   from its own code. Here the family is defined explicitly, adjusted in one
   place, and the family size is written into the artifact so the paper can
   state it rather than assert it.

FAMILY. All condition-versus-baseline and design contrasts, for every model and
both inference modes, adjusted together. That is the largest defensible family:
adjusting within a model would let a reader ask why the panel does not count,
and adjusting across everything is the conservative choice.

THE CLASS CONTRAST, ADDED LATER. Section 7 of the paper splits these
condition-versus-baseline tests into two classes -- the conditions that destroy
a structural delimiter and the conditions that change nothing structural --
compares the two, and concludes that no delimiter mechanism exists. That
conclusion is a statement about the DIFFERENCE between the classes, and nothing
in this file used to estimate it: two point estimates and two failures to reject
were carrying an assertion of absence. `class_contrast()` now puts a bootstrap
interval on that difference, on both the mean |effect| and the significance
rate, resampling name pairs, and reports the smallest class difference the
design could have detected at 80% power. The section's honest claim is bounded
by that floor rather than by the absence of a star.

    .venv/Scripts/python.exe paper-a/src/analyze_mech_panel.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import effectsize as es  # noqa: E402
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data" / "mechanism_panel"
OUT = DATA / "mech_panel_analysis.json"
RNG = np.random.default_rng(20260728)

# WHY 40,000 AND NOT 4,000. A bootstrap p-value is quantised at 1/n_boot, so at
# 4,000 replicates the resolvable values are multiples of 0.00025 and many of
# the 168 contrasts landed on exactly the same raw p (0.0063 = 25/4000).
# Benjamini-Hochberg ranks by p, so tied p-values are ordered arbitrarily, and
# four contrasts sat at adjusted p = 0.0496 versus 0.0500 -- on opposite sides
# of the threshold by a tie-break rather than by their data. Raising the
# resolution to 1/40,000 separates the ties and makes the BH ranking a function
# of the measurements. It costs about a minute.
N_BOOT = 40_000

ORDER = ["D0", "D1", "D2", "D3", "D7", "D8", "D9", "D5", "D4", "D10", "D6"]
NOTE = {
    "D0": "baseline",
    "D1": "space early, delimiter intact",
    "D2": "space late, delimiter intact",
    "D3": "token changed beside intact delimiter",
    "D7": "both delimiters substituted (still delimiters)",
    "D4": "FIRST delimiter destroyed (posting | candidate)",
    "D5": "SECOND delimiter destroyed (resume | question)",
    "D6": "both delimiters destroyed",
    "D8": "NAME +1 token, no delimiter touched",
    "D9": "NAME +2 tokens, no delimiter touched",
    "D10": "NAME +1 AND second delimiter destroyed",
}
# How far the candidate's name is pushed, in tokens. Verified against the
# tokenizer for every condition and every template before running.
NSHIFT = {"D0": 0, "D1": 0, "D2": 0, "D3": 0, "D7": 0, "D5": 0,
          "D4": 1, "D6": 1, "D8": 1, "D9": 2, "D10": 1}
NDELIM = {"D0": 0, "D1": 0, "D2": 0, "D3": 0, "D7": 0, "D4": 1, "D5": 1, "D6": 2,
          "D8": 0, "D9": 0, "D10": 1}

# Contrasts. Each is (label, condition, reference). Every one is a within-cell
# paired difference, so the names and resumes cancel.
CONTRASTS = [
    ("D1 vs base  [control: space early]", "D1", "D0"),
    ("D2 vs base  [control: space late]", "D2", "D0"),
    ("D3 vs base  [control: token beside intact delimiter]", "D3", "D0"),
    ("D7 vs base  [substitution, delimiter survives]", "D7", "D0"),
    ("D4 vs base  [FIRST delimiter destroyed]", "D4", "D0"),
    ("D5 vs base  [SECOND delimiter destroyed]", "D5", "D0"),
    ("D6 vs base  [both destroyed]", "D6", "D0"),
    ("C1  D4 - D3  [destruction vs token-at-boundary]", "D4", "D3"),
    ("C2  D6 - D7  [destruction vs substitution]", "D6", "D7"),
    ("C5  D4 - D5  [which boundary matters]", "D4", "D5"),
    # --- the contrasts that separate position from delimiter ---------------
    # D0-D7 confound them perfectly: every condition that moved the effect also
    # pushed the name one token later. These four break the confound.
    ("P1  D8 - D0  [POSITION ALONE: name +1, nothing fragmented]", "D8", "D0"),
    ("P2  D9 - D0  [POSITION DOSE: name +2, nothing fragmented]", "D9", "D0"),
    ("P3  D4 - D8  [FRAGMENTATION with position held equal]", "D4", "D8"),
    ("P4  D10 - D8 [FRAGMENTATION added on top of position]", "D10", "D8"),
]

# The two condition classes the paper's section 7 compares. D10 destroys a
# delimiter but has no contrast against the D0 baseline -- it is only ever
# measured against D8 -- so it belongs to neither class and is not silently
# assigned to one.
CLASS_DESTROY = ("D4", "D5", "D6")
CLASS_CONTROL = ("D1", "D2", "D3", "D7", "D8", "D9")
# WHICH SPLIT THE HEADLINE NUMBERS USE. D7 was designed as a null control and is
# not one. Tokenised against the reference checkpoint's own vocabulary --
# paper-a/data/instrument/condition_tokens_llama-3.1-8b-instruct.json, where its
# `delimiter_disposition` is "substituted" against "fragmented" for D4/D5/D6 and
# "intact" for D1/D2/D3/D8/D9 -- it replaces both merged delimiter tokens with a
# different single token, fragmenting nothing and displacing nothing. That is
# neither of the two things the classes name, so it is left out of both here,
# which is also how the paper's condition-class table splits once it reads the
# same file. The other two readings are computed in full beside it rather than
# argued about.
PRIMARY_SCHEME = "fragmented_vs_intact"

# z(0.975) + z(0.80) = 2.8016, the multiplier on a standard error that gives the
# smallest true value detectable at 80% power with a two-sided 5% test. Same
# constant `mde()` takes as `alpha_z`, named here so the two cannot drift.
Z_POWER_80 = 2.80


def load():
    """Every panel row, one per design cell.

    EXCLUDING THE D9 RECHECK. `mech_d9recheck_*.jsonl` holds a deliberate
    re-measurement of the D8 and D9 base-arm cells, made to adjudicate whether
    the stored D9 rows had been produced by a superseded condition definition.
    adjudicate_d9.py returned VINDICATED, which means the stored rows stand and
    the recheck is an INDEPENDENT cross-session reproducibility measurement --
    not a replacement.

    A bare `mech_*.jsonl` glob pulled those files in anyway, and because
    `sorted()` places `mech_d9recheck_chat_*` after `mech_chat_*` but before
    `mech_raw_*`, the recheck values overwrote the originals in chat mode and
    lost to them in raw mode. 576 cells, one mode only, moving exactly the
    conditions the position contrasts P1-P4 are built from. Silent, asymmetric,
    and invisible in the output. The prefix is now matched explicitly.
    """
    rows = []
    for f in sorted(DATA.glob("mech_*.jsonl")):
        if f.name.startswith("mech_d9recheck_"):
            continue
        for r in st.read_jsonl(f):
            if (r.get("white_margin") is not None
                    and r.get("black_margin") is not None):
                rows.append(r)
    best = {}
    for r in rows:
        best[(r["model"], r["mode"], r["cond"], r["template"], r["pair"])] = r
    return list(best.values())


def mde(by, ref="D0", alpha_z=2.80):
    """Minimum detectable effect at 80% power, two-sided alpha = 0.05.

    The project's analysis rules require an MDE beside every null, and this
    study is now mostly nulls, so the requirement is load-bearing rather than
    ceremonial. Reported two ways, because they answer different questions:

      ABSOLUTE      the smallest paired shift the design could have found,
                    with the SE taken over per-pair means so the unit is the
                    name pair (n = 24), not the (template, pair) cell (n = 72).
                    Compare it to the effect seen in another model to ask
                    "would we have seen it here if it were the same size?"

      RELATIVE      the same quantity as a percentage of the model's own
                    dynamic range, taken as its qualification contrast. A model
                    that barely separates a strong resume from a weak one has
                    little to move, and its null is correspondingly weak
                    evidence.

    Both are needed. The base checkpoint rules out a delimiter effect the same
    ABSOLUTE size as the instruction-tuned model's, and cannot rule out one the
    same RELATIVE size, and those two statements support opposite conclusions
    about instruction tuning.
    """
    out = {}
    ref_map = {(r["template"], r["pair"]): r["white_margin"] - r["black_margin"]
               for r in by.get(ref, [])}
    for cond, cells in by.items():
        if cond == ref:
            continue
        m = {(r["template"], r["pair"]): r["white_margin"] - r["black_margin"]
             for r in cells}
        keys = sorted(set(m) & set(ref_map))
        if len(keys) < 3:
            continue
        d = np.array([m[k] - ref_map[k] for k in keys], dtype=float)
        # THE UNIT IS THE NAME PAIR, NOT THE (template, pair) CELL. The 72
        # cells hold 24 independent pairs; treating them as 72 draws was the
        # exact defect the resampling-unit study measured on these contrasts
        # (clustered/iid SE ratio 0.644-1.479, a third of cells widening by
        # more than 10%), and every other estimator in this file already
        # clusters on the pair. Collapsing to per-pair means over templates is
        # the same correction paired_contrast applies; the cell-level sd is
        # still reported for the dispersion description.
        pair_means = {}
        for k, v in ((k, m[k] - ref_map[k]) for k in keys):
            pair_means.setdefault(k[1], []).append(v)
        pm = np.array([float(np.mean(v)) for v in pair_means.values()])
        out[cond] = dict(n=len(d), n_pairs=len(pm), sd=float(d.std(ddof=1)),
                         mde_absolute=float(
                             alpha_z * pm.std(ddof=1) / np.sqrt(len(pm))))
    return out


def qualification(cells_by_cond):
    """m(strong resume) - m(weak resume), name held fixed.

    One observation per (pair, side): the same name under both templates. This
    is the specificity control -- a contrast built from the identical calls that
    produced the demographic contrast, on the identical scale.
    """
    out = {}
    for cond, cells in cells_by_cond.items():
        by = defaultdict(dict)
        for r in cells:
            for side in ("white", "black"):
                by[(r["pair"], side)][r["template"]] = r[f"{side}_margin"]
        d = np.array([v["T1_strong"] - v["T3_marginal"] for v in by.values()
                      if "T1_strong" in v and "T3_marginal" in v], dtype=float)
        if len(d) == 0:
            continue
        bt = es.boot_ci(d, lambda a: float(a.mean()), N_BOOT, RNG)
        out[cond] = dict(n=len(d), est=bt["est"], ci=bt["ci"])
    return out


def baseline_condition(label):
    """Which condition a family member tests against D0, or None.

    MIRRORS THE RULE THE PAPER'S SECTION 7 USES TO BUILD ITS TABLE, on purpose.
    The membership it produces is written into the artifact so the table and
    this interval can be checked against each other rather than assumed to
    agree; two files applying the same rule from memory is how a table and its
    caption come apart.

    D8 and D9 carry no contrast labelled "vs base". They are tested as P1 and
    P2, which is the same comparison under a different name, and matching on
    the label prefix alone drops them -- which would remove from the control
    class exactly the two conditions that displace the name without touching a
    delimiter, the controls the position confound exists to supply.
    """
    tag = label.split()[0]
    if tag.startswith("D") and "vs base" in label:
        return tag
    if label.startswith("P1"):
        return "D8"
    if label.startswith("P2"):
        return "D9"
    return None


def bh_rows(p):
    """Benjamini-Hochberg along the last axis, one family per row.

    es.benjamini_hochberg is the definition and stays the definition. This is
    the same procedure vectorised, because the significance RATE has to be
    re-decided inside every bootstrap replicate and 40,000 x 168 sequential
    adjustments is an hour of Python for an answer that is a few seconds of
    numpy. The artifact records the two agreeing on the observed family rather
    than asserting they must.
    """
    m = p.shape[-1]
    order = np.argsort(p, axis=-1)
    ps = np.take_along_axis(p, order, axis=-1)
    adj = ps * m / np.arange(1, m + 1, dtype=float)
    adj = np.minimum(np.minimum.accumulate(adj[..., ::-1], axis=-1)[..., ::-1], 1.0)
    unsorted = np.empty_like(adj)
    np.put_along_axis(unsorted, order, adj, axis=-1)
    return unsorted


def class_contrast(rows, out, n_boot=N_BOOT, rng=None):
    """An interval on the DIFFERENCE between the two condition classes, and the
    smallest difference this design could have found.

    WHY THIS EXISTS. The paper splits every condition-versus-baseline test into
    the conditions that destroy a structural delimiter and the conditions that
    change nothing structural, prints the two class means side by side, and
    concludes that no delimiter mechanism is present. Nothing was ever estimated
    ON THE DIFFERENCE between them. Two point estimates and two failures to
    reject were being read as an assertion of absence -- which is the inference
    this paper spends a whole section warning other people against. A null needs
    an interval on the contested quantity and a floor beneath it, or it is not a
    result, and the size of mechanism that survives is exactly what a reader
    needs in order to disagree with us.

    THE RESAMPLING UNIT IS THE NAME PAIR, AND IT HAS TO BE. The baseline tests
    are not independent: every one of them is measured on the same matched name
    pairs, so the two class means move together, and an interval that treated
    the tests as independent draws would be wrong by an unknown factor in an
    unknown direction. Resampling pairs -- the same drawn pairs used for every
    test within a replicate -- carries that shared dependence into the
    difference. Because each pair contributes exactly one cell per template to
    every contrast, the cluster bootstrap of es.boot_ci is algebraically
    identical to a bootstrap over the pair-level means, which is what is done
    here and is why it is affordable.

    WHY THE SIGNIFICANCE RATE NEEDS A PLUG-IN P-VALUE. That rate is a function
    of p_BH, so its bootstrap distribution requires re-deciding all 168 tests
    inside every replicate, and a bootstrap p-value inside a bootstrap is 1.6
    billion resamples. For a sample mean the nonparametric bootstrap standard
    error has a closed form, sd(x, ddof=0)/sqrt(k), so the normal plug-in p is
    available with no simulation at all. It is not assumed adequate:
    `plugin_p_check` in the artifact records what it does on the observed data,
    against the stored bootstrap values it is standing in for.

    THE CLOSED-FORM MDD IS OPTIMISTIC HERE, AND IS NOT THE ONE TO QUOTE. The
    usual 2.80 x SE assumes a statistic that is normal with the same dispersion
    under the null and under the alternative. Mean |effect| is neither: |.| is
    convex, so near zero the sampling distribution is folded and skewed, and it
    unfolds -- widening -- as a real class difference pushes it away from zero.
    Measured by injecting a mechanism of known size into a null-centred copy of
    this panel and counting how often the interval excludes zero, 2.80 x SE
    lands well short of 80% power; the shortfall is recorded as
    `mdd_80_calibration_factor` and the whole measured power curve is kept in
    `calibration.power_curve` so the claim is checkable rather than asserted.
    The simulated value is what `mdd_80_calibrated_*` reports and what the
    section's claim should rest on;
    the closed form is kept beside it, named as the closed form, because it is
    what `mde()` above computes and a reader comparing the two should be able
    to see the gap rather than wonder about it.

    THE SAME MEASUREMENT SHOWS THE PERCENTILE INTERVAL IS SLIGHTLY NARROW. Its
    true rejection rate under a genuine null is about 7-9%, not 5%, so its upper
    end is a little tighter than 95% coverage would warrant and quoting it as
    the largest surviving mechanism would overstate what was ruled out. The
    bound reported for that purpose is obtained by inverting the test instead:
    the largest true class difference under which the observed statistic would
    still fall inside the central 95% of its sampling distribution. Only the
    upper limit is identified -- the statistic is U-shaped in the size of the
    injected mechanism, because a large negative one drives the destroying class
    away from zero in the other direction -- and the upper limit is the one a
    claim of absence needs.

    MDD, NOT MDE. `mde()` above is the smallest shift from baseline a SINGLE
    condition could have shown. This is the smallest DIFFERENCE BETWEEN THE TWO
    CLASSES the comparison could have found. Both are kept: they bound different
    sentences, and it is this one that bounds the claim the section makes.

    WHAT THE BOUND IS NOT. It is a bound on the CLASS AVERAGE, over every model
    and both inference modes. A mechanism present in one checkpoint and absent
    in the other five would be diluted six-fold here and could exceed the bound
    while still fitting the data, so this number licenses "no delimiter
    mechanism across the panel" and not "no delimiter mechanism in any model".
    The per-model floors that bound the latter are `mde()`, which is why both
    stay in the artifact.

    D7 IS REPORTED THREE WAYS, BECAUSE ITS CLASS HAS ALREADY MOVED ONCE. D7 was
    designed as a null control and the tokenizer audit found it is not one: it
    replaces both merged delimiter tokens with a different single token,
    fragmenting nothing but leaving no instance of the original delimiter. It is
    a substitution, which is neither of the two things the classes name. Every
    scheme is therefore computed in full and none is left implicit:

      fragmented_vs_intact       D7 in neither class. This is what the paper's
                                 condition-class table splits on once it reads
                                 the tokenizer, and it is the primary.
      d7_counted_as_control      D7 among the conditions that changed nothing
                                 structural. What the table split on before.
      d7_counted_as_destroying   D7 among the delimiter-destroying conditions.

    The primary scheme's numbers are also hoisted to the top of the block so a
    caller cannot wire the paper to a scheme by accident. The bound moves by
    less than a third across all three, which is the useful result: the
    section's conclusion does not turn on the disputed condition.
    """
    rng = rng or np.random.default_rng(20260802)
    M, cond_of, members, templates = class_panel(rows)
    k = M.shape[1]

    # ---- does the plug-in stand in for the bootstrap p it replaces? --------
    stored_p = np.array([out[m][mo]["contrasts"][lab]["p"] for m, mo, lab in members])
    stored_bh = np.array([out[m][mo]["contrasts"][lab]["p_bh"] for m, mo, lab in members])
    stored_est = np.array([out[m][mo]["contrasts"][lab]["logodds"] for m, mo, lab in members])
    est_pi = M.mean(axis=1)
    p_pi = np.maximum(
        2.0 * sps.norm.sf(np.abs(est_pi / (M.std(axis=1, ddof=0) / np.sqrt(k)))),
        1.0 / n_boot)
    bh_ref = np.asarray(es.benjamini_hochberg(list(p_pi)))
    bh_vec = bh_rows(p_pi[None, :])[0]

    # ---- anchors that turn a log-odds bound into a sentence ----------------
    base_med = float(np.median([abs(out[m][mo]["per_condition"]["D0"]["logodds"]["est"])
                                for m in out for mo in out[m]]))
    qual_med = float(np.median([abs(out[m][mo]["qualification"]["D0"]["est"])
                                for m in out for mo in out[m]
                                if "D0" in out[m][mo].get("qualification", {})]))

    intact = tuple(c for c in CLASS_CONTROL if c != "D7")
    schemes = {}
    for name, dset, cset, note in (
            ("fragmented_vs_intact", CLASS_DESTROY, intact,
             "D7 in neither class: the tokenizer calls it a substitution, which "
             "is neither fragmentation nor an untouched delimiter"),
            ("d7_counted_as_control", CLASS_DESTROY, CLASS_CONTROL,
             "D7 among the conditions that changed nothing structural, as the "
             "condition was declared rather than as it tokenises"),
            ("d7_counted_as_destroying", CLASS_DESTROY + ("D7",), intact,
             "D7 among the delimiter-destroying conditions, on the ground that "
             "no instance of the original delimiter token survives it")):
        s = class_statistics(M, cond_of, dset, cset, n_boot, rng)
        s.update(calibrate(M, cond_of, dset, cset, rng))
        # how far the closed form was out. A reader who has met 2.80 x SE
        # elsewhere in this paper is owed the size of the correction, not only
        # the corrected number.
        s["mdd_80_calibration_factor"] = (s["mdd_80_calibrated_mean_abs_effect"]
                                          / s["mdd_80_closed_form_mean_abs_effect"])
        s["compatible_mechanism_upper_limit_pct_of_baseline"] = (
            100.0 * s["compatible_mechanism_upper_limit_logodds"] / base_med)
        s["mdd_80_calibrated_pct_of_baseline"] = (
            100.0 * s["mdd_80_calibrated_mean_abs_effect"] / base_med)
        s["mdd_80_calibrated_pct_of_qualification_range"] = (
            100.0 * s["mdd_80_calibrated_mean_abs_effect"] / qual_med)
        s["note"] = note
        schemes[name] = s

    return dict(
        resampling_unit="name pair",
        n_pairs=int(k), n_templates=len(templates), n_boot=int(n_boot),
        bh_family_size=int(M.shape[0]),
        baseline_effect_median_abs_logodds=base_med,
        qualification_range_median=qual_med,
        primary_scheme=PRIMARY_SCHEME,
        schemes=schemes,
        plugin_p_check=dict(
            max_abs_estimate_difference=float(np.abs(est_pi - stored_est).max()),
            max_abs_p_difference=float(np.abs(p_pi - stored_p).max()),
            max_abs_bh_vectorised_vs_reference=float(np.abs(bh_vec - bh_ref).max()),
            n_bh_decisions=int(len(stored_bh)),
            n_bh_decisions_matched=int(((bh_ref < .05) == (stored_bh < .05)).sum()),
            n_significant_stored=int((stored_bh < .05).sum()),
            n_significant_plugin=int((bh_ref < .05).sum())),
        # HOISTED so the paper cannot wire itself to a scheme by accident. These
        # are the primary scheme's numbers and nothing else's.
        **{kk: vv for kk, vv in schemes[PRIMARY_SCHEME].items() if kk != "note"})


def class_panel(rows):
    """The whole BH family reduced to one row per test, one column per name pair.

    Each entry is that pair's mean contrast value over the templates. This is
    not an approximation: every pair contributes the same number of cells to
    every contrast, so a cluster bootstrap that resamples pairs and concatenates
    their rows produces exactly the mean of the sampled pair-level means. Making
    that reduction explicit is what turns a nested resampling problem into two
    matrix products, which is the only reason the calibration below is
    affordable. A pair missing from a contrast would break the equality, so such
    a contrast is dropped whole rather than averaged over a different set of
    templates than its neighbours.
    """
    pairs = sorted({r["pair"] for r in rows})
    templates = sorted({r["template"] for r in rows})
    cols, cond_of, members = [], [], []
    for model in sorted({r["model"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows if r["model"] == model}):
            by = {}
            for r in rows:
                if r["model"] == model and r["mode"] == mode:
                    by.setdefault(r["cond"], {})[(r["template"], r["pair"])] = (
                        r["white_margin"] - r["black_margin"])
            for label, a, b in CONTRASTS:
                if a not in by or b not in by:
                    continue
                col = []
                for p in pairs:
                    v = [by[a][(t, p)] - by[b][(t, p)] for t in templates
                         if (t, p) in by[a] and (t, p) in by[b]]
                    if not v:
                        col = None
                        break
                    col.append(float(np.mean(v)))
                if col is None:
                    continue
                cols.append(col)
                cond_of.append(baseline_condition(label) or "-")
                members.append((model, mode, label))
    return (np.asarray(cols, dtype=float), np.asarray(cond_of), members,
            templates)


def class_statistics(M, cond_of, destroying, control, n_boot, rng):
    """Both class rows, both differences, and percentile intervals on each."""
    k = M.shape[1]
    inD = np.isin(cond_of, np.asarray(destroying))
    inC = np.isin(cond_of, np.asarray(control))

    def statistics(sub):
        """Every class statistic at once, for a stack of resampled panels."""
        est = sub.mean(axis=-1)
        se = np.maximum(sub.std(axis=-1, ddof=0) / np.sqrt(sub.shape[-1]), 1e-300)
        # floored at 1/n_boot for the same reason es.pvalue_from_boots floors:
        # the stored p-values it must rank alongside cannot resolve finer
        p = np.maximum(2.0 * sps.norm.sf(np.abs(est / se)), 1.0 / n_boot)
        sig = bh_rows(p) < 0.05
        raw = p < 0.05
        e = np.abs(est)
        mD, mC = e[..., inD].mean(-1), e[..., inC].mean(-1)
        return np.stack([mD - mC,
                         sig[..., inD].mean(-1) - sig[..., inC].mean(-1),
                         raw[..., inD].mean(-1) - raw[..., inC].mean(-1),
                         mD / np.maximum(mC, 1e-300),
                         mD, mC,
                         sig[..., inD].mean(-1), sig[..., inC].mean(-1)], axis=-1)

    observed = statistics(M[None, :, :])[0]
    boots = np.empty((n_boot, observed.shape[0]))
    step = 500                      # chunked so the index array stays small
    for i in range(0, n_boot, step):
        n = min(step, n_boot - i)
        idx = rng.integers(0, k, size=(n, k))
        boots[i:i + n] = statistics(np.swapaxes(M[:, idx], 0, 1))

    def summarise(j, null=0.0, mdd=True):
        b = boots[:, j]
        d = dict(est=float(observed[j]),
                 ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                 se=float(b.std(ddof=1)), boot_mean=float(b.mean()), null=null,
                 p=float(es.pvalue_from_boots(b, float(observed[j]), null, n_boot)))
        if mdd:
            d["mdd_80_closed_form"] = float(Z_POWER_80 * b.std(ddof=1))
        return d

    mean_diff = summarise(0)
    sig_diff = summarise(1)
    return dict(
        conditions_delimiter_destroying=list(destroying),
        conditions_nothing_structural=list(control),
        n_tests_delimiter_destroying=int(inD.sum()),
        n_tests_nothing_structural=int(inC.sum()),
        mean_abs_effect_delimiter_destroying=float(observed[4]),
        mean_abs_effect_nothing_structural=float(observed[5]),
        sig_rate_delimiter_destroying=float(observed[6]),
        sig_rate_nothing_structural=float(observed[7]),
        mean_abs_effect_difference=mean_diff,
        sig_rate_difference=sig_diff,
        sig_rate_difference_unadjusted_p=summarise(2),
        mean_abs_effect_ratio=summarise(3, null=1.0, mdd=False),
        mdd_80_closed_form_mean_abs_effect=mean_diff["mdd_80_closed_form"],
        mdd_80_closed_form_sig_rate=sig_diff["mdd_80_closed_form"])


def calibrate(M, cond_of, destroying, control, rng, n_sim=4000, n_inner=2000,
              n_invert=40000, target=0.80):
    """What the interval and the power formula actually do, measured not assumed.

    THE CONSTRUCTION. Subtract from the destroying class the constant that makes
    the two class means of |effect| exactly equal: that is a copy of this panel,
    with this panel's dispersion and correlation structure, in which the
    mechanism is known to be absent. Injecting a constant `s` into the
    destroying class then puts a mechanism of known size back in, and because
    |m + s.sgn(m)| = |m| + s the true class difference under injection is
    exactly `s` -- no calibration curve is needed to interpret the axis.

    WHY IT IS AFFORDABLE. Resampling name pairs is a multinomial draw of pair
    multiplicities, and every statistic here is a mean over pairs, so a panel
    and all of its bootstrap replicates are two matrix products. Resampling a
    resample is again multinomial, with probabilities proportional to the outer
    multiplicities, so the whole double bootstrap stays closed-form. That buys
    4,000 simulated panels x 2,000 replicates, which locates the 80% point to
    three digits instead of leaving it to a formula that turns out not to hold.

    WHAT IT REPORTS. The true rejection rate of the reported interval when the
    mechanism is genuinely absent; the injected size at which the interval
    attains `target` power; and, by inverting the test, the largest true class
    difference under which the observed statistic would still lie inside the
    central 95% of its sampling distribution. The last of those, not the
    bootstrap interval's upper end, is what bounds the surviving mechanism.
    """
    k = M.shape[1]
    inD = np.isin(cond_of, np.asarray(destroying))
    inC = np.isin(cond_of, np.asarray(control))
    obs_means = M.mean(axis=1)
    sgn = np.sign(obs_means)
    sgn[sgn == 0] = 1.0

    # THE NULL-CENTRING ROOT HAS TO BE BRACKETED, NOT GUESSED AT. Subtracting
    # `s` in each contrast's own direction changes the class mean by
    # mean(||obs| - s|), which is CONVEX in s: past the median of |obs| the
    # terms start folding through zero and the class mean rises again, so there
    # is a second root that centres nothing. Bisection over a wide bracket walks
    # to it, produces a "null world" with an enormous class difference, and the
    # calibration then reports 100% power at zero injection. Bracket the search
    # on the decreasing branch, which ends at the median, and fail loudly rather
    # than return a number from the wrong root.
    def gap(s):
        e = np.abs(obs_means - s * sgn * inD)
        return float(e[inD].mean() - e[inC].mean())

    lo = -(float(np.abs(obs_means).max()) + 1.0)
    hi = float(np.median(np.abs(obs_means[inD])))
    if not (gap(lo) > 0 >= gap(hi)):
        raise RuntimeError("no null-centring shift on the decreasing branch: "
                           f"gap({lo:.4f})={gap(lo):.4g}, gap({hi:.4f})={gap(hi):.4g}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    centring = 0.5 * (lo + hi)
    if abs(gap(centring)) > 1e-12:
        raise RuntimeError(f"null-centring did not converge: gap={gap(centring):.4g}")
    # |m| == |m * sgn|, so folding the sign in makes the injection a plain +s on
    # the destroying rows and lets one set of draws serve the whole shift grid
    A = (M - (centring * sgn * inD)[:, None]) * sgn[:, None]
    D_idx, C_idx = np.flatnonzero(inD), np.flatnonzero(inC)

    shifts = np.arange(0.0, 0.0301, 0.0005)
    sh = shifts[:, None, None]
    reject = np.zeros((len(shifts), n_sim), dtype=bool)
    uniform = np.full(k, 1.0 / k)
    for s in range(n_sim):
        c_out = rng.multinomial(k, uniform)
        c_in = rng.multinomial(k, c_out / k, size=n_inner)
        m_in = (c_in @ A.T) / k
        b = (np.abs(m_in[None, :, D_idx] + sh).mean(axis=2)
             - np.abs(m_in[:, C_idx]).mean(axis=1)[None, :])
        q_lo, q_hi = np.percentile(b, [2.5, 97.5], axis=1)
        reject[:, s] = (q_lo > 0) | (q_hi < 0)
    power = reject.mean(axis=1)

    j = int(np.searchsorted(power, target))
    if 0 < j < len(shifts):
        x0, x1, y0, y1 = shifts[j - 1], shifts[j], power[j - 1], power[j]
        mdd = float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    else:
        mdd = float("nan")

    # test inversion, on the branch where the statistic is monotone in the size
    # of the injected mechanism (see the caller's docstring: only the upper
    # limit is identified, and only the upper limit is claimed)
    e_obs = np.abs(obs_means)
    t_obs = float(e_obs[inD].mean() - e_obs[inC].mean())
    mm = (rng.multinomial(k, uniform, size=n_invert) @ A.T) / k
    grid = np.arange(0.0, 0.0601, 0.00025)
    q025 = np.array([np.percentile(np.abs(mm[:, D_idx] + d).mean(1)
                                   - np.abs(mm[:, C_idx]).mean(1), 2.5)
                     for d in grid])
    monotone = bool(np.all(np.diff(q025) > 0))
    if not monotone:
        raise RuntimeError("inversion branch is not monotone; np.interp would "
                           "return a value from the wrong branch")
    upper = float(np.interp(t_obs, q025, grid))
    if not (q025[0] <= t_obs <= q025[-1]):
        raise RuntimeError(f"observed statistic {t_obs:.5g} outside the inversion "
                           f"grid [{q025[0]:.5g}, {q025[-1]:.5g}]")
    return dict(
        interval_true_rejection_rate_at_null=float(power[0]),
        mdd_80_calibrated_mean_abs_effect=mdd,
        compatible_mechanism_upper_limit_logodds=upper,
        calibration=dict(n_sim=int(n_sim), n_inner=int(n_inner),
                         n_invert=int(n_invert), target_power=target,
                         null_centring_shift=float(centring),
                         inversion_branch_monotone=monotone,
                         power_curve={f"{s:.4f}": float(p)
                                      for s, p in zip(shifts, power)}))


def main() -> int:
    rows = load()
    if not rows:
        sys.exit(f"no data in {DATA}")

    out = {}
    family = []          # (model, mode, label, p, holder-dict)

    for model in sorted({r["model"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows if r["model"] == model}):
            mr = [r for r in rows if r["model"] == model and r["mode"] == mode]
            by = {c: [r for r in mr if r["cond"] == c] for c in ORDER}
            by = {c: v for c, v in by.items() if v}
            if "D0" not in by:
                continue
            per = {c: es.describe(v, N_BOOT, RNG) for c, v in by.items()}
            cons = {}
            for label, a, b in CONTRASTS:
                if a in by and b in by:
                    c = es.paired_contrast(by[a], by[b], ("template", "pair"),
                                           N_BOOT, RNG)
                    if c:
                        cons[label] = c
                        family.append((model, mode, label, c["p"], c))

            # dose-response, reported two ways because they disagree
            base = per["D0"]["logodds"]["est"]
            xs = [NDELIM[c] for c in per]
            ys = [abs(per[c]["logodds"]["est"] - base) for c in per]
            r_all = float(np.corrcoef(xs, ys)[0, 1]) if len(set(xs)) > 1 else None
            # permutation p for the correlation: 8 points is few enough to be
            # exactly permuted, so no distributional assumption is needed
            p_dose = None
            if r_all is not None:
                perm = []
                ya = np.array(ys)
                for _ in range(20000):
                    perm.append(abs(np.corrcoef(xs, RNG.permutation(ya))[0, 1]))
                p_dose = float((np.array(perm) >= abs(r_all)).mean())

            xs2 = [NSHIFT[c] for c in per]
            r_pos = (float(np.corrcoef(xs2, ys)[0, 1])
                     if len(set(xs2)) > 1 else None)

            out.setdefault(model, {})[mode] = dict(
                dose_r_nameshift=r_pos,
                n_pairs_per_condition=per["D0"]["n"],
                per_condition={c: {k: v for k, v in e.items()} for c, e in per.items()},
                contrasts=cons,
                dose_r=r_all, dose_p_permutation=p_dose,
                qualification=qualification(by),
                mde=mde(by))

    # ---- one BH adjustment over the whole family --------------------------
    ps = [f[3] for f in family]
    adj = es.benjamini_hochberg(ps)
    for (model, mode, label, _p, holder), a in zip(family, adj):
        holder["p_bh"] = a
    for m in out:
        for mode in out[m]:
            out[m][mode]["bh_family_size"] = len(ps)

    # ---- the interval on the difference the paper's conclusion is about ----
    # A SEPARATE RNG, DELIBERATELY. Drawing from the module RNG here would shift
    # every subsequent value in its stream, so adding this analysis would have
    # silently moved numbers already in the paper. Seeded independently, every
    # pre-existing figure in this artifact is bit-identical to before.
    #
    # STORED IN EVERY MODEL x MODE BLOCK, like bh_family_size above. It is a
    # family-level quantity and the artifact's top level is a map of model names
    # that several readers iterate over; a top-level key of a different shape
    # would break them.
    dcc = class_contrast(rows, out, N_BOOT, np.random.default_rng(20260801))
    for m in out:
        for mode in out[m]:
            out[m][mode]["delimiter_class_contrast"] = dcc

    # ---- report -----------------------------------------------------------
    print("=" * 100)
    print(f"STUDY 5. Mechanism panel. BH family = {len(ps)} contrasts, adjusted together.")
    print("Effects are the paired demographic margin difference in log-odds.")
    print("=" * 100)

    for model in out:
        for mode in out[model]:
            r = out[model][mode]
            print(f"\n{'-'*100}\n{model}   mode={mode}   "
                  f"n={r['n_pairs_per_condition']} pairs per condition\n{'-'*100}")
            base = r["per_condition"]["D0"]["logodds"]["est"]
            print(f"  {'cond':<5}{'ndel':>5}{'name':>6}{'delta':>10}{'vs base':>10}"
                  f"{'Ps':>8}   {'condition'}")
            for c in ORDER:
                if c not in r["per_condition"]:
                    continue
                e = r["per_condition"][c]
                print(f"  {c:<5}{NDELIM[c]:>5}{NSHIFT[c]:>+6}"
                      f"{e['logodds']['est']:>+10.4f}"
                      f"{e['logodds']['est']-base:>+10.4f}"
                      f"{e['superiority']['est']:>8.3f}   {NOTE[c]}")
            print()
            for label, c in r["contrasts"].items():
                star = " *" if c["p_bh"] < 0.05 else ""
                print(f"    {label:<52}{c['logodds']:>+8.4f} "
                      f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]  "
                      f"p={c['p']:.4f} p_BH={c['p_bh']:.4f}{star}")
            if r["dose_r"] is not None:
                print(f"    {'dose-response r (|shift| vs n destroyed)':<52}"
                      f"{r['dose_r']:>+8.3f}                    "
                      f"permutation p={r['dose_p_permutation']:.4f}")
            mm = r.get("mde", {})
            q0 = r["qualification"].get("D0", {}).get("est")
            if mm and q0:
                worst = max(v["mde_absolute"] for v in mm.values())
                print()
                print(f"    MDE at 80% power, worst across conditions: "
                      f"{worst:.4f} absolute, "
                      f"{100*worst/abs(q0):.1f}% of this model's qualification range "
                      f"({q0:+.3f})")
            q = r["qualification"]
            if q:
                qb = q.get("D0", {}).get("est")
                print(f"\n    QUALIFICATION CONTRAST m(strong)-m(weak), same calls, "
                      f"n={q['D0']['n']} per condition")
                print(f"      {'cond':<6}{'qual':>10}{'vs base':>10}")
                for c in ORDER:
                    if c in q:
                        print(f"      {c:<6}{q[c]['est']:>+10.3f}"
                              f"{q[c]['est']-qb:>+10.3f}")

    # ---- the two rows of the paper's class table, and the gap between them --
    print("\n" + "=" * 100)
    print("THE DIFFERENCE BETWEEN THE TWO CONDITION CLASSES, resampling name "
          f"pairs (n={dcc['n_pairs']})")
    print("=" * 100)
    print(f"  {'condition class':<30}{'tests':>7}{'sig. after BH':>16}"
          f"{'mean |effect|':>16}")
    print(f"  {'delimiter destroyed':<30}"
          f"{dcc['n_tests_delimiter_destroying']:>7}"
          f"{dcc['sig_rate_delimiter_destroying'] * 100:>15.1f}%"
          f"{dcc['mean_abs_effect_delimiter_destroying']:>16.4f}")
    print(f"  {'nothing structural changed':<30}"
          f"{dcc['n_tests_nothing_structural']:>7}"
          f"{dcc['sig_rate_nothing_structural'] * 100:>15.1f}%"
          f"{dcc['mean_abs_effect_nothing_structural']:>16.4f}")
    md, sd_ = dcc["mean_abs_effect_difference"], dcc["sig_rate_difference"]
    rt = dcc["mean_abs_effect_ratio"]
    print(f"\n  difference in mean |effect|   {md['est']:>+9.4f} "
          f"[{md['ci'][0]:+.4f},{md['ci'][1]:+.4f}]  p={md['p']:.3f}")
    print(f"  ratio of mean |effect|        {rt['est']:>9.3f} "
          f"[{rt['ci'][0]:.3f},{rt['ci'][1]:.3f}]  p={rt['p']:.3f}")
    print(f"  difference in sig. rate      {sd_['est'] * 100:>+9.2f} pp "
          f"[{sd_['ci'][0] * 100:+.2f},{sd_['ci'][1] * 100:+.2f}] pp  "
          f"p={sd_['p']:.3f}")
    print("\n  MINIMUM DETECTABLE CLASS DIFFERENCE at 80% power, two-sided 0.05")
    print(f"    mean |effect|, CALIBRATED  "
          f"{dcc['mdd_80_calibrated_mean_abs_effect']:.5f} log-odds"
          f"   ({dcc['mdd_80_calibrated_pct_of_baseline']:.1f}% of the median "
          f"baseline demographic effect, "
          f"{dcc['baseline_effect_median_abs_logodds']:.4f})")
    print(f"    mean |effect|, closed form "
          f"{dcc['mdd_80_closed_form_mean_abs_effect']:.5f}  "
          f"(2.80 x SE understates by {dcc['mdd_80_calibration_factor']:.2f}x; "
          f"the interval's true rejection rate under a genuine null is "
          f"{dcc['interval_true_rejection_rate_at_null'] * 100:.1f}%, not 5%)")
    print(f"    sig. rate, closed form     "
          f"{dcc['mdd_80_closed_form_sig_rate'] * 100:.1f} pp  (uncalibrated)")
    print("  LARGEST DELIMITER MECHANISM COMPATIBLE WITH THE DATA, by test inversion")
    print(f"    {dcc['compatible_mechanism_upper_limit_logodds']:.5f} log-odds"
          f"  = {dcc['compatible_mechanism_upper_limit_pct_of_baseline']:.1f}% of "
          f"the baseline demographic effect")
    print(f"  EVERY SPLIT OF D7 (primary = {dcc['primary_scheme']})")
    for nm, s in dcc["schemes"].items():
        star = " *" if nm == dcc["primary_scheme"] else "  "
        print(f"   {star}{nm:<26} {s['n_tests_delimiter_destroying']:>3}v"
              f"{s['n_tests_nothing_structural']:<3}"
              f" difference {s['mean_abs_effect_difference']['est']:+.5f}"
              f" [{s['mean_abs_effect_difference']['ci'][0]:+.5f},"
              f"{s['mean_abs_effect_difference']['ci'][1]:+.5f}]"
              f"  MDD {s['mdd_80_calibrated_mean_abs_effect']:.5f}"
              f"  upper {s['compatible_mechanism_upper_limit_logodds']:.5f}")
    pc = dcc["plugin_p_check"]
    print(f"  plug-in p check: estimates agree to {pc['max_abs_estimate_difference']:.1e}, "
          f"p to {pc['max_abs_p_difference']:.4f}, "
          f"{pc['n_bh_decisions_matched']}/{pc['n_bh_decisions']} BH verdicts identical")

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
