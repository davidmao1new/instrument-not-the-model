"""Study 2, re-analysed on defensible scales, plus two analyses the design
already supported and the first pass did not run.

SUPERSEDES analyze_delta_stability.py and analyze_margin.py. Those remain on
disk for provenance. What changed and why is in effectsize.py; in one line, the
percentage-point conversion was evaluated at p = 0.5 for four models of which
only one is anywhere near p = 0.5.

This script produces four things.

1. THE EFFECT, ON EVERY SCALE. Per model and per wording: the log-odds margin
   difference, the probability of superiority, and a percentage-point interval
   rather than a percentage-point number.

2. THE SPECIFICATION CURVE (gap G9). Twelve wordings x three resume templates is
   thirty-six defensible ways to run this study, and a researcher publishing one
   number picked one of them. Simonsohn, Simmons and Nelson (2020) named the
   right display for this situation: estimate every specification, sort by point
   estimate, and report what fraction of the paths a researcher could honestly
   have taken would have yielded a significant effect, and in which direction.
   sigma_variant answers the same question in one number; the curve answers it
   in a way a reader can audit.

3. THE SHORTLIST BRIDGE (gap G8). A margin is not a decision. Screening
   shortlists, so the decision-relevant quantity is who ends up in the top k.
   Ranking every candidate by margin and cutting at k gives a selection-rate gap
   in percentage points that IS commensurable with a callback-rate gap of the
   correspondence-audit kind, which nothing else in this paper is. It is
   reported at three cut points, because k is one more researcher degree of
   freedom and hiding it inside a single number would repeat the error the paper
   is about.

4. THE BINARY ARM'S DEGENERACY, STATED (gap G12). Llama-2-7B-chat returns the
   grammar-constrained verdict "yes" in 99.8% of calls, so delta, psi and the
   McNemar p are structurally 0, 0 and 1 for that model. A table of zeros reads
   like a carefully estimated null. It is not one: it is an instrument with no
   dynamic range left, and the paper has to say so in those words.

    .venv/Scripts/python.exe paper-a/src/analyze_study2_v2.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import effectsize as es  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data" / "delta_stability"
OUT = DATA / "study2_v2.json"
RNG = np.random.default_rng(20260728)

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
}
CUTS = (0.10, 0.25, 0.50)
ALPHA = 0.05


def load() -> list[dict]:
    """Complete rows only, deduplicated to the last write per cell.

    The glob is deliberately narrow. `_binary_only_superseded/` sits inside this
    directory and contains an earlier run of two models under a design that
    could not measure the estimand; a recursive glob would silently pool it.
    """
    rows = []
    for f in sorted(DATA.glob("delta_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    best = {}
    for r in rows:
        if r.get("white_margin") is None or r.get("black_margin") is None:
            continue
        best[(r["model"], r["variant"], r["template"], r["pair"])] = r
    return list(best.values())


def factor_decomposition(specs) -> dict:
    """Which of the two crossed factors carries the sign instability.

    WHY THIS EXISTS. The curve crosses twelve wordings WITH three résumé
    templates, and the curve was being read as a statement about wording alone
    -- "the same model, the same names, the same résumés, and a defensible
    choice of wording determines the direction". That sentence is not licensed
    by this design. The three templates are three DIFFERENT résumés: different
    degree, different GPA, different job title, different skills, deliberately
    built at different strength levels (experiment_delta_stability.TEMPLATES).
    So a sign change between two points on the curve may be a change of
    wording, a change of résumé, or both, and the curve as displayed cannot
    tell a reader which. Pooling two factors and then describing the result as
    if only one had varied is the error this paper is about; committing it in
    our own §4.1 would be worse than not running the analysis.

    THE DECOMPOSITION IS A CONTRAST CENSUS. A CONTRAST is a pair of the 36
    specifications that differs on exactly one factor and agrees on the other:
    3 x C(12,2) = 198 WORDING contrasts (same template, two wordings) and
    12 x C(3,2) = 36 TEMPLATE contrasts (same wording, two templates). A
    contrast FLIPS when the two point estimates carry opposite signs, and it
    flips REPORTABLY when both specifications are also significant after BH.
    Reportable is the one the prose may lean on, because it counts the pairs of
    papers that could BOTH have been published, in opposite directions, off
    this design. Raw flips are carried alongside so the paper cannot be caught
    claiming "no sign changes within a template" when a near-zero null estimate
    happens to sit on the other side of zero.

    Contrast rates are reported as well as counts, because 198 and 36 are not
    the same denominator and comparing the raw counts would favour whichever
    factor has more levels.

    THE SUM-OF-SQUARES SPLIT IS DESCRIPTIVE and is labelled as such. It
    apportions the spread of the 36 POINT ESTIMATES between the two factors
    additively, with the remainder holding interaction plus estimation error.
    It is not a variance-components model, it has no interval, and it must not
    be quoted as though it did; fit_variance_components.py is where an
    estimated σ with a posterior lives.
    """
    grid = {(s["variant"], s["template"]): s for s in specs}
    variants = sorted({s["variant"] for s in specs})
    templates = sorted({s["template"] for s in specs})
    # A contrast census only means what it says on a COMPLETE grid: a missing
    # cell would quietly shrink one factor's denominator and make that factor
    # look more stable than it is. Fail rather than report a biased census.
    if len(grid) != len(variants) * len(templates):
        sys.exit(f"spec grid is not fully crossed: {len(grid)} cells for "
                 f"{len(variants)} wordings x {len(templates)} templates")

    def direction(s) -> tuple[int, int]:
        """(sign of the estimate, sign a researcher could actually report)."""
        raw = 0 if s["logodds"] == 0 else (1 if s["logodds"] > 0 else -1)
        return raw, (raw if s["p_bh"] < ALPHA else 0)

    def census(groups, label_of) -> dict:
        """Unordered-pair census within each group of specifications."""
        by_level, pair_counts = {}, defaultdict(int)
        n = raw = rep = 0
        for gname, cells in groups.items():
            g_raw = g_rep = 0
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    a, b = cells[i], cells[j]
                    a_raw, a_rep = direction(a)
                    b_raw, b_rep = direction(b)
                    n += 1
                    if a_raw and b_raw and a_raw != b_raw:
                        g_raw += 1
                    if a_rep and b_rep and a_rep != b_rep:
                        g_rep += 1
                        pair_counts["|".join(sorted((label_of(a), label_of(b))))] += 1
            raw += g_raw
            rep += g_rep
            by_level[gname] = dict(
                n_contrasts=len(cells) * (len(cells) - 1) // 2,
                n_flips_raw=g_raw, n_flips_reportable=g_rep)
        return dict(
            n_contrasts=n, n_flips_raw=raw, n_flips_reportable=rep,
            flip_rate_raw=raw / n if n else float("nan"),
            flip_rate_reportable=rep / n if n else float("nan"),
            n_levels_with_reportable_flip=sum(
                1 for v in by_level.values() if v["n_flips_reportable"]),
            n_levels=len(by_level),
            reportable_flip_pairs=dict(sorted(pair_counts.items())),
            by_level=by_level)

    wording = census({t: [grid[(v, t)] for v in variants] for t in templates},
                     lambda s: s["variant"])
    template = census({v: [grid[(v, t)] for t in templates] for v in variants},
                      lambda s: s["template"])

    def marginal(levels, cells_of) -> dict:
        out = {}
        for lv in levels:
            cells = cells_of(lv)
            lo = [c["logodds"] for c in cells]
            out[lv] = dict(
                n=len(cells),
                n_sig_positive=sum(1 for c in cells
                                   if c["p_bh"] < ALPHA and c["logodds"] > 0),
                n_sig_negative=sum(1 for c in cells
                                   if c["p_bh"] < ALPHA and c["logodds"] < 0),
                mean_logodds=float(np.mean(lo)),
                min_logodds=float(min(lo)), max_logodds=float(max(lo)))
        return out

    by_template = marginal(templates, lambda t: [grid[(v, t)] for v in variants])
    by_wording = marginal(variants, lambda v: [grid[(v, t)] for t in templates])

    x = np.array([[grid[(v, t)]["logodds"] for t in templates] for v in variants],
                 dtype=float)
    grand = float(x.mean())
    ss_total = float(((x - grand) ** 2).sum())
    ss_template = float(len(variants) * ((x.mean(axis=0) - grand) ** 2).sum())
    ss_wording = float(len(templates) * ((x.mean(axis=1) - grand) ** 2).sum())
    share = (lambda s: s / ss_total if ss_total else float("nan"))

    w_rep, t_rep = wording["n_flips_reportable"], template["n_flips_reportable"]
    if not w_rep and not t_rep:
        carrier = "neither"
        claim = ("No two significant specifications disagree in sign, so the "
                 "curve does not support a sign-instability claim on either "
                 "factor for this model.")
    elif t_rep and not w_rep:
        carrier = "template"
        claim = ("Every reportable sign reversal is a change of résumé "
                 "template. Holding the template fixed, no choice of wording "
                 "reverses a significant effect.")
    elif w_rep and not t_rep:
        carrier = "wording"
        claim = ("Every reportable sign reversal is a change of wording within "
                 "one fixed résumé.")
    else:
        carrier = "both"
        claim = ("Reportable sign reversals occur both across wordings within a "
                 "fixed template and across templates within a fixed wording.")

    return dict(
        factors=dict(wording=variants, template=templates),
        across_wordings_within_template=wording,
        across_templates_within_wording=template,
        by_template=by_template, by_wording=by_wording,
        sign_instability_carrier=carrier, licensed_claim=claim,
        dispersion_share=dict(
            note=("Descriptive split of the spread of the 36 point estimates. "
                  "Not a variance-components model and carries no interval."),
            ss_total=ss_total, ss_template=ss_template, ss_wording=ss_wording,
            ss_residual=ss_total - ss_template - ss_wording,
            frac_template=share(ss_template), frac_wording=share(ss_wording),
            frac_residual=share(ss_total - ss_template - ss_wording)))


def spec_curve(mrows) -> dict:
    """Every (wording, template) specification, estimated independently."""
    specs = []
    by = defaultdict(list)
    for r in mrows:
        by[(r["variant"], r["variant_kind"], r["template"])].append(r)
    for (v, kind, t), cells in sorted(by.items()):
        d = es.paired_deltas(cells)
        bt = es.boot_ci(d, lambda a: float(a.mean()), 4000, RNG)
        ps, cliff = es.superiority(d)
        specs.append(dict(variant=v, kind=kind, template=t, n=len(d),
                          logodds=bt["est"], ci=bt["ci"],
                          p=es.pvalue_from_boots(bt["boots"], bt["est"], 0.0, 4000),
                          ps=ps, cliff=cliff))
    ps_adj = es.benjamini_hochberg([s["p"] for s in specs])
    for s, a in zip(specs, ps_adj):
        s["p_bh"] = a
    sig_pos = sum(1 for s in specs if s["p_bh"] < ALPHA and s["logodds"] > 0)
    sig_neg = sum(1 for s in specs if s["p_bh"] < ALPHA and s["logodds"] < 0)
    # sign_flips says only THAT the curve changes sign. It is kept unchanged --
    # the paper already cites it -- and factor_decomposition says WHICH of the
    # two crossed factors the change is attributable to, which is what the
    # prose actually needs and what sign_flips alone cannot license.
    return dict(specs=sorted(specs, key=lambda s: s["logodds"]),
                n_specs=len(specs), n_sig_positive=sig_pos, n_sig_negative=sig_neg,
                n_null=len(specs) - sig_pos - sig_neg,
                sign_flips=bool(sig_pos and sig_neg),
                range_logodds=[min(s["logodds"] for s in specs),
                               max(s["logodds"] for s in specs)],
                factor_decomposition=factor_decomposition(specs))


def shortlist(mrows) -> dict:
    """Selection-rate gap under a top-k shortlisting rule.

    Within each (wording, template) the candidates are the 2n name-instances of
    that cell, ranked by margin. Ties are broken by taking the higher rank for
    neither group: a tie block is credited to each group in proportion to its
    membership, so a model that cannot separate two candidates cannot be scored
    as preferring one.
    """
    out = {}
    for cut in CUTS:
        gaps_by_variant = defaultdict(list)
        by = defaultdict(list)
        for r in mrows:
            by[(r["variant"], r["template"])].append(r)
        for (v, t), cells in by.items():
            cand = ([(r["white_margin"], "W") for r in cells]
                    + [(r["black_margin"], "B") for r in cells])
            k = max(1, int(round(cut * len(cand))))
            cand.sort(key=lambda x: -x[0])
            thresh = cand[k - 1][0]
            above = [c for c in cand if c[0] > thresh]
            tied = [c for c in cand if c[0] == thresh]
            slots = k - len(above)
            sel = defaultdict(float)
            for _, g in above:
                sel[g] += 1.0
            if tied and slots > 0:
                for g in ("W", "B"):
                    n_g = sum(1 for _, gg in tied if gg == g)
                    if n_g:
                        sel[g] += slots * n_g / len(tied)
            n_w = sum(1 for _, g in cand if g == "W")
            n_b = len(cand) - n_w
            gaps_by_variant[v].append(100.0 * (sel["W"] / n_w - sel["B"] / n_b))
        per_v = {v: float(np.mean(g)) for v, g in gaps_by_variant.items()}
        vals = np.array(list(per_v.values()))
        out[f"top_{int(cut*100)}pct"] = dict(
            mean_gap_pp=float(vals.mean()),
            sd_across_wordings_pp=float(vals.std(ddof=1)),
            min_pp=float(vals.min()), max_pp=float(vals.max()),
            range_pp=float(vals.max() - vals.min()),
            sign_flips_across_wordings=bool((vals > 0).any() and (vals < 0).any()),
            per_variant=per_v)
    return out


def binary_health(mrows) -> dict:
    """Is the thresholded verdict carrying any information at all?"""
    yes = tot = 0
    disagree = 0
    for r in mrows:
        for s in ("white", "black"):
            v, m = r.get(s), r.get(s + "_margin")
            if v is None:
                continue
            tot += 1
            yes += (v == "yes")
            if m is not None and ((v == "yes") != (m > 0)):
                disagree += 1
    rate = yes / tot if tot else float("nan")
    return dict(n=tot, accept_rate=rate,
                degenerate=bool(rate > 0.98 or rate < 0.02),
                verdict_margin_disagreement=disagree / tot if tot else float("nan"))


def main() -> int:
    rows = load()
    if not rows:
        sys.exit(f"no data in {DATA}")
    out = {}
    for m in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == m]
        overall = es.describe(mrows, 10000, RNG)
        arms = {a: es.describe([r for r in mrows if r["variant_kind"] == a], 10000, RNG)
                for a in ("semantic", "null")}
        per_variant = {}
        for v in sorted({r["variant"] for r in mrows}):
            cells = [r for r in mrows if r["variant"] == v]
            per_variant[v] = es.describe(cells, 4000, RNG)
            per_variant[v]["kind"] = cells[0]["variant_kind"]
        ps_vals = np.array([per_variant[v]["superiority"]["est"] for v in per_variant])
        out[m] = dict(overall=overall, arms=arms, per_variant=per_variant,
                      ps_sd_across_wordings=float(ps_vals.std(ddof=1)),
                      ps_range=[float(ps_vals.min()), float(ps_vals.max())],
                      spec_curve=spec_curve(mrows), shortlist=shortlist(mrows),
                      binary=binary_health(mrows))

    print("=" * 100)
    print("STUDY 2 RE-ANALYSIS. Primary effect size is the probability of superiority Ps:")
    print("the fraction of matched pairs in which the White-named resume outranks the Black-named one.")
    print("=" * 100)
    print(f"{'model':<26}{'Ps':>7}{'95% CI':>16}{'Cliff d':>9}{'log-odds':>10}"
          f"{'pp interval':>18}{'saturated':>11}")
    for m, r in out.items():
        o = r["overall"]
        s = o["superiority"]
        print(f"{SHORT.get(m,m)[:25]:<26}{s['est']:>7.3f}"
              f"{f'[{s[chr(99)+chr(105)][0]:.3f},{s[chr(99)+chr(105)][1]:.3f}]':>16}"
              f"{s['cliff']:>+9.3f}{o['logodds']['est']:>+10.4f}"
              f"{f'[{o[chr(112)+chr(112)+chr(95)+chr(105)+chr(110)+chr(116)+chr(101)+chr(114)+chr(118)+chr(97)+chr(108)][0]:+.2f},{o[chr(112)+chr(112)+chr(95)+chr(105)+chr(110)+chr(116)+chr(101)+chr(114)+chr(118)+chr(97)+chr(108)][1]:+.2f}]':>18}"
              f"{o['saturated_frac']:>11.1%}")

    print("\nWORDING INSTABILITY, on the Ps scale (0.5 = no effect).")
    print(f"{'model':<26}{'Ps':>7}{'SD':>8}{'min':>7}{'max':>7}{'range':>8}"
          f"{'SD/|Ps-.5|':>12}   semantic / null SD")
    for m, r in out.items():
        sd = r["ps_sd_across_wordings"]
        ps = r["overall"]["superiority"]["est"]
        lo, hi = r["ps_range"]
        sem = [r["per_variant"][v]["superiority"]["est"]
               for v in r["per_variant"] if r["per_variant"][v]["kind"] == "semantic"]
        nul = [r["per_variant"][v]["superiority"]["est"]
               for v in r["per_variant"] if r["per_variant"][v]["kind"] == "null"]
        print(f"{SHORT.get(m,m)[:25]:<26}{ps:>7.3f}{sd:>8.4f}{lo:>7.3f}{hi:>7.3f}"
              f"{hi-lo:>8.3f}{sd/abs(ps-0.5):>12.2f}   "
              f"{np.std(sem, ddof=1):.4f} / {np.std(nul, ddof=1):.4f}"
              f"{'   <- null exceeds semantic' if np.std(nul,ddof=1)>np.std(sem,ddof=1) else ''}")

    print("\nSPECIFICATION CURVE. 12 wordings x 3 templates = 36 defensible analyses per model.")
    print(f"{'model':<26}{'specs':>7}{'sig +':>7}{'sig -':>7}{'null':>7}"
          f"{'sign flips?':>13}   log-odds range")
    for m, r in out.items():
        s = r["spec_curve"]
        print(f"{SHORT.get(m,m)[:25]:<26}{s['n_specs']:>7}{s['n_sig_positive']:>7}"
              f"{s['n_sig_negative']:>7}{s['n_null']:>7}{str(s['sign_flips']):>13}"
              f"   [{s['range_logodds'][0]:+.3f}, {s['range_logodds'][1]:+.3f}]")

    print("\nWHICH FACTOR CARRIES THE SIGN INSTABILITY. A contrast holds one factor fixed and")
    print("changes the other. 'Reportable' = both specifications significant after BH, i.e.")
    print("two papers that could both have been published, in opposite directions.")
    print(f"{'model':<26}{'wording flips':>16}{'template flips':>16}"
          f"{'SS templ':>10}{'SS word':>9}{'SS resid':>10}   carrier")
    for m, r in out.items():
        f = r["spec_curve"]["factor_decomposition"]
        w, t = f["across_wordings_within_template"], f["across_templates_within_wording"]
        d = f["dispersion_share"]
        wf = f"{w['n_flips_reportable']}/{w['n_contrasts']}"
        tf = f"{t['n_flips_reportable']}/{t['n_contrasts']}"
        print(f"{SHORT.get(m,m)[:25]:<26}{wf:>16}{tf:>16}"
              f"{d['frac_template']:>10.1%}{d['frac_wording']:>9.1%}"
              f"{d['frac_residual']:>10.1%}   {f['sign_instability_carrier']}")
        for tname, v in f["by_template"].items():
            print(f"    {tname:<14}sig +{v['n_sig_positive']:>3}  sig -{v['n_sig_negative']:>3}"
                  f"   mean log-odds {v['mean_logodds']:+.4f}"
                  f"   [{v['min_logodds']:+.4f}, {v['max_logodds']:+.4f}]")

    print("\nSHORTLIST BRIDGE. Rank by margin, take the top k, report the selection-rate gap.")
    print("This is the only quantity here commensurable with a callback-rate gap.")
    print(f"{'model':<26}{'cut':>8}{'gap pp':>9}{'SD across wordings':>20}"
          f"{'range pp':>11}{'flips sign?':>13}")
    for m, r in out.items():
        for cut, s in r["shortlist"].items():
            print(f"{SHORT.get(m,m)[:25]:<26}{cut:>8}{s['mean_gap_pp']:>+9.2f}"
                  f"{s['sd_across_wordings_pp']:>20.2f}{s['range_pp']:>11.2f}"
                  f"{str(s['sign_flips_across_wordings']):>13}")

    print("\nBINARY ARM HEALTH. A thresholded verdict with no dynamic range is not a null result.")
    print(f"{'model':<26}{'n calls':>9}{'accept rate':>13}{'degenerate?':>13}"
          f"{'verdict/margin disagree':>26}")
    for m, r in out.items():
        b = r["binary"]
        print(f"{SHORT.get(m,m)[:25]:<26}{b['n']:>9}{b['accept_rate']:>13.3f}"
              f"{str(b['degenerate']):>13}{b['verdict_margin_disagreement']:>26.3f}")

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
