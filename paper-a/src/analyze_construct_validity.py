"""Does the instrument respond to the construct it names?

WHY THIS EXISTS. Every study in this literature, including ours, computes a
difference between résumés carrying White-associated and Black-associated names
and calls the difference a demographic effect. That inference has a premise:
that the model encodes the name-to-race association the name list was validated
for, and that the between-name variation the audit averages over is the
construct rather than something else. The premise is never checked. It is a
manipulation check, and correspondence audits of humans do not need one because
the name list was validated ON humans -- Bertrand and Mullainathan ran a
perception survey. A language model is a different population and inherits none
of that validation.

Bertrand and Mullainathan published two per-name covariates that make the check
possible without any new measurement, and both are transcribed into
paper-a/data/reference/bm2004_names.json from the printed tables:

  perception_*      Table A1, p.1012. An independent human survey's probability
                    that the name is perceived as belonging to its assigned race.
                    This is the construct the list was validated against.
  callback_pct      Table 8, p.1008. The name's own callback rate in the field
                    experiment. This is the criterion -- what real employers did.
  mother_education  Table 8. Fraction of mothers with at least a high-school
                    education for babies given that name. Bertrand and
                    Mullainathan include it because socioeconomic status is the
                    obvious alternative explanation for a name effect, and they
                    test and reject it for their own data.

FOUR QUESTIONS, in increasing order of how much they threaten the design.

  Q1 CONVERGENT. Across names, does the model's per-name margin move with the
     human perception probability? A model that does not encode the association
     is not being asked the question the audit thinks it is asking, and a null
     on that model is uninformative rather than reassuring.

  Q2 CRITERION. Within race, does the model's per-name margin track the field
     experiment's per-name callback rate? Aggregate agreement with a rank
     correlation of zero would mean the model reproduces the headline number
     without reproducing the phenomenon -- the strongest reason to doubt that
     an LLM audit is measuring the same thing a correspondence audit measures.

  Q3 DISCRIMINANT. Does the margin track the socioeconomic proxy at least as
     well as it tracks race? Bertrand and Mullainathan could rule this out for
     their data. If we cannot for ours, then "race effect" is the wrong label.

  Q4 THE ANCHOR'S OWN INSTABILITY. Inside a single race-gender cell of the
     field experiment, the per-name callback rate ranges further than the
     headline between-race gap. That is the name-draw result of §4.2, present
     in the canonical study, and it is not an artefact of language models.

WHAT THIS CANNOT DO. Twelve names per race is a small n for a correlation, so
the intervals are wide and a null here is weak evidence. Every correlation is
therefore reported with a bootstrap interval over NAMES and with the width
stated, and the section that uses it says what it can and cannot support. The
sign and the interval are the result; a point estimate alone would be a misuse.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/analyze_construct_validity.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
NAMES = ROOT / "paper-a" / "data" / "names"
REF = ROOT / "paper-a" / "data" / "reference" / "bm2004_names.json"
OUT = NAMES / "construct_validity.json"

N_BOOT = 20_000
N_PERM = 20_000
SEED = 20260801

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B",
         "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}


# --------------------------------------------------------------------------
# covariates
# --------------------------------------------------------------------------
def load_covariates() -> dict:
    """Per-first-name covariates, keyed by name.

    `perception` is the probability the name is perceived as its OWN assigned
    race, so it is on one scale across the two races: a high value means
    an unambiguous signal in whichever direction the name points.
    """
    d = json.loads(REF.read_text(encoding="utf-8"))
    cov = {}
    for cell, entries in d["first_names"].items():
        race, gender = cell.split("_")
        for e in entries:
            perception = e.get("perception_white") if race == "white" \
                else e.get("perception_black")
            cov[e["name"]] = dict(
                race=race, gender=gender,
                callback_pct=e.get("callback_pct"),
                mother_education_pct=e.get("mother_education_pct"),
                perception=perception,
            )
    return cov, d


# --------------------------------------------------------------------------
# per-name margins from the study-4 grid
# --------------------------------------------------------------------------
def per_name_margins(model: str) -> dict:
    """Mean decision margin for each FIRST NAME, over surnames, templates and
    wordings.

    Each row of the grid carries two observations -- one per arm -- and they are
    read symmetrically, so a White first name and a Black first name are placed
    on the same scale by construction. Deduplicated on the full cell key first,
    because the recheck subset re-measures some cells.
    """
    f = NAMES / f"names_{model}.jsonl"
    if not f.exists():
        return {}
    best = {}
    for r in st.read_jsonl(f):
        if r.get("white_margin") is None or r.get("black_margin") is None:
            continue
        best[(r["variant"], r["template"], r["pair"])] = r
    acc = defaultdict(list)
    for r in best.values():
        acc[r["white_first"]].append(r["white_margin"])
        acc[r["black_first"]].append(r["black_margin"])
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


# --------------------------------------------------------------------------
# correlation with a bootstrap interval over NAMES
# --------------------------------------------------------------------------
def corr_ci(x, y, rng, n_boot=N_BOOT, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4:
        return None
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    est = float(fn(x, y).statistic)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(x[idx])) < 3 or len(set(y[idx])) < 3:
            boots[b] = np.nan
            continue
        boots[b] = fn(x[idx], y[idx]).statistic
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p against rho = 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(n=n, rho=est, ci=[float(lo), float(hi)],
                width=float(hi - lo), p=float(min(1.0, p)), method=method)


def _pearson_rows(a, b):
    """Row-wise Pearson correlation of two (n_draw, n) matrices."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def pooled_within_race_ci(x, y, race, rng, n_boot=N_BOOT, n_perm=N_PERM):
    """Correlate two variables ranked INSIDE each race, then pooled.

    WHY THIS IS NOT corr_ci ON THE POOLED RANKS, which is how it was first
    written. That version ranked once, froze the ranks, and handed them to a
    bootstrap that resamples rows as if they were exchangeable draws from one
    bivariate population. Two things go wrong at once, and both run the same
    way. The resample is not stratified, so a draw can hold eighteen White
    names and six Black ones -- a composition this statistic is not defined on,
    since it exists precisely to hold race fixed. And the ranks are not
    recomputed inside the draw, so the numbers being correlated stop describing
    the sample being correlated: a draw that repeats one name six times still
    carries that name's rank among the original twelve. Both understate the
    spread. On llama-3.1-8b-instruct the frozen-rank bootstrap gave SD 0.153
    against an exact null SD of 0.214, which narrowed the interval enough to
    exclude zero and drove the derived p from about 0.13 to 0.039 -- the only
    "excludes zero" result in the whole construct-validity check, and one the
    discussion had to be written around.

    This version resamples names WITHIN each race block and re-ranks inside
    every draw, and reports an exact permutation p beside the interval: under
    the null of no within-race association, permuting one variable inside each
    block leaves the statistic's distribution unchanged, so the tail is
    computed rather than assumed.

    ONE PROPERTY WORTH STATING, because the paper leans on it. When the two
    blocks are the same size and each carries the complete rank set, every
    block's mean rank equals the pooled mean rank, so the pooled Pearson is
    exactly the average of the two within-race correlations and the
    between-race contrast contributes nothing. That is what makes this the
    "part that is not the race effect". It holds here (twelve names per race)
    and would not hold for unbalanced blocks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    race = np.asarray(race)
    blocks = []
    for mask in (race == 1, race == 0):
        ok = mask & np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3:
            blocks.append((x[ok], y[ok]))
    if len(blocks) < 2:
        return None

    rx = np.concatenate([stats.rankdata(bx) for bx, _ in blocks])
    ry = np.concatenate([stats.rankdata(by) for _, by in blocks])
    est = float(_pearson_rows(rx[None, :], ry[None, :])[0])

    bxs, bys = [], []
    for bx, by in blocks:
        idx = rng.integers(0, len(bx), (n_boot, len(bx)))
        bxs.append(stats.rankdata(bx[idx], axis=1))
        bys.append(stats.rankdata(by[idx], axis=1))
    boots = _pearson_rows(np.concatenate(bxs, axis=1),
                          np.concatenate(bys, axis=1))
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())

    pxs, pys = [], []
    for bx, by in blocks:
        rky = stats.rankdata(by)
        perm = np.argsort(rng.random((n_perm, len(by))), axis=1)
        pxs.append(np.tile(stats.rankdata(bx), (n_perm, 1)))
        pys.append(rky[perm])
    null = _pearson_rows(np.concatenate(pxs, axis=1),
                         np.concatenate(pys, axis=1))
    null = null[np.isfinite(null)]
    p_perm = (1 + int((np.abs(null) >= abs(est) - 1e-12).sum())) / (len(null) + 1)

    return dict(n=int(len(rx)), n_per_race=[int(len(bx)) for bx, _ in blocks],
                rho=est, ci=[float(lo), float(hi)], width=float(hi - lo),
                p=float(min(1.0, p_boot)), p_perm=float(min(1.0, p_perm)),
                boot_sd=float(boots.std()), null_sd=float(null.std()),
                method="pearson_on_within_race_ranks")


def grid_replicate_noise(model: str, n_obs_per_name: int) -> dict | None:
    """The name grid's OWN noise floor, from the byte-identical S1/N1 twin.

    §4.2 used to borrow §5.2's zero floor to argue that the between-name spread
    it measures is real rather than sampling error. That floor belongs to a
    different serving configuration -- concurrency one with the prompt cache
    off -- and this grid ran at the default concurrency with cache reuse on, so
    it does not transfer. The grid has a floor of its own: S1 and N1 are the
    same prompt sent as two variants, so the gap between their margins on a
    cell is measurement noise and nothing else.

    A per-name margin averages n_obs observations, so the noise it inherits is
    the per-observation SD over sqrt(n_obs) -- which is the number §4.2 needs,
    not the per-observation one.
    """
    f = NAMES / f"names_{model}.jsonl"
    if not f.exists():
        return None
    cells = defaultdict(dict)
    for r in st.read_jsonl(f):
        if r["variant"] not in ("S1", "N1"):
            continue
        if r.get("white_margin") is None or r.get("black_margin") is None:
            continue
        cells[(r["template"], r["pair"])][r["variant"]] = r
    d = []
    for v in cells.values():
        if "S1" not in v or "N1" not in v:
            continue
        for arm in ("white_margin", "black_margin"):
            d.append(v["S1"][arm] - v["N1"][arm])
    if len(d) < 8:
        return None
    d = np.asarray(d, dtype=float)
    sigma_obs = float(d.std(ddof=1) / np.sqrt(2.0))
    return dict(
        n_replicate_pairs=int(len(d)),
        frac_identical=float(np.mean(d == 0.0)),
        sigma_per_observation=sigma_obs,
        sigma_on_name_mean=float(sigma_obs / np.sqrt(max(n_obs_per_name, 1))),
    )


def mde_for_pooled_within_race(n_per_race: int, rng, alpha=0.05, power=0.80,
                               n_sim=4000) -> float:
    """MDE for the POOLED within-race correlation the paper reports.

    That statistic ranks the two variables inside each race separately, pools
    the ranks, and takes a Pearson correlation on them: 2 * n_per_race
    observations, with the between-race contrast removed by construction. Its
    power is therefore between the n = 12 and n = 24 figures, and neither of
    those is the right thing to quote beside it.
    """
    lo, hi = 0.0, 0.99
    for _ in range(24):
        mid = (lo + hi) / 2
        L = np.linalg.cholesky(np.array([[1.0, mid], [mid, 1.0]]))
        hits = 0
        for _s in range(n_sim):
            rx, ry = [], []
            for _grp in range(2):
                xy = L @ rng.standard_normal((2, n_per_race))
                rx.extend(stats.rankdata(xy[0]))
                ry.extend(stats.rankdata(xy[1]))
            if stats.pearsonr(rx, ry).pvalue < alpha:
                hits += 1
        if hits / n_sim < power:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def mde_for_correlation(n: int, rng, alpha=0.05, power=0.80, n_sim=4000) -> float:
    """Smallest |rho| this n can detect at 80% power, by simulation.

    A null correlation on twelve names is not evidence of no relationship, and
    the only honest way to say how weak the evidence is, is to state the effect
    the design could have caught.
    """
    lo, hi = 0.0, 0.99
    for _ in range(24):
        mid = (lo + hi) / 2
        cov = np.array([[1.0, mid], [mid, 1.0]])
        L = np.linalg.cholesky(cov)
        hits = 0
        for _s in range(n_sim):
            z = rng.standard_normal((2, n))
            xy = L @ z
            r = stats.spearmanr(xy[0], xy[1])
            if r.pvalue < alpha:
                hits += 1
        if hits / n_sim < power:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def main() -> int:
    cov, raw = load_covariates()
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "n_boot": N_BOOT,
           "source": "paper-a/data/reference/bm2004_names.json",
           "models": {}}

    # ---------------------------------------------------------------- Q4 ----
    # the field experiment's own within-cell instability. no model involved.
    rng_cell = raw["within_race_callback_range_pp"]
    cells = {k: v for k, v in rng_cell.items() if not k.startswith("_")}
    head = raw["_headline"]["gap_pp"]
    out["field_anchor"] = dict(
        headline_gap_pp=float(head),
        within_cell_range_pp={k: float(v) for k, v in cells.items()},
        max_within_cell_pp=float(max(cells.values())),
        min_within_cell_pp=float(min(cells.values())),
        n_cells_exceeding_headline=int(sum(1 for v in cells.values() if v > head)),
        n_cells=len(cells),
    )
    print("=" * 100)
    print("Q4. THE FIELD ANCHOR'S OWN NAME-DRAW INSTABILITY (no model involved)")
    print("=" * 100)
    print(f"  headline between-race gap                 {head:.2f} pp")
    for k, v in sorted(cells.items()):
        flag = "  <-- exceeds the headline gap" if v > head else ""
        print(f"  within-cell per-name range, {k:<14}{v:.1f} pp{flag}")
    print(f"  cells whose within-race spread exceeds the between-race gap: "
          f"{out['field_anchor']['n_cells_exceeding_headline']} of {len(cells)}")

    # -------------------------------------------------------------- Q1-Q3 ---
    print("\n" + "=" * 100)
    print("Q1-Q3. DOES THE MODEL'S PER-NAME MARGIN TRACK THE CONSTRUCT?")
    print("=" * 100)

    grid_names = None
    for model in ORDER:
        pn = per_name_margins(model)
        if not pn:
            continue
        names = sorted(n for n in pn if n in cov)
        if grid_names is None:
            grid_names = names
        m = np.array([pn[n][0] for n in names])
        race = np.array([1.0 if cov[n]["race"] == "white" else 0.0 for n in names])
        perc = np.array([cov[n]["perception"] for n in names], dtype=float)
        call = np.array([cov[n]["callback_pct"] for n in names], dtype=float)
        ses = np.array([cov[n]["mother_education_pct"] for n in names], dtype=float)

        # Q1 convergent. Signed perception: +p for White names, -p for Black,
        # so a model that encodes the association should show a POSITIVE slope
        # regardless of which direction its overall race effect runs -- we test
        # against the model's own race direction below rather than assume one.
        signed = np.where(race == 1, perc, -perc)
        q1 = dict(corr_ci(signed, m, rng) or {})
        # Q1 WITHIN RACE, which is the statistic the manipulation check needs
        # and which was missing: the all-names version above is dominated by
        # the between-race contrast it was meant to validate independently.
        #
        # Perception has to be signed BEFORE ranking. It is the probability a
        # name is read as its own assigned race, so within White names a high
        # value means a distinctly White name and within Black names a high
        # value means a distinctly BLACK one -- the two blocks predict opposite
        # movements of the margin. Ranking the raw probability inside each
        # block and pooling would average a positive against a negative and
        # return nothing however well the model encodes the association.
        # Signing first puts both blocks on "how far toward White does this
        # name point", the direction the audit's own estimand runs. Inside a
        # block the multiplier is constant, so this is the raw ranking on White
        # names and its reverse on Black ones, and the race separation that
        # ruins the all-names version cannot survive ranking within a block.
        q1["pooled_within_race"] = pooled_within_race_ci(signed, m, race, rng)

        # the race effect on this same per-name scale, for reference
        beta = float(m[race == 1].mean() - m[race == 0].mean())

        # Q2 criterion, WITHIN race so the between-race difference cannot drive it
        q2 = {}
        for rname, mask in (("white", race == 1), ("black", race == 0)):
            q2[rname] = corr_ci(call[mask], m[mask], rng)
        q2["pooled_within_race"] = pooled_within_race_ci(call, m, race, rng)

        # Q3 discriminant, within race as well
        q3 = {}
        for rname, mask in (("white", race == 1), ("black", race == 0)):
            q3[rname] = corr_ci(ses[mask], m[mask], rng)
        q3["pooled_within_race"] = pooled_within_race_ci(ses, m, race, rng)

        # THE GRID'S OWN NOISE FLOOR. §4.2 argues the between-name spread is
        # real rather than sampling error, and used to borrow §5.2's zero floor
        # to do it -- a floor measured at concurrency one with the cache off,
        # which is not how this grid ran. The S1/N1 twin inside the grid gives
        # it a floor measured under its own serving conditions.
        _nobs = int(np.median([pn[n][1] for n in names]))
        _noise = grid_replicate_noise(model, _nobs)
        if _noise:
            _sd_wr = float(np.sqrt(np.mean([m[race == 1].var(ddof=1),
                                            m[race == 0].var(ddof=1)])))
            _noise["sd_name_margin_within_race"] = _sd_wr
            _noise["noise_over_within_race_sd"] = float(
                _noise["sigma_on_name_mean"] / _sd_wr) if _sd_wr > 0 else None

        out["models"][model] = dict(
            n_names=len(names), n_obs_per_name=_nobs,
            race_effect_on_name_scale=beta,
            replicate_noise=_noise,
            q1_perception=q1, q2_callback=q2, q3_ses=q3,
            per_name={n: dict(margin=pn[n][0], n=pn[n][1], **{
                k: cov[n][k] for k in ("race", "gender", "callback_pct",
                                       "mother_education_pct", "perception")})
                for n in names},
        )
        print(f"\n{SHORT.get(model, model)}   ({len(names)} names, "
              f"{_nobs} obs per name)")
        print(f"  race effect on the per-name margin scale        {beta:+.4f}")
        if _noise:
            print(f"  grid replicate floor (S1/N1)  sd/obs={_noise['sigma_per_observation']:.4f}"
                  f"  sd/name mean={_noise['sigma_on_name_mean']:.4f}"
                  f"  = {_noise['noise_over_within_race_sd']:.1%} of the"
                  f" within-race spread")
        if q1.get("rho") is not None:
            print(f"  Q1 margin vs signed perception, all 24 rho={q1['rho']:+.3f}  "
                  f"95% CI [{q1['ci'][0]:+.3f}, {q1['ci'][1]:+.3f}]  p={q1['p']:.3f}")
        for label, w in (("Q1 margin vs perception,  within race", q1.get("pooled_within_race")),
                         ("Q2 margin vs BM callback, within race", q2["pooled_within_race"]),
                         ("Q3 margin vs SES proxy,   within race", q3["pooled_within_race"])):
            if w:
                print(f"  {label} r  ={w['rho']:+.3f}  "
                      f"95% CI [{w['ci'][0]:+.3f}, {w['ci'][1]:+.3f}]  "
                      f"p_perm={w['p_perm']:.3f}  (boot sd {w['boot_sd']:.3f} vs "
                      f"null sd {w['null_sd']:.3f})")

    # what could this n have detected at all?
    if grid_names:
        n_within = len(grid_names) // 2
        out["mde"] = dict(
            n_names_total=len(grid_names), n_names_per_race=n_within,
            mde_rho_all_names=mde_for_correlation(len(grid_names),
                                                  np.random.default_rng(SEED + 1)),
            mde_rho_within_race=mde_for_correlation(n_within,
                                                    np.random.default_rng(SEED + 2)),
            # THE MDE OF THE STATISTIC WE ACTUALLY REPORT. The paper quotes the
            # POOLED within-race correlation -- ranks taken inside each race,
            # then pooled -- which has all 24 names in it, not 12. Quoting the
            # n = 12 figure beside it overstated how little the design could
            # see, and every use of it ran in the direction of excusing a null.
            # Simulated on the same construction as the reported statistic.
            mde_rho_pooled_within_race=mde_for_pooled_within_race(
                n_within, np.random.default_rng(SEED + 3)),
        )
        print(f"\n  MINIMUM DETECTABLE CORRELATION at 80% power")
        print(f"    over all {out['mde']['n_names_total']} names       "
              f"|rho| >= {out['mde']['mde_rho_all_names']:.2f}")
        print(f"    within one race ({n_within} names)  "
              f"|rho| >= {out['mde']['mde_rho_within_race']:.2f}")
        print("    A null below these is uninformative and is reported as such.")

    # a cross-model summary the paper can quote without re-deriving
    q1s = [v["q1_perception"] for v in out["models"].values()
           if v.get("q1_perception", {}).get("rho") is not None]
    def _within(key):
        return [v[key]["pooled_within_race"] for v in out["models"].values()
                if v.get(key, {}).get("pooled_within_race")]
    def summarise(xs):
        if not xs:
            return None
        r = [x["rho"] for x in xs]
        d = dict(n_models=len(xs), min=float(min(r)), max=float(max(r)),
                 n_excluding_zero=int(sum(1 for x in xs
                                          if x["ci"][0] * x["ci"][1] > 0)),
                 max_abs=float(max(abs(v) for v in r)))
        # the exact test, where there is one. n_excluding_zero is a property of
        # the percentile interval; n_p_perm_below_05 is the permutation tail,
        # and the two are reported side by side rather than one standing in for
        # the other, because an earlier version of this file had a bootstrap
        # that disagreed with its own null.
        if all("p_perm" in x for x in xs):
            d["n_p_perm_below_05"] = int(sum(1 for x in xs if x["p_perm"] < 0.05))
            d["min_p_perm"] = float(min(x["p_perm"] for x in xs))
        return d
    out["summary"] = dict(q1_perception=summarise(q1s),
                          q1_perception_within_race=summarise(_within("q1_perception")),
                          q2_callback_within_race=summarise(_within("q2_callback")),
                          q3_ses_within_race=summarise(_within("q3_ses")))

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
