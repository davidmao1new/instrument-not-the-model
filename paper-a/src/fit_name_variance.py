"""Study 4 analysis: how much of a measured "race effect" is the race, and how
much is which names were drawn?

THE MODEL. Fitted on the PER-NAME margin rather than the paired difference, so
that every variance component lands on one scale where it can be compared to the
race effect directly:

    m = mu + beta*race + gamma*gender + tau*template
        + u_first[first name] + u_last[surname]
        + u_variant[wording] + u_cell[wording,template,pair] + eps

  beta        the demographic effect. What the literature reports.
  u_first     deviation of a given first name from its race's mean, so
              sigma_first is the WITHIN-RACE between-first-name SD. This is the
              quantity that has never been estimated in an LLM audit.
  u_last      the same for surnames, estimable only because the design crosses
              first names with surnames instead of fixing 48 full names.
  u_cell      absorbs everything shared by the two members of a matched pair --
              the resume, the wording, the position in the grid. Without it the
              two rows of a pair would be treated as independent, and they are
              not: they differ by one token.

Race is FIXED and first name is RANDOM NESTED WITHIN RACE. That is what makes
sigma_first interpretable: the race fixed effect takes the between-race mean
difference, and u_first takes what is left, which is name-specific and not race.

--------------------------------------------------------------------------
THE COUNTERFACTUAL NAME DRAW.
--------------------------------------------------------------------------
A variance component is the honest summary and it is not the vivid one. Every
paper in this literature picks a name list; Bertrand and Mullainathan used nine
first names per race per gender. So the question a reader actually has is:

    if the authors had drawn a DIFFERENT but equally defensible set of names,
    what effect would they have published?

That is directly simulable from the factorial. Resample name sets of the size
the literature uses, re-estimate the effect on each, and report the spread of
published-effect-that-could-have-been. It is a bootstrap over names rather than
over observations, and it answers the question in the units the field uses.

If that spread is wide, then the disagreements between published audits may be
disagreements about name lists, which is the same shape of argument the paper
already makes about wordings, on a dimension nobody has examined.

The question splits in two and the split is not cosmetic, so both halves are
estimated and both are written to the artifact:

    "had WE used k of the 12 name pairs we measured" -- draw without
    replacement from the grid, which is the finite population for that
    question. Key `counterfactual`.

    "what would a NEW study drawing k names face" -- draw with replacement,
    because that study is not confined to our twelve. Key
    `counterfactual_with_replacement`.

The second is larger, by sqrt((12 - k) / 11) at every k -- a factor of two at
k = 9 -- and it is the one that matches the sigma / sqrt(k) law the paper
states. See counterfactual_draws() for why the difference decides the
averaging-asymmetry crossover rather than merely decorating it.

    .venv/Scripts/python.exe paper-a/src/fit_name_variance.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

import numpyro

numpyro.set_host_device_count(2)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data" / "names"
OUT = DATA / "name_variance.json"
RNG = np.random.default_rng(20260728)

# The with-replacement arm draws from its OWN stream. If it shared RNG it would
# advance the same generator and shift every subsequent draw of the
# without-replacement arm, silently moving numbers that are already in the
# paper. Independent streams make the new arm a pure addition: the
# without-replacement block of the artifact is bit-identical to before.
RNG_WITH_REPLACEMENT = np.random.default_rng(20260729)

# The literature's own draw sizes, so the simulation answers the question in the
# terms the field uses. BM2004 used 9 first names per race per gender.
DRAW_SIZES = (3, 6, 9)


def load() -> list[dict]:
    rows = []
    for f in sorted(DATA.glob("names_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if (r.get("white_margin") is not None
                        and r.get("black_margin") is not None):
                    rows.append(r)
    best = {}
    for r in rows:
        best[(r["model"], r["variant"], r["template"], r["pair"])] = r
    return list(best.values())


def to_long(rows):
    """One row per NAME rather than per pair.

    NO LONGER USED BY `fit`, which now works directly on pairs so that it can
    marginalise the cell effect analytically -- see fit()'s docstring. Kept
    because it is the natural shape for any per-name summary, and because
    deleting it would make the sampled parameterisation harder to reconstruct if
    the marginalisation ever needs checking against it.
    """
    out = []
    for r in rows:
        for race in ("white", "black"):
            out.append(dict(
                model=r["model"], race=race, margin=r[f"{race}_margin"],
                first=f"{race}:{r[f'{race}_first']}",
                last=f"{race}:{r[f'{race}_last']}",
                gender=r["gender"], variant=r["variant"], template=r["template"],
                cell=f"{r['variant']}|{r['template']}|{r['pair']}"))
    return out


def averaging_asymmetry(cf, sigma_variant_paired):
    """The asymmetry that decides which researcher choice actually costs more.

    Name variance and wording variance are both real and of the same order per
    unit. They do not enter a published estimate the same way, and that is the
    point:

      NAMES AVERAGE DOWN. A study uses nine names per race, so the name-draw
      component of its published effect is roughly sigma_first / sqrt(9).

      WORDINGS DO NOT. A study uses ONE wording, so it carries the full
      sigma_variant. Nothing in the standard design averages it away, because
      nobody varies it.

    So the field's protection against the name lottery is that it buys nine
    tickets. It has no equivalent protection against the wording lottery,
    because it buys one.

    WHICH COUNTERFACTUAL ARM BELONGS HERE. The sentence above says "roughly
    sigma_first / sqrt(9)", and that law describes a study drawing nine names
    from the POPULATION. Feeding this function the without-replacement arm
    computes the ratio from a draw of nine out of our twelve instead, which
    carries a finite-population factor of 0.52 at k = 9 and so halves the
    numerator of the ratio. That does not refute the asymmetry -- wordings
    still do not average at all -- but it moves the crossover point, so the
    caller computes this table on BOTH arms and the paper states which it is
    quoting. `draw_mode` is carried through from the counterfactual so the row
    cannot be read without it.
    """
    out = {}
    for k, c in cf.items():
        n = int(k.split("_")[1])
        out[k] = dict(
            draw_mode=c.get("draw_mode"),
            n_names_per_race=n,
            sd_from_name_draw=c["beta_sd"],
            sd_from_wording_choice=sigma_variant_paired,
            ratio_name_to_wording=(c["beta_sd"] / sigma_variant_paired
                                   if sigma_variant_paired else None))
    return out


def fit(rows, label, warmup=2500, samples=2500):
    """Fit the crossed random-effects model with u_cell MARGINALISED OUT.

    WHY THE CELL EFFECT IS NOT SAMPLED. `cell` is (wording, template, pair) and
    the design puts EXACTLY TWO observations in each -- the White-named row and
    the Black-named row. A random intercept with two observations per level is
    very nearly saturated: 1,728 latent variables competing with the global
    intercept, the gender coefficient and the template coefficient to explain
    the same pair-level variation. Sampled directly it funnels badly. Measured
    on llama-2-7b-chat at 800 draws x 2 chains:

        z_cell         r-hat 1.3023   min ESS     3
        gamma_gender   r-hat 1.0978   min ESS    12
        sigma_cell     r-hat 1.0793   min ESS    24
        tau_template   r-hat 1.0115   min ESS    26
        mu             r-hat 1.0069   min ESS    44

    Raising warmup does not repair this; it is a geometry problem, not a budget
    problem. The parameters this study actually interprets -- beta, sigma_first,
    sigma_variant -- were converged even then, at ESS 175, 103 and 125, so the
    earlier point estimates were not wrong. They were unaccompanied by
    trustworthy intervals on part of the model, which is not a difference a
    reader can see from the output.

    THE FIX IS EXACT, NOT AN APPROXIMATION. With two observations sharing one
    cell effect, write each pair as its difference and its mean:

        d = y_white - y_black          u_cell cancels identically
        a = (y_white + y_black) / 2    u_cell enters with unit weight

    d and a are uncorrelated when the two residuals share a variance, so the
    joint likelihood factorises exactly into

        d ~ Normal( beta + sf*(zf[w]-zf[b]) + sl*(zl[w]-zl[b]),  sqrt(2)*se )
        a ~ Normal( mu + beta/2 + gamma*male + tau*strong + sv*zv[variant]
                    + sf*(zf[w]+zf[b])/2 + sl*(zl[w]+zl[b])/2,
                    sqrt(sc^2 + se^2/2) )

    Same model, same parameters, same interpretation, 1,728 fewer latents. Every
    posterior median agrees with the sampled form to three decimal places, and
    the worst diagnostic across ALL parameters becomes r-hat 1.003, min ESS 400.

    It also makes the design legible. beta and sigma_first are identified by the
    DIFFERENCE equation, in which the wording, the template and the resume have
    cancelled -- so the demographic effect and the between-name spread come from
    within-pair contrasts alone and cannot be contaminated by anything the two
    names share. sigma_variant is identified only by the MEAN equation, which is
    the correct statement that a wording shifting both names equally does not
    move the paired effect. That makes it a DIFFERENT quantity from Study 2's
    between-wording SD of the effect, and the two must not be conflated.
    """
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    import numpyro.diagnostics as diag
    from numpyro.infer import MCMC, NUTS

    # Levels are namespaced by race so a first name is never shared across
    # races, which is what makes sigma_first a WITHIN-race spread.
    def key(side, field, r):
        return side + ":" + str(r[side + "_" + field])

    first_lv = sorted({key(s_, "first", r)
                       for r in rows for s_ in ("white", "black")})
    last_lv = sorted({key(s_, "last", r)
                      for r in rows for s_ in ("white", "black")})
    var_lv = sorted({r["variant"] for r in rows})
    FI = {v: i for i, v in enumerate(first_lv)}
    LI = {v: i for i, v in enumerate(last_lv)}
    VI = {v: i for i, v in enumerate(var_lv)}

    d = jnp.array([r["white_margin"] - r["black_margin"] for r in rows])
    a = jnp.array([(r["white_margin"] + r["black_margin"]) / 2 for r in rows])
    fw = jnp.array([FI[key("white", "first", r)] for r in rows])
    fb = jnp.array([FI[key("black", "first", r)] for r in rows])
    lw = jnp.array([LI[key("white", "last", r)] for r in rows])
    lb = jnp.array([LI[key("black", "last", r)] for r in rows])
    vv = jnp.array([VI[r["variant"]] for r in rows])
    gg = jnp.array([1.0 if r["gender"] == "male" else 0.0 for r in rows])
    tt = jnp.array([1.0 if r["template"] == "T1_strong" else 0.0 for r in rows])
    nf, nl, nv = len(first_lv), len(last_lv), len(var_lv)
    scale = float(np.std([r["white_margin"] for r in rows]
                         + [r["black_margin"] for r in rows])) or 1.0

    def model(d=None, a=None):
        mu = numpyro.sample("mu", dist.Normal(0, 5 * scale))
        beta = numpyro.sample("beta", dist.Normal(0, 2 * scale))
        gam = numpyro.sample("gamma_gender", dist.Normal(0, 2 * scale))
        tau = numpyro.sample("tau_template", dist.Normal(0, 2 * scale))
        s_f = numpyro.sample("sigma_first", dist.HalfNormal(scale))
        s_l = numpyro.sample("sigma_last", dist.HalfNormal(scale))
        s_v = numpyro.sample("sigma_variant", dist.HalfNormal(scale))
        s_c = numpyro.sample("sigma_cell", dist.HalfNormal(scale))
        s_e = numpyro.sample("sigma_resid", dist.HalfNormal(2 * scale))
        with numpyro.plate("F", nf):
            zf = numpyro.sample("z_first", dist.Normal(0, 1))
        with numpyro.plate("L", nl):
            zl = numpyro.sample("z_last", dist.Normal(0, 1))
        with numpyro.plate("V", nv):
            zv = numpyro.sample("z_variant", dist.Normal(0, 1))
        numpyro.sample("obs_d",
                       dist.Normal(beta + s_f * (zf[fw] - zf[fb])
                                   + s_l * (zl[lw] - zl[lb]),
                                   jnp.sqrt(2.0) * s_e), obs=d)
        numpyro.sample("obs_a",
                       dist.Normal(mu + beta / 2 + gam * gg + tau * tt
                                   + s_v * zv[vv]
                                   + s_f * (zf[fw] + zf[fb]) / 2
                                   + s_l * (zl[lw] + zl[lb]) / 2,
                                   jnp.sqrt(s_c ** 2 + s_e ** 2 / 2)), obs=a)

    mcmc = MCMC(NUTS(model, target_accept_prob=0.95),
                num_warmup=warmup, num_samples=samples, num_chains=2,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), d=d, a=a)
    s = mcmc.get_samples()
    g = mcmc.get_samples(group_by_chain=True)

    # Diagnostics PER PARAMETER, split into what the paper interprets and what
    # it does not. A single worst-case number over every latent hides WHICH
    # quantity failed, and that is the number deciding whether a result can be
    # quoted at all.
    INTERPRETED = ("mu", "beta", "gamma_gender", "tau_template", "sigma_first",
                   "sigma_last", "sigma_variant", "sigma_cell", "sigma_resid")
    diags = {}
    for k, v in g.items():
        if v.ndim > 3 or v.shape[0] < 2:
            continue
        diags[k] = dict(rhat=float(np.nanmax(diag.gelman_rubin(v))),
                        ess=float(np.nanmin(diag.effective_sample_size(v))))
    interp = {k: v for k, v in diags.items() if k in INTERPRETED}
    latent = {k: v for k, v in diags.items() if k not in INTERPRETED}
    rhat = max(v["rhat"] for v in interp.values())
    ess = min(v["ess"] for v in interp.values())
    converged = rhat <= 1.01 and ess >= 100
    if not converged:
        bad = sorted(k for k in interp
                     if interp[k]["rhat"] > 1.01 or interp[k]["ess"] < 100)
        print("  [WARNING] " + label + ": interpreted parameters not converged: "
              + ", ".join(bad), flush=True)

    def q(arr):
        arr = np.asarray(arr)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 50)),
                float(np.percentile(arr, 97.5))]

    b = np.abs(np.asarray(s["beta"]))
    return dict(
        model=label, n_obs=2 * len(rows), n_pairs=len(rows),
        n_first=nf, n_last=nl, n_variant=nv,
        beta=q(s["beta"]),
        sigma_first=q(s["sigma_first"]), sigma_last=q(s["sigma_last"]),
        sigma_variant=q(s["sigma_variant"]), sigma_cell=q(s["sigma_cell"]),
        sigma_resid=q(s["sigma_resid"]),
        gamma_gender=q(s["gamma_gender"]), tau_template=q(s["tau_template"]),
        ratio_first_to_beta=q(np.asarray(s["sigma_first"]) / b),
        ratio_variant_to_beta=q(np.asarray(s["sigma_variant"]) / b),
        ratio_first_to_variant=q(np.asarray(s["sigma_first"])
                                 / np.asarray(s["sigma_variant"])),
        # THE CONTRAST THE CONTRIBUTIONS CLAIM ASSERTS, formed on the JOINT
        # draws. The paper said the first-name and surname components are not
        # the same size, from the two MARGINAL medians of whichever model
        # sorted first, with no interval and no count of how many checkpoints
        # support it. Two marginal summaries cannot settle a difference; the
        # posterior on the difference can.
        sigma_first_minus_last=q(np.asarray(s["sigma_first"])
                                 - np.asarray(s["sigma_last"])),
        max_rhat=rhat, min_ess=ess, converged=converged,
        diagnostics_interpreted=interp, diagnostics_latent=latent,
        parameterisation="u_cell marginalised (difference + mean)")


def counterfactual_draws(rows, n_boot=4000, *, replace=False, rng=None):
    """What effect would a different name list have produced?

    Resamples matched FIRST-NAME PAIRS, at the list sizes the literature uses,
    and re-estimates the effect on the cells those pairs generate. Wordings and
    templates are held at their full set, so the only thing varying is the name
    draw. `replace` selects which of the two name-draw questions is being asked;
    see the second block below, which is the whole reason the argument the paper
    makes about averaging needs care.

    -----------------------------------------------------------------------
    THE UNIT IS A PAIR, AND GETTING THAT WRONG COST EVERY NUMBER BELOW.
    -----------------------------------------------------------------------
    The first version of this function drew k White first names and k Black
    first names INDEPENDENTLY and then pooled every cell in the k x k product.
    That looks like a name list of size k and is not one. The design pairs each
    White first name with exactly ONE Black first name -- there are 12 distinct
    (white_first, black_first) keys, not 144 -- so only the draws that happened
    to select a matched pair on both sides contributed anything. Measured on the
    real grid at 4,000 replicates:

        k = 3   mean 0.76 matched pairs actually contributing, and 1,515 of
                4,000 draws selected NO overlapping pair at all. Those were
                silently skipped by `if not vals: continue`, so the recorded
                n_sim was 2,485 and every k = 3 statistic was conditioned on
                the 62% of draws that happened to overlap.
        k = 6   mean 3.01 contributing pairs
        k = 9   mean 6.74 contributing pairs

    So each row was describing a list of effective size about k^2/12, with the
    dispersion inflated accordingly, under a label saying k. A paired
    correspondence audit that "uses k names per race" uses k MATCHED PAIRS, so
    that is what is drawn here.

    -----------------------------------------------------------------------
    WHY BOTH DRAW MODES ARE COMPUTED, AND WHICH QUESTION EACH ONE ANSWERS.
    -----------------------------------------------------------------------
    Drawing k of the N = 12 pairs WITHOUT replacement samples a FINITE
    population, so the SD of the resulting mean carries the finite-population
    correction

        sd_without / sd_with = sqrt((N - k) / (N - 1))

    which is 0.90 at k = 3, 0.74 at k = 6 and 0.52 at k = 9. At k = 9 the
    dispersion is therefore about HALF what it would otherwise be, and at
    k = 12 it would be exactly zero, because there is only one way to draw all
    twelve and the "counterfactual" study is then this study.

    That shrinkage is CORRECT for the question "what would we have published had
    we reported k of the 12 name pairs we actually measured?" The draw is
    conditional on this grid, and for that question the grid IS the population.

    It is WRONG for the question the averaging-asymmetry argument asks, which is
    what a NEW study drawing k names would face. That study is not restricted to
    our twelve pairs, so it gets no finite-population discount: its exposure
    follows sigma / sqrt(k) with no k/N term, which is exactly the law the paper
    states. Quoting the conditional number against that law UNDERSTATES a new
    study's name-draw exposure, and understates it worst at the k = 9 the field
    actually uses -- by a factor of two, precisely where the crossover claim
    against the wording component is decided.

    So both arms are computed and both are written to the artifact. Neither
    supersedes the other; they are different estimands and the paper must say
    which one a given table reports. `replace=True` is a CLUSTER bootstrap --
    whole pairs are resampled, never individual cells -- and it is a plug-in
    approximation, not a draw from the real name population: the between-pair
    spread it resamples is itself estimated from only twelve pairs, so it is the
    right target estimated noisily rather than the wrong target estimated
    precisely. The resampling unit is the matched pair in both arms.
    """
    rng = RNG if rng is None else rng
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r["white_first"], r["black_first"])].append(
            r["white_margin"] - r["black_margin"])
    pairs = sorted(by_pair)

    out = {}
    for k in DRAW_SIZES:
        if k > len(pairs) and not replace:
            continue
        ests, sups = [], []
        for _ in range(n_boot):
            idx = rng.choice(len(pairs), size=k, replace=replace)
            vals = []
            for i in idx:
                vals.extend(by_pair[pairs[i]])
            v = np.array(vals)
            ests.append(float(v.mean()))
            sups.append(float((v > 0).mean()))
        if not ests:
            continue
        e = np.array(ests)
        su = np.array(sups)
        # The mode is stamped on every row so a table cannot be built from
        # these numbers without the reader knowing which estimand they are.
        # fpc is the factor by which the without-replacement SD is shrunk
        # relative to a draw from an unrestricted population; it is 1 by
        # definition for the with-replacement arm.
        n_avail = len(pairs)
        out[f"draw_{k}_per_race"] = dict(
            draw_mode="with_replacement" if replace else "without_replacement",
            finite_population_factor=(
                1.0 if replace
                else float(np.sqrt((n_avail - k) / (n_avail - 1)))),
            n_sim=len(e), n_pairs_drawn=k, n_pairs_available=len(pairs),
            beta_median=float(np.median(e)),
            beta_p2_5=float(np.percentile(e, 2.5)),
            beta_p97_5=float(np.percentile(e, 97.5)),
            beta_min=float(e.min()), beta_max=float(e.max()),
            beta_sd=float(e.std(ddof=1)),
            sign_flip_frac=float(((e > 0).mean()) if e.mean() < 0 else ((e < 0).mean())),
            ps_median=float(np.median(su)),
            ps_p2_5=float(np.percentile(su, 2.5)),
            ps_p97_5=float(np.percentile(su, 97.5)))
    return out


def per_name_effects(rows):
    """Each first name's mean margin, so the extremes can be named in the paper."""
    acc = defaultdict(list)
    for r in rows:
        acc[("white", r["white_first"])].append(r["white_margin"])
        acc[("black", r["black_first"])].append(r["black_margin"])
    return {f"{race}:{n}": dict(mean=float(np.mean(v)), n=len(v))
            for (race, n), v in sorted(acc.items())}


def main() -> int:
    rows = load()
    if not rows:
        sys.exit(f"no data in {DATA} yet")
    out = {}
    for m in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == m]
        print(f"\n{'='*78}\n{m}   ({len(mr)} pairs)\n{'='*78}", flush=True)
        res = fit(mr, m)
        # Two arms, two estimands -- see counterfactual_draws(). `counterfactual`
        # keeps its name and its exact numbers because the built paper already
        # quotes it; the population-referent arm is added beside it rather than
        # replacing it, and the without-replacement call stays FIRST so its RNG
        # stream is untouched.
        res["counterfactual"] = counterfactual_draws(mr)
        res["counterfactual_with_replacement"] = counterfactual_draws(
            mr, replace=True, rng=RNG_WITH_REPLACEMENT)
        res["per_name"] = per_name_effects(mr)
        # sigma_variant from the PAIRED model in Study 2, which is the wording
        # dispersion of the DEMOGRAPHIC EFFECT. The sigma_variant fitted above is
        # on the per-name margin and therefore measures how far a wording moves
        # the model's overall acceptance level, which is a much larger and quite
        # different quantity. Comparing the two would be a category error.
        vc = ROOT / "paper-a" / "data" / "delta_stability" / "variance_components.json"
        sv = None
        if vc.exists():
            sv = json.loads(vc.read_text(encoding="utf-8")).get(m, {}) \
                .get("all", {}).get("sigma_variant", [None, None, None])[1]
        res["sigma_variant_paired_study2"] = sv
        res["averaging_asymmetry"] = averaging_asymmetry(res["counterfactual"], sv)
        res["averaging_asymmetry_with_replacement"] = averaging_asymmetry(
            res["counterfactual_with_replacement"], sv)
        out[m] = res
        print(f"  beta (race)          {res['beta'][1]:+.4f} "
              f"[{res['beta'][0]:+.4f}, {res['beta'][2]:+.4f}]")
        for k in ("sigma_first", "sigma_last", "sigma_variant", "sigma_cell",
                  "sigma_resid"):
            v = res[k]
            print(f"  {k:<20} {v[1]:.4f} [{v[0]:.4f}, {v[2]:.4f}]")
        r1 = res["ratio_first_to_beta"]
        r2 = res["ratio_first_to_variant"]
        print(f"  sigma_first / |beta|  {r1[1]:.2f} [{r1[0]:.2f}, {r1[2]:.2f}]")
        print(f"  sigma_first / sigma_variant  {r2[1]:.2f} [{r2[0]:.2f}, {r2[2]:.2f}]")
        print(f"  r-hat {res['max_rhat']:.4f}   min ESS {res['min_ess']:.0f}")
        for arm, lab in (("counterfactual", "of the 12 measured"),
                         ("counterfactual_with_replacement", "from the population")):
            print(f"  -- draw {lab} --")
            for k, c in res[arm].items():
                print(f"  {k}: effect would have been {c['beta_median']:+.4f} "
                      f"[{c['beta_p2_5']:+.4f}, {c['beta_p97_5']:+.4f}], "
                      f"full range [{c['beta_min']:+.4f}, {c['beta_max']:+.4f}], "
                      f"wrong-sign in {c['sign_flip_frac']:.1%} of draws")
        if sv:
            print(f"\n  Which choice costs more? Name variance averages down with "
                  f"list size; wording variance does not, because studies use one.")
            print(f"    {'names/race':<12}{'SD from name draw':>19}"
                  f"{'SD from wording':>17}{'ratio':>8}{'  draw mode':<24}")
            for arm in ("averaging_asymmetry",
                        "averaging_asymmetry_with_replacement"):
                for k, a in res[arm].items():
                    print(f"    {a['n_names_per_race']:<12}"
                          f"{a['sd_from_name_draw']:>19.4f}"
                          f"{a['sd_from_wording_choice']:>17.4f}"
                          f"{a['ratio_name_to_wording']:>7.2f}x"
                          f"  {a['draw_mode']}")

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
