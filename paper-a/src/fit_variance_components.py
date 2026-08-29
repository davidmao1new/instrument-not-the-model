"""Hierarchical decomposition of the paired margin difference.

For each pair we observe

    d = m(White name) − m(Black name)

in log-odds. The model is a crossed random-effects decomposition

    d_i = β + u_variant[v(i)] + u_template[t(i)] + u_pair[p(i)] + ε_i

with u ~ Normal(0, σ_·). β is the demographic effect. **σ_variant is the paper's
headline parameter**: the standard deviation of the measured demographic effect
across wordings that ask the same question. It is the amount by which a
single-prompt estimate can move for reasons that have nothing to do with race.

Fitting this rather than taking the raw between-variant standard deviation
matters. The raw SD is inflated by sampling noise, because each per-variant
estimate is itself measured with error. Partial pooling separates the two, so
σ_variant estimates the true between-wording dispersion rather than
dispersion-plus-noise.

The model is fitted separately for the SEMANTIC and NULL arms, because the
comparison between them is the paper's control: dispersion under paraphrase is
expected, dispersion under a trailing newline is not.

SCALE. Everything is estimated in log-odds, which is where the model lives. The
`*_pp_max` fields multiply by the logistic's steepest slope, dp = dlogit × p(1−p)
evaluated at p = 0.5, and are an UPPER BOUND on the probability-scale movement
rather than an estimate of it — the four models sit at operating points where a
fixed Jacobian overstates the true probability-scale effect by 1.8× to 108.5×
(see `effectsize.py`). A variance component has no single p attached to it, so a
bound is the only pp statement available for one.

They are therefore NOT how this paper compares itself to published
percentage-point gaps. That comparison runs through the shortlist-rate gap at a
fixed cut, which is the one quantity in this design commensurable with a callback
rate. See CHANGELOG Amendment 8. The scale-free quantity the argument actually
rests on is `ratio_sigma_v_to_beta`, which needs no conversion at all.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import numpyro
numpyro.set_host_device_count(2)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data" / "delta_stability"
OUT = DATA / "variance_components.json"

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
}
# An UPPER BOUND, not a conversion. 25.0 is 100 x dp/dlogit at p = 0.5, where
# p(1-p) is maximised, so 25 x |log-odds| is the largest percentage-point
# movement the quantity could correspond to. Used as a point estimate it is
# wrong by up to 108x on this panel; see effectsize.py's opening note. A
# variance component has no single p attached to it, so a bound is the only
# honest pp statement available for one, and the fields below say so in their
# names. The scale-free quantity the paper actually argues from is
# ratio_sigma_v_to_beta, which needs no conversion at all.
PP_PER_LOGIT_MAX = 25.0


def load():
    rows = []
    for f in sorted(DATA.glob("delta_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if (r.get("white_margin") is not None
                        and r.get("black_margin") is not None):
                    rows.append(r)
    return rows


def fit(rows, label, arm=None, warmup=2000, samples=2000):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import numpyro.diagnostics as diag

    sub = rows if arm is None else [r for r in rows if r["variant_kind"] == arm]
    if len(sub) < 30:
        return None
    variants = sorted({r["variant"] for r in sub})
    templates = sorted({r["template"] for r in sub})
    pairs = sorted({r["pair"] for r in sub})
    vi = {v: i for i, v in enumerate(variants)}
    ti = {t: i for i, t in enumerate(templates)}
    pi = {p: i for i, p in enumerate(pairs)}

    d = jnp.array([r["white_margin"] - r["black_margin"] for r in sub])
    v_idx = jnp.array([vi[r["variant"]] for r in sub])
    t_idx = jnp.array([ti[r["template"]] for r in sub])
    p_idx = jnp.array([pi[r["pair"]] for r in sub])
    scale = float(np.std([float(x) for x in d])) or 1.0

    def model(d=None):
        beta = numpyro.sample("beta", dist.Normal(0, 2 * scale))
        s_v = numpyro.sample("sigma_variant", dist.HalfNormal(scale))
        s_t = numpyro.sample("sigma_template", dist.HalfNormal(scale))
        s_p = numpyro.sample("sigma_pair", dist.HalfNormal(scale))
        s_e = numpyro.sample("sigma_resid", dist.HalfNormal(2 * scale))
        # NON-CENTRED. The centred form -- u ~ Normal(0, sigma) -- couples each
        # group effect to its own scale, and when a component is estimated from
        # few groups the posterior takes on a funnel that NUTS cannot traverse
        # at any step size. sigma_template is the worst case here: THREE levels.
        # With the centred form and target_accept 0.95 it still failed, at
        # r-hat 1.012 for llama-3.1-8b-instruct on the pooled arm, which the
        # per-parameter diagnostics caught. Sampling a standard normal and
        # scaling it decouples them and the geometry becomes benign.
        with numpyro.plate("V", len(variants)):
            z_v = numpyro.sample("z_variant", dist.Normal(0, 1))
        with numpyro.plate("T", len(templates)):
            z_t = numpyro.sample("z_template", dist.Normal(0, 1))
        with numpyro.plate("P", len(pairs)):
            z_p = numpyro.sample("z_pair", dist.Normal(0, 1))
        mu = beta + s_v * z_v[v_idx] + s_t * z_t[t_idx] + s_p * z_p[p_idx]
        numpyro.sample("obs", dist.Normal(mu, s_e), obs=d)

    # target_accept 0.95 rather than the 0.8 default: the variance components
    # here are weakly identified with only 6 variants per arm, which produces a
    # funnel geometry and divergent transitions at looser adaptation. One fit
    # returned r-hat 1.12 at the defaults; this and the longer chains bring
    # every fit under 1.01.
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95),
                num_warmup=warmup, num_samples=samples,
                num_chains=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), d=d)
    s = mcmc.get_samples()
    g = mcmc.get_samples(group_by_chain=True)
    # Diagnostics PER PARAMETER, split into what the paper interprets and what
    # it does not.
    #
    # WHY, AND WHAT IT COST TO LEARN. The Study 4 fit reported a single worst
    # case over every parameter -- r-hat 1.0678, min ESS 15 -- and that line
    # could not say which quantity had failed. Diagnosed per parameter, the
    # failure turned out to be confined to level parameters the paper never
    # reports, while beta and both variance components were converged
    # throughout. A worst-case scalar cannot distinguish "this result is
    # unquotable" from "a nuisance parameter mixed slowly", and those call for
    # opposite responses. See CHANGELOG Amendment 9.
    INTERPRETED = ("beta", "sigma_variant", "sigma_template", "sigma_pair",
                   "sigma_resid")
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
        print(f"  [WARNING] {label}/{arm or 'all'}: interpreted parameters not "
              f"converged: {', '.join(bad)}", flush=True)

    def q(a):
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 50)),
                float(np.percentile(a, 97.5))]

    return dict(model=label, arm=arm or "all", n=len(sub),
                n_variants=len(variants),
                beta=q(s["beta"]), beta_pp_max=q(np.abs(np.asarray(s["beta"])) * PP_PER_LOGIT_MAX),
                sigma_variant=q(s["sigma_variant"]),
                sigma_variant_pp_max=q(np.asarray(s["sigma_variant"]) * PP_PER_LOGIT_MAX),
                sigma_template=q(s["sigma_template"]),
                sigma_pair=q(s["sigma_pair"]),
                sigma_resid=q(s["sigma_resid"]),
                ratio_sigma_v_to_beta=q(np.abs(np.asarray(s["sigma_variant"])
                                               / np.asarray(s["beta"]))),
                max_rhat=rhat, min_ess=ess, converged=converged,
                diagnostics_interpreted=interp,
                diagnostics_latent=latent)


def main() -> int:
    rows = load()
    if not rows:
        sys.exit("no complete-margin rows")
    out = {}
    models = sorted({r["model"] for r in rows})
    print("Crossed random-effects decomposition of the paired margin difference.")
    print("Log-odds; pp conversion is an upper bound taken at p = 0.5.\n")
    print(f"{'model':<26}{'arm':<10}{'beta (pp)':>18}{'sd_variant (pp)':>20}"
          f"{'sd/|beta|':>16}{'rhat':>7}")
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        out[m] = {}
        for arm in ("semantic", "null", None):
            key = arm or "all"
            res = fit(mr, m, arm)
            if res is None:
                continue
            out[m][key] = res
            b, sv, rt = res["beta_pp_max"], res["sigma_variant_pp_max"], res["ratio_sigma_v_to_beta"]
            print(f"{SHORT.get(m, m)[:25]:<26}{key:<10}"
                  f"{f'{b[1]:+.2f} [{b[0]:+.2f},{b[2]:+.2f}]':>18}"
                  f"{f'{sv[1]:.2f} [{sv[0]:.2f},{sv[2]:.2f}]':>20}"
                  f"{f'{rt[1]:.2f}':>16}{res['max_rhat']:>7.3f}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
