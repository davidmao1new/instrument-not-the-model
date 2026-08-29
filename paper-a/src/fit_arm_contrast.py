"""Do semantically null edits move the effect MORE than paraphrases do?

WHY THIS FILE EXISTS. The paper makes a comparison it did not test. Study 2
fits the variance decomposition separately for the semantic arm and the null
arm, then compares the two sigma_variant point estimates in prose: on
Llama-3.1-8B the null arm's 0.050 against the semantic arm's 0.022, reported as
the nulls moving the estimate more than the paraphrases do.

That is two separate fits compared by eye. Their credible intervals overlap
completely on **all four models**:

    llama-2-7b-chat            semantic [0.0441, 0.2365]  null [0.0056, 0.1285]
    llama-3.1-8b-instruct      semantic [0.0023, 0.0859]  null [0.0152, 0.1443]
    mistral-7b-instruct-v0.1   semantic [0.0074, 0.0706]  null [0.0018, 0.0530]
    mistral-7b-instruct-v0.3   semantic [0.0067, 0.2120]  null [0.0039, 0.1637]

CLAUDE.md names this failure explicitly as the single most common error in this
literature: comparing two separately-estimated quantities and treating a
difference in their point estimates, or in their significance, as evidence that
they differ. The project's own rules forbid it and the paper did it.

THE FIX. One model, both arms, a separate between-wording standard deviation for
each, so the DIFFERENCE has a posterior:

    d = beta + u_variant[v] + u_template[t] + u_pair[p] + eps
    u_variant[v] ~ Normal(0, sigma_arm[arm(v)])

sigma_semantic and sigma_null are then estimated jointly on the same data with
the same nuisance structure, and P(sigma_null > sigma_semantic) is read straight
off the posterior. beta is shared across arms, which is not an assumption but a
consequence of the design: a null edit changes no meaning, so it cannot move the
true effect, and the null arm's variants are the same request as the semantic
arm's baseline.

ONE WRINKLE, HANDLED. Variant N1 is byte-identical to S1 by construction. Its
true variant effect is therefore not an independent draw from the null arm's
distribution, it is the same draw as S1's. The fit is reported both ways.

THIS DOCSTRING USED TO NAME A DIRECTION FOR THAT BIAS, and the paper copied the
claim into print. It said including N1 makes the null arm look slightly LESS
dispersed than it is. That is only true when sigma_semantic is the smaller of
the two -- duplicating S1's draw into the null arm pulls sigma_null toward
sigma_semantic, whichever way that is. On this panel sigma_semantic is the
LARGER on three of four checkpoints, and dropping N1 moves sigma_null down on
two and up on two. There is no unconditional direction to assert, because the
sign of the bias depends on the very ordering the fit exists to test. Do not
put one back.

    .venv/Scripts/python.exe paper-a/src/fit_arm_contrast.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import numpyro

numpyro.set_host_device_count(2)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data" / "delta_stability"
OUT = DATA / "arm_contrast.json"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402


def load():
    rows = {}
    for f in sorted(DATA.glob("delta_*.jsonl")):
        for r in st.read_jsonl(f):
            if (r.get("white_margin") is not None
                    and r.get("black_margin") is not None):
                rows[(r["model"], r["variant"], r["template"], r["pair"])] = r
    return list(rows.values())


def fit(sub, label, warmup=1500, samples=1500):
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    import numpyro.diagnostics as diag
    from numpyro.infer import MCMC, NUTS

    def index(key):
        lv = sorted({r[key] for r in sub})
        m = {v: i for i, v in enumerate(lv)}
        return jnp.array([m[r[key]] for r in sub]), len(lv), lv

    d = jnp.array([r["white_margin"] - r["black_margin"] for r in sub])
    vi, nv, variants = index("variant")
    ti, nt, _ = index("template")
    pi, npair, _ = index("pair")
    # arm of each VARIANT level, not of each row
    arm_of_variant = jnp.array(
        [0 if next(r for r in sub if r["variant"] == v)["variant_kind"] == "semantic"
         else 1 for v in variants])
    scale = float(np.std([float(x) for x in d])) or 1.0

    def model(d=None):
        beta = numpyro.sample("beta", dist.Normal(0, 2 * scale))
        s_arm = numpyro.sample("sigma_arm", dist.HalfNormal(scale).expand([2]))
        s_t = numpyro.sample("sigma_template", dist.HalfNormal(scale))
        s_p = numpyro.sample("sigma_pair", dist.HalfNormal(scale))
        s_e = numpyro.sample("sigma_resid", dist.HalfNormal(2 * scale))
        # Non-centred throughout: with six variants per arm the centred form
        # funnels badly exactly under the null this test is powered to detect.
        with numpyro.plate("V", nv):
            zv = numpyro.sample("z_variant", dist.Normal(0, 1))
        with numpyro.plate("T", nt):
            zt = numpyro.sample("z_template", dist.Normal(0, 1))
        with numpyro.plate("P", npair):
            zp = numpyro.sample("z_pair", dist.Normal(0, 1))
        u_v = s_arm[arm_of_variant] * zv
        mu = beta + u_v[vi] + s_t * zt[ti] + s_p * zp[pi]
        numpyro.sample("obs", dist.Normal(mu, s_e), obs=d)

    mcmc = MCMC(NUTS(model, target_accept_prob=0.95), num_warmup=warmup,
                num_samples=samples, num_chains=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), d=d)
    s = mcmc.get_samples()
    g = mcmc.get_samples(group_by_chain=True)
    rhat = max(float(np.nanmax(diag.gelman_rubin(v))) for v in g.values()
               if v.ndim <= 3 and v.shape[0] > 1)
    div = int(mcmc.get_extra_fields().get("diverging", np.zeros(1)).sum()) \
        if mcmc.get_extra_fields() else 0

    sem = np.asarray(s["sigma_arm"])[:, 0]
    nul = np.asarray(s["sigma_arm"])[:, 1]
    diff = nul - sem

    def q(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 50)),
                float(np.percentile(a, 97.5))]

    return dict(model=label, n=len(sub), n_variants=nv,
                sigma_semantic=q(sem), sigma_null=q(nul),
                difference_null_minus_semantic=q(diff),
                prob_null_exceeds_semantic=float((diff > 0).mean()),
                ratio_null_over_semantic=q(nul / np.maximum(sem, 1e-9)),
                beta=q(np.asarray(s["beta"])),
                max_rhat=rhat, divergences=div)


def main() -> int:
    rows = load()
    if not rows:
        sys.exit("no data")
    out = {}
    print("=" * 94)
    print("DO SEMANTICALLY NULL EDITS MOVE THE EFFECT MORE THAN PARAPHRASES DO?")
    print("One joint fit per model, a separate between-wording SD per arm, so the")
    print("difference has a posterior instead of being compared by eye.")
    print("=" * 94)
    for drop_dup in (False, True):
        tag = "N1 dropped (byte-identical to S1)" if drop_dup else "all 12 variants"
        print(f"\n--- {tag} ---")
        print(f"{'model':<26}{'sigma_sem':>11}{'sigma_null':>12}"
              f"{'difference (null - sem)':>26}{'P(null>sem)':>13}{'rhat':>7}")
        for m in sorted({r["model"] for r in rows}):
            sub = [r for r in rows if r["model"] == m
                   and not (drop_dup and r["variant"] == "N1")]
            res = fit(sub, m)
            out.setdefault(m, {})["dropN1" if drop_dup else "all"] = res
            d = res["difference_null_minus_semantic"]
            print(f"{m:<26}{res['sigma_semantic'][1]:>11.4f}{res['sigma_null'][1]:>12.4f}"
                  f"{f'{d[1]:+.4f} [{d[0]:+.4f},{d[2]:+.4f}]':>26}"
                  f"{res['prob_null_exceeds_semantic']:>13.3f}{res['max_rhat']:>7.3f}")

    print("\nREADING. A posterior probability near 0.5 means the data cannot tell the two")
    print("arms apart. The defensible claim is not that nulls move the effect MORE than")
    print("paraphrases, but that they move it AT ALL, and by an amount of the same order.")
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
