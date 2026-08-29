"""A noise floor for the frontier arm, from a replicate the design already paid for.

WHY THIS WAS MISSING AND WHY IT MATTERS. §4.7 reports a between-wording
dispersion on four API checkpoints and compares its ratio to the open-weight
panel's. §5.2 establishes that the local panel's noise floor is exactly zero --
284 of 284 repeats bitwise identical -- so a local dispersion is known to be
wording and not jitter. The frontier arm had no such floor, which leaves the
obvious objection open: an API is a shared, batched, non-deterministic service,
so some of that dispersion could be the service rather than the wording. Without
a floor the §4.7 ratio cannot be read as the local one is.

THE FLOOR WAS ALREADY ON DISK, UNCOMPUTED. Variant N1 is declared "baseline;
identical to S1" in experiment_delta_stability.VARIANTS, and it is: building
both prompts over every template and name gives 72 of 72 byte-identical pairs
(system string and user message alike). The frontier runner iterates VARIANTS
blindly, so every cell was sent to the API TWICE under two labels, on separate
requests. Any difference between the S1 and N1 readings of the same cell is
therefore measurement noise by construction -- there is no wording difference
left for it to be.

THE COMPARISON HAS TO BE AT THE RIGHT LEVEL, WHICH IS THE EASY THING TO GET
WRONG. §4.7's `sd_across_wordings` is the spread of twelve WORDING MEANS, each
already an average over twelve name pairs and three templates. A noise floor
computed per cell is a much larger number measuring a different thing, and
setting the two side by side would understate the dispersion badly. So the
floor is built at the same level:

  for each name pair p, average the superiority score over templates under S1
  to get a_p and under N1 to get b_p; let d_p = a_p - b_p. The two readings are
  the same prompt, so E[d_p] = 0 and Var(d_p) = 2 Var_noise(a_p). A wording
  mean averages over the n pairs, so the noise SD OF A WORDING MEAN is

      sigma_noise = sd(d_p) / sqrt(2n)

  on n - 1 degrees of freedom, with the pair as the resampling unit exactly as
  §6.1 requires.

AND THE DISPERSION HAS TO BE CORRECTED, not merely compared. Each of the twelve
wording means was measured once, so the observed spread already CONTAINS this
noise: sd_obs^2 = sigma_wording^2 + sigma_noise^2. The wording-attributable SD
is therefore sqrt(sd_obs^2 - sigma_noise^2), and reporting sd_obs as though it
were all wording overstates the effect this paper is about. The corrected value
is what §4.7 should quote.

A floor at or above the observed dispersion would mean §4.7's frontier numbers
are consistent with pure service noise and the section would have to be
withdrawn. The question is worth asking precisely because the answer could go
either way.

TWO SCALES, BOTH REPORTED, NEVER MIXED. The margin difference is the raw
quantity and its SD is the floor in log-odds. The probability-of-superiority
score is what §4.7 aggregates, and only that one compares with the dispersion
§4.7 prints. Dividing one by the other is the error §4.1 exists to warn about.

AND THE SAME FLOOR IS COMPUTED FOR THE OPEN-WEIGHT PANEL, WHICH IS THE WHOLE
POINT. The first version of this file corrected the frontier ratios and left
the local ones alone, on the stated ground that "the local floor is zero". That
is false, and the evidence against it was already in this repository:
delta_stability/noise_floor.json records bitwise agreement of 13.9 % to 66.7 %
on the Study 2 cells that produce the local ratios, and §5.2 says so in prose.
The "284 of 284 identical" result is real but belongs to the SECOND-TASK study
on housing and moderation, under a serving configuration Study 2 predates -- a
different corpus from the one being corrected. Correcting one panel with an
estimator and not the other, while comparing the two panels' corrected numbers,
is exactly the kind of unreported measurement choice this paper exists to
measure. Both panels now go through the same code, and the artifact records the
floor for each.

The local floors are not negligible: 0.029 on Llama-2-7B and 0.035 on
Llama-3.1-8B -- the latter numerically equal to gpt-4o's -- removing 9 % and
20 % of the observed between-wording variance respectively.

    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/analyze_frontier_noise.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "paper-a" / "src"
D = ROOT / "paper-a" / "data" / "frontier"
DS = ROOT / "paper-a" / "data" / "delta_stability"
OUT = D / "frontier_noise_floor.json"
ORDER = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
LOCAL = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]

sys.path.insert(0, str(SRC))
import experiment_delta_stability as ds  # noqa: E402


def superiority(w, b):
    """1 / 0.5 / 0, exactly as effectsize.superiority scores a pair."""
    return 1.0 if w > b else (0.0 if w < b else 0.5)


def prompts_identical() -> tuple[bool, int, int]:
    """Verify the premise instead of trusting the note that states it."""
    s1, n1 = ds.VARIANTS["S1"], ds.VARIANTS["N1"]
    same = diff = 0
    if s1["system"] != n1["system"]:
        return False, 0, 1
    for _t, body in ds.TEMPLATES.items():
        for wname, bname in ds.PAIRS:
            for nm in (wname, bname):
                if ds.user_message(s1, nm, body) == ds.user_message(n1, nm, body):
                    same += 1
                else:
                    diff += 1
    return diff == 0, same, diff


def floor_for(rows, sd_obs, r_obs):
    """The replicate floor for one model, from its S1/N1 cells.

    Panel-agnostic on purpose. The first version of this file inlined this in a
    loop over the frontier models only, and the open-weight panel was left
    uncorrected on the false ground that its floor was zero. A function both
    panels call cannot drift apart that way.

    `sd_obs` is the observed between-wording SD on the superiority scale and
    `r_obs` the published dispersion-to-effect ratio; both may be None, in
    which case only the floor itself is returned.
    """
    by = collections.defaultdict(dict)
    for r in rows:
        if r.get("variant") in ("S1", "N1"):
            by[(r.get("template"), r.get("pair"))][r["variant"]] = r

    d_margin, identical, usable = [], 0, 0
    a_by_pair = collections.defaultdict(list)
    b_by_pair = collections.defaultdict(list)
    for (_t, pair), v in by.items():
        a, b = v.get("S1"), v.get("N1")
        if not a or not b:
            continue
        if any(x.get(f"{arm}_margin") is None
               for x in (a, b) for arm in ("white", "black")):
            continue
        usable += 1
        d_margin.append((a["white_margin"] - a["black_margin"])
                        - (b["white_margin"] - b["black_margin"]))
        a_by_pair[pair].append(superiority(a["white_margin"], a["black_margin"]))
        b_by_pair[pair].append(superiority(b["white_margin"], b["black_margin"]))
        if (a["white_margin"] == b["white_margin"]
                and a["black_margin"] == b["black_margin"]):
            identical += 1

    pairs = sorted(set(a_by_pair) & set(b_by_pair))
    if usable < 2 or len(pairs) < 2:
        return dict(n_replicate_cells=usable, usable=False,
                    reason="no usable replicate (margins censored)")

    dp = np.array([np.mean(a_by_pair[p]) - np.mean(b_by_pair[p])
                   for p in pairs])
    n_pairs = len(pairs)
    # THE FLOOR ON A WORDING MEAN. Var(d_p) = 2 Var_noise(a_p); a wording mean
    # averages n pairs, so its noise SD is sd(d_p) / sqrt(2n).
    sigma_noise = float(dp.std(ddof=1) / np.sqrt(2 * n_pairs))
    am = np.array(d_margin)

    corrected = share = None
    if sd_obs is not None:
        # sd_obs already CONTAINS the noise: each wording mean was read once.
        corrected = float(np.sqrt(max(sd_obs ** 2 - sigma_noise ** 2, 0.0)))
        share = float(min(sigma_noise ** 2 / sd_obs ** 2, 1.0)) if sd_obs else None

    rec = dict(
        n_replicate_cells=usable, n_pairs=n_pairs, usable=True,
        n_bitwise_identical=identical,
        frac_bitwise_identical=identical / usable,
        noise_sd_margin_logodds=float(am.std(ddof=1)),
        noise_mean_margin_logodds=float(am.mean()),
        noise_max_abs_margin_logodds=float(np.abs(am).max()),
        noise_sd_of_a_wording_mean=sigma_noise,
        observed_sd_across_wordings=sd_obs,
        wording_sd_corrected_for_noise=corrected,
        noise_share_of_observed_variance=share,
        noise_over_observed_sd=(sigma_noise / sd_obs) if sd_obs else None,
        published_ratio_sd_to_effect=r_obs,
        # The correction scales the published ratio by corrected/observed: the
        # denominator is the effect, which this replicate says nothing about.
        ratio_sd_to_effect_corrected=(
            float(r_obs * corrected / sd_obs)
            if r_obs is not None and sd_obs else None),
    )
    return rec


def jsonl(paths):
    out = []
    for p in paths:
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                out.append(json.loads(ln))
    return out


def show(name, rec):
    if not rec.get("usable"):
        print(f"{name:<26}{rec['n_replicate_cells']:>7}   {rec.get('reason','')}")
        return
    sd_obs = rec.get("observed_sd_across_wordings")
    corr = rec.get("wording_sd_corrected_for_noise")
    share = rec.get("noise_share_of_observed_variance")
    print(f"{name:<26}{rec['n_replicate_cells']:>7}"
          f"{rec['n_bitwise_identical']:>11}"
          f"{rec['noise_sd_of_a_wording_mean']:>11.4f}"
          + (f"{sd_obs:>12.4f}" if sd_obs is not None else f"{'--':>12}")
          + (f"{corr:>11.4f}" if corr is not None else f"{'--':>11}")
          + (f"{share:>9.1%}" if share is not None else f"{'--':>9}"))


def main() -> int:
    ok, n_same, n_diff = prompts_identical()
    out = {
        "_what": "The measurement noise floor of the frontier arm, from the "
                 "byte-identical S1/N1 replicate the wording design already "
                 "contains.",
        "_why": "§4.7 reports a between-wording dispersion on an API, which is "
                "a shared non-deterministic service. Without a floor that "
                "dispersion cannot be distinguished from service jitter, and "
                "the §5.2 floor covers only the local panel.",
        "_premise_verified": {
            "s1_and_n1_are_the_same_prompt": ok,
            "n_prompt_slots_identical": n_same,
            "n_prompt_slots_differing": n_diff,
            "note": ds.VARIANTS["N1"].get("note", ""),
        },
        "models": {},
    }
    if not ok:
        print("S1 and N1 are NOT byte-identical; the replicate does not exist")
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 1

    fm = {}
    p = D / "frontier_margin_analysis.json"
    if p.exists():
        fm = json.loads(p.read_text(encoding="utf-8")).get("models", {})

    print(f"{'model':<26}{'cells':>7}{'identical':>11}{'noise sd':>11}"
          f"{'obs sd':>12}{'corrected':>11}{'noise%':>9}")
    print("-" * 88)
    # ---- the frontier panel -------------------------------------------
    for m in ORDER:
        f = D / f"margin_{m}.jsonl"
        if not f.exists():
            continue
        rec = floor_for(jsonl([f]),
                        fm.get(m, {}).get("sd_across_wordings"),
                        fm.get(m, {}).get("ratio_sd_to_effect"))
        rec["panel"] = "frontier"
        out["models"][m] = rec
        show(m, rec)

    # ---- the open-weight panel, through the SAME function ---------------
    # This is the correction round 5 forced. The first version of this file
    # computed the frontier floors, corrected the frontier ratios, and left the
    # open-weight ratios alone on the stated ground that "the local floor is
    # zero". It is not: Study 2's own S1/N1 cells agree bitwise on 13.9 % to
    # 66.7 %, which delta_stability/noise_floor.json has recorded all along and
    # §5.2 states in prose. The 284-of-284 identical result is real and belongs
    # to the second-task study on housing and moderation, under a serving
    # configuration Study 2 predates -- a different corpus from the one whose
    # ratios were being left uncorrected.
    s2p = DS / "study2_v2.json"
    s2 = json.loads(s2p.read_text(encoding="utf-8")) if s2p.exists() else {}
    ds_rows = jsonl(sorted(DS.glob("*.jsonl")))
    if ds_rows:
        print()
        print("  open-weight panel, same estimator")
        for m in LOCAL:
            rows = [r for r in ds_rows if r.get("model") == m]
            if not rows or m not in s2:
                continue
            sd_obs = s2[m].get("ps_sd_across_wordings")
            eff = abs(s2[m]["overall"]["superiority"]["est"] - 0.5)
            r_obs = (sd_obs / eff) if sd_obs is not None and eff > 1e-9 else None
            rec = floor_for(rows, sd_obs, r_obs)
            rec["panel"] = "local"
            out["models"][m] = rec
            show(m, rec)


    # THE SUMMARY IS PER PANEL. A single pooled summary is what let the prose
    # say "the local floor is zero" while the artifact held only frontier
    # numbers: there was no local row to contradict it.
    for _panel in ("frontier", "local"):
        _p = {k: v for k, v in out["models"].items()
              if v.get("panel") == _panel and v.get("usable")}
        if not _p:
            continue
        _f = [v["noise_sd_of_a_wording_mean"] for v in _p.values()]
        _s = [v["noise_share_of_observed_variance"] for v in _p.values()
              if v.get("noise_share_of_observed_variance") is not None]
        _r = [v["ratio_sd_to_effect_corrected"] for v in _p.values()
              if v.get("ratio_sd_to_effect_corrected") is not None]
        _o = [v["published_ratio_sd_to_effect"] for v in _p.values()
              if v.get("published_ratio_sd_to_effect") is not None]
        out[f"summary_{_panel}"] = dict(
            n_models=len(_p),
            n_bitwise_deterministic=sum(
                1 for v in _p.values() if v["frac_bitwise_identical"] == 1.0),
            min_frac_bitwise_identical=min(
                v["frac_bitwise_identical"] for v in _p.values()),
            max_frac_bitwise_identical=max(
                v["frac_bitwise_identical"] for v in _p.values()),
            min_noise_sd_of_a_wording_mean=min(_f),
            max_noise_sd_of_a_wording_mean=max(_f),
            min_noise_share_of_variance=min(_s) if _s else None,
            max_noise_share_of_variance=max(_s) if _s else None,
            min_ratio_published=min(_o) if _o else None,
            max_ratio_published=max(_o) if _o else None,
            min_ratio_corrected=min(_r) if _r else None,
            max_ratio_corrected=max(_r) if _r else None,
            floor_is_zero_on_every_model=all(
                v["noise_sd_of_a_wording_mean"] == 0.0 for v in _p.values()),
        )
    out["_symmetry_note"] = (
        "Both panels are corrected with the same estimator. An earlier version "
        "corrected only the frontier ratios, on the ground that the local "
        "floor was zero; the local floor is not zero on the Study 2 corpus "
        "that produces the local ratios, and the 284-of-284 identical result "
        "that was cited for it belongs to the second-task study on a different "
        "serving configuration.")

    live = {k: v for k, v in out["models"].items() if v.get("usable")}
    rats = [v["noise_over_observed_sd"] for v in live.values()
            if v.get("noise_over_observed_sd") is not None]
    shares = [v["noise_share_of_observed_variance"] for v in live.values()
              if v.get("noise_share_of_observed_variance") is not None]
    out["summary"] = dict(
        n_models_with_a_replicate=len(live),
        n_models_bitwise_deterministic=sum(
            1 for v in live.values() if v["frac_bitwise_identical"] == 1.0),
        min_frac_bitwise_identical=min(
            (v["frac_bitwise_identical"] for v in live.values()), default=None),
        max_frac_bitwise_identical=max(
            (v["frac_bitwise_identical"] for v in live.values()), default=None),
        min_noise_sd_of_a_wording_mean=min(
            (v["noise_sd_of_a_wording_mean"] for v in live.values()),
            default=None),
        max_noise_sd_of_a_wording_mean=max(
            (v["noise_sd_of_a_wording_mean"] for v in live.values()),
            default=None),
        min_noise_over_observed_sd=min(rats, default=None),
        max_noise_over_observed_sd=max(rats, default=None),
        min_noise_share_of_variance=min(shares, default=None),
        max_noise_share_of_variance=max(shares, default=None),
        # The verdict, stated by the data. The dispersion survives only if the
        # floor is below the observed spread on every model that has both.
        dispersion_exceeds_the_floor_on_every_model=(
            bool(rats) and all(r < 1.0 for r in rats)),
        n_models_where_noise_exceeds_dispersion=sum(1 for r in rats if r >= 1.0),
        # The headline contrast with §5.2, which found a floor of exactly zero
        # on the local panel over 284 repeats.
        api_is_bitwise_reproducible=all(
            v["frac_bitwise_identical"] == 1.0 for v in live.values()),
        min_ratio_corrected=min(
            (v["ratio_sd_to_effect_corrected"] for v in live.values()
             if v.get("ratio_sd_to_effect_corrected") is not None),
            default=None),
        max_ratio_corrected=max(
            (v["ratio_sd_to_effect_corrected"] for v in live.values()
             if v.get("ratio_sd_to_effect_corrected") is not None),
            default=None),
        n_models_with_a_corrected_ratio=sum(
            1 for v in live.values()
            if v.get("ratio_sd_to_effect_corrected") is not None),
    )
    s = out["summary"]
    print()
    for _panel, _label in (("frontier", "frontier API"),
                           ("local", "open-weight")):
        _s = out.get(f"summary_{_panel}")
        if not _s:
            continue
        print(f"  {_label} panel ({_s['n_models']} models)")
        print(f"    bitwise-identical replicate cells "
              f"{_s['min_frac_bitwise_identical']:.1%} to "
              f"{_s['max_frac_bitwise_identical']:.1%}")
        print(f"    noise floor on a wording mean   "
              f"{_s['min_noise_sd_of_a_wording_mean']:.4f} to "
              f"{_s['max_noise_sd_of_a_wording_mean']:.4f}"
              + ("   (zero on every model)"
                 if _s["floor_is_zero_on_every_model"] else ""))
        if _s["min_noise_share_of_variance"] is not None:
            print(f"    share of observed variance      "
                  f"{_s['min_noise_share_of_variance']:.1%} to "
                  f"{_s['max_noise_share_of_variance']:.1%}")
        if _s["min_ratio_corrected"] is not None:
            print(f"    ratio sd/effect  published      "
                  f"{_s['min_ratio_published']:.2f} to "
                  f"{_s['max_ratio_published']:.2f}")
            print(f"                     corrected      "
                  f"{_s['min_ratio_corrected']:.2f} to "
                  f"{_s['max_ratio_corrected']:.2f}")
        print()
    print(f"  dispersion exceeds the floor on every model: "
          f"{out['summary']['dispersion_exceeds_the_floor_on_every_model']}")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
