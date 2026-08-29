"""The pairing itself is a researcher degree of freedom, and it is free.

WHAT THIS IS. Section 3 explains why each résumé is scored ALONE and the pair
differenced afterwards: presenting both in one prompt makes the answer a function
of presentation order on every checkpoint we gated. That decision has a
consequence the paper did not follow up. If the two résumés are never in the same
prompt, then nothing in the MEASUREMENT pairs them. The pairing happens in the
analysis, after every number already exists, and which White name is placed
opposite which Black name is a choice the analyst makes.

`stimuli.build_name_grid` says so in its own docstring -- "Which White name is
placed opposite which Black name is arbitrary" -- and that was written as a
reassurance. This script asks whether it is one.

WHY IT CAN MATTER AT ALL, given that the pairing looks like it should cancel.
On the log-odds scale the mean paired difference is the difference of the two
means, so it is invariant to the matching and the reassurance is exactly right.
It is not right for the statistic this literature actually reports. The primary
effect size here, and the discordant-pair analysis that correspondence audits
have used since Bertrand and Mullainathan, are functions of WHICH résumé beat
WHICH -- and that is a property of the matching, not of the two marginal
distributions. Anything computed pair-by-pair inherits the choice: the
probability of superiority, the sign test, the paired interval's width.

WHAT IS COMPUTED, per model, on the Study 4 grid (48 pairs x 3 templates x 12
wordings, every résumé scored alone):

  1. The headline probability of superiority under the grid's actual pairing.
  2. Its distribution over random re-pairings within gender, which is the set of
     pairings an equally careful researcher could have written down.
  3. The exact BEST and WORST pairing, by maximum-weight bipartite matching over
     the pooled win counts. These are not adversarial constructions a real
     analyst would find; they bound what the choice can do.
  4. Where the TOKEN-BALANCED pairings sit inside that distribution -- because
     §4.3's balanced subset is itself one particular matching, and a reader is
     entitled to ask whether its effect differs because it is balanced or
     because it is a different matching.

The comparison the paper needs is against its own yardstick: the between-wording
standard deviation. A dispersion from re-pairing that is small next to it is a
footnote; one that is comparable is another axis of the same size, obtained
without touching the model, the names, the prompt or the data.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/analyze_pairing_freedom.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
NAMES = ROOT / "paper-a" / "data" / "names"
INSTR = ROOT / "paper-a" / "data" / "instrument"
VCOMP = ROOT / "paper-a" / "data" / "delta_stability" / "variance_components.json"
OUT = NAMES / "pairing_freedom.json"

N_PERM = 20_000
SEED = 20260801

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B",
         "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}


def load(model: str):
    """Per-cell margins keyed by (template, variant), one vector per full name.

    Deduplicated on the full cell key because the recheck subset re-measures
    some cells; the same rule every other analysis in this paper uses.
    """
    f = NAMES / f"names_{model}.jsonl"
    if not f.exists():
        return None
    best = {}
    for r in st.read_jsonl(f):
        if r.get("white_margin") is None or r.get("black_margin") is None:
            continue
        best[(r["variant"], r["template"], r["pair"])] = r
    # NOTE. In this schema `white`/`black` are the thresholded VERDICTS and the
    # names are `white_name`/`black_name`. `stimuli.NAME_GRID` uses the other
    # convention, so the two are kept apart explicitly here.
    cells = defaultdict(lambda: {"white": {}, "black": {}, "gender": {}})
    for r in best.values():
        c = cells[(r["template"], r["variant"])]
        c["white"][r["white_name"]] = r["white_margin"]
        c["black"][r["black_name"]] = r["black_margin"]
        c["gender"][r["white_name"]] = r["gender"]
        c["gender"][r["black_name"]] = r["gender"]
    return dict(cells)


def build_tensors(cells):
    """Stack cells into arrays: W[g] is (n_cells, n_names) for that gender."""
    keys = sorted(cells)
    out = {}
    for gender in ("female", "male"):
        wnames = sorted({n for k in keys for n, g in cells[k]["gender"].items()
                         if g == gender and n in cells[k]["white"]})
        bnames = sorted({n for k in keys for n, g in cells[k]["gender"].items()
                         if g == gender and n in cells[k]["black"]})
        W = np.array([[cells[k]["white"][n] for n in wnames] for k in keys])
        B = np.array([[cells[k]["black"][n] for n in bnames] for k in keys])
        out[gender] = dict(cells=keys, wnames=wnames, bnames=bnames, W=W, B=B)
    return out


def psup_for_matching(T, perms):
    """Probability of superiority pooled over cells, for one matching per gender.

    `perms[gender]` maps white index -> black index.
    """
    wins = 0
    total = 0
    for gender, t in T.items():
        p = perms[gender]
        w = t["W"]
        b = t["B"][:, p]
        wins += int((w > b).sum())
        total += w.size
    return wins / total if total else float("nan")


def main() -> int:
    rng = np.random.default_rng(SEED)
    vcomp = json.loads(VCOMP.read_text(encoding="utf-8")) \
        if VCOMP.exists() else {}
    bal = json.loads((INSTR / "token_balanced_grid.json").read_text(
        encoding="utf-8")) if (INSTR / "token_balanced_grid.json").exists() else None

    out = {"n_perm": N_PERM, "seed": SEED, "models": {}}
    print("=" * 100)
    print("THE PAIRING IS CHOSEN AFTER THE DATA EXISTS. WHAT DOES THE CHOICE MOVE?")
    print("=" * 100)
    print(f"{'model':<15}{'as built':>10}{'perm mean':>11}{'perm SD':>9}"
          f"{'p2.5':>8}{'p97.5':>8}{'best':>8}{'worst':>8}{'sign flips':>12}")

    for model in ORDER:
        cells = load(model)
        if not cells:
            continue
        T = build_tensors(cells)

        # ---- the pairing the grid actually used ---------------------------
        # Read straight off the grid definition rather than inferred, so a
        # change to build_name_grid cannot silently desynchronise this.
        wb = {}
        for p in st.NAME_GRID:
            wb.setdefault(p["gender"], {})[p["white"]] = p["black"]
        perms_actual = {}
        for gender, t in T.items():
            bi = {n: i for i, n in enumerate(t["bnames"])}
            perms_actual[gender] = np.array(
                [bi[wb[gender][n]] for n in t["wnames"]])

        p_actual = psup_for_matching(T, perms_actual)

        # ---- random re-pairings ------------------------------------------
        vals = np.empty(N_PERM)
        for i in range(N_PERM):
            perms = {g: rng.permutation(len(t["wnames"]))
                     for g, t in T.items()}
            vals[i] = psup_for_matching(T, perms)

        # ---- exact best and worst ----------------------------------------
        # weight(w,b) = number of cells in which white name w beats black b.
        best_wins = 0
        worst_wins = 0
        total = 0
        for gender, t in T.items():
            w = t["W"][:, :, None]           # (cells, nw, 1)
            b = t["B"][:, None, :]           # (cells, 1, nb)
            wins = (w > b).sum(axis=0)       # (nw, nb)
            n_cells = t["W"].shape[0]
            ri, ci = linear_sum_assignment(-wins)
            best_wins += int(wins[ri, ci].sum())
            ri, ci = linear_sum_assignment(wins)
            worst_wins += int(wins[ri, ci].sum())
            total += n_cells * len(t["wnames"])
        p_best = best_wins / total
        p_worst = worst_wins / total

        # ---- where do the token-balanced pairings sit? -------------------
        p_balanced = None
        n_balanced_pairs = None
        if bal:
            # A pairing is balanced iff the two full names occupy the same
            # number of tokens on EVERY tokenizer in the panel, using the
            # in-context counts already measured by build_token_balanced_grid.
            per = bal["per_model_tokens"]
            models_b = bal["models"]
            wins_b, tot_b = 0, 0
            npair = 0
            for gender, t in T.items():
                ok = np.zeros((len(t["wnames"]), len(t["bnames"])), dtype=bool)
                for i, a in enumerate(t["wnames"]):
                    for j, c in enumerate(t["bnames"]):
                        if all(per[m]["full"].get(a) is not None
                               and per[m]["full"].get(c) is not None
                               and per[m]["full"][a] == per[m]["full"][c]
                               for m in models_b):
                            ok[i, j] = True
                if not ok.any():
                    continue
                w = t["W"][:, :, None]
                b = t["B"][:, None, :]
                wins = (w > b).sum(axis=0)
                n_cells = t["W"].shape[0]
                # maximum-cardinality matching restricted to balanced edges,
                # broken toward neither side: use a cost of 0 on legal edges and
                # a large penalty on illegal ones, then keep only legal pairs.
                cost = np.where(ok, 0.0, 1e6)
                ri, ci = linear_sum_assignment(cost)
                keep = [(i, j) for i, j in zip(ri, ci) if ok[i, j]]
                npair += len(keep)
                for i, j in keep:
                    wins_b += int(wins[i, j])
                    tot_b += n_cells
            if tot_b:
                p_balanced = wins_b / tot_b
                n_balanced_pairs = npair

        sd = float(vals.std(ddof=1))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        flips = float(((vals - 0.5) * (p_actual - 0.5) < 0).mean())
        sigma_word = None
        if vcomp:
            v = (vcomp.get("models") or vcomp).get(model)
            if isinstance(v, dict):
                for k in ("sigma_variant", "sd_across_wordings",
                          "raw_sd_psup", "sigma_word"):
                    if isinstance(v.get(k), (int, float)):
                        sigma_word = float(v[k])
                        break

        out["models"][model] = dict(
            p_actual=p_actual, perm_mean=float(vals.mean()), perm_sd=sd,
            perm_ci=[float(lo), float(hi)],
            perm_min=float(vals.min()), perm_max=float(vals.max()),
            best_possible=p_best, worst_possible=p_worst,
            range_best_worst=p_best - p_worst,
            sign_flip_rate_vs_actual=flips,
            n_balanced_pairs=n_balanced_pairs, p_balanced=p_balanced,
            sigma_word_reference=sigma_word,
            n_cells=int(sum(t["W"].shape[0] for t in T.values())),
        )
        print(f"{SHORT.get(model, model):<15}{p_actual:>10.4f}"
              f"{vals.mean():>11.4f}{sd:>9.4f}{lo:>8.4f}{hi:>8.4f}"
              f"{p_best:>8.4f}{p_worst:>8.4f}{flips:>11.1%}")

    if out["models"]:
        sds = [v["perm_sd"] for v in out["models"].values()]
        rng_bw = [v["range_best_worst"] for v in out["models"].values()]
        out["summary"] = dict(
            min_perm_sd=float(min(sds)), max_perm_sd=float(max(sds)),
            min_best_worst_range=float(min(rng_bw)),
            max_best_worst_range=float(max(rng_bw)),
            n_models=len(sds),
            n_models_where_repairing_flips_sign=int(sum(
                1 for v in out["models"].values()
                if v["sign_flip_rate_vs_actual"] > 0)),
        )
        print("\n  SD of the headline effect across defensible re-pairings: "
              f"{min(sds):.4f} to {max(sds):.4f}")
        print(f"  best-minus-worst achievable range: "
              f"{min(rng_bw):.4f} to {max(rng_bw):.4f}")
        print("\n  READING. On the log-odds scale the mean paired difference is")
        print("  the difference of the two means and does not move at all. The")
        print("  probability of superiority does, because it is a function of")
        print("  which résumé beat which, and that is the analyst's choice.")

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
