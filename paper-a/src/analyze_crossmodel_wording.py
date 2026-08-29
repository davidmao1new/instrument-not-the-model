# -*- coding: utf-8 -*-
"""Is the wording effect a property of the wording, or of the model?

Across the five task contexts that share the same 12-wording grid, the
per-wording race-effect profiles correlate near zero between model
FAMILIES -- the wording effect is almost entirely model-idiosyncratic --
with one exception: the two same-family Mistral checkpoints agree. That
both sharpens and softens the paper: an auditor cannot transfer a
"safe wording" between models, and what does transfer follows lineage.

PROVENANCE. Drafted inside an adversarial audit workflow (25 Aug 2026)
and independently recomputed from the method description alone before
adoption; every number matched to the printed digits.

    sh paper-a/src/_py.sh paper-a/src/analyze_crossmodel_wording.py

Writes paper-a/data/instrument/crossmodel_wording.json.
"""
import json, itertools, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rng = np.random.default_rng(20260825)
import pathlib
_R = pathlib.Path(__file__).resolve().parents[2]
ROOT = str(_R / "paper-a" / "data")
OUT = _R / "paper-a" / "data" / "instrument" / "crossmodel_wording.json"
RESULTS = {"specifications": {}, "per_context": {}, "loco": {}}
MODELS = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
          "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
VARIANTS = ["S1","S2","S3","S4","S5","S6","N1","N2","N3","N4","N5","N6"]
ARM = np.array([0]*6 + [1]*6)          # 0 = semantic arm, 1 = null-edit arm
PAIRS = list(itertools.combinations(range(4), 2))
FAM = (2, 3)                            # mistral v0.1 x v0.3, the one same-family pair

def load(path, filt=None):
    recs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") or r.get("white_margin") is None or r.get("black_margin") is None:
            continue
        if filt and not filt(r):
            continue
        recs.append(r)
    return recs

CONTEXTS = {}
CONTEXTS["resume_BA"] = {m: load(f"{ROOT}/delta_stability/delta_{m}.jsonl") for m in MODELS}
for occ in ["SWE", "RN"]:
    CONTEXTS[f"resume_{occ}"] = {m: load(f"{ROOT}/occupation/occ_{m}.jsonl",
                                         lambda r, o=occ: r["occupation"] == o) for m in MODELS}
for dom in ["housing", "moderation"]:
    CONTEXTS[dom] = {m: load(f"{ROOT}/second_task/{dom}_{m}.jsonl") for m in MODELS}
CTX = list(CONTEXTS)

def matrix(pm, stat="mean"):
    """12 wordings x 4 models; per-wording mean of (white_margin - black_margin),
    or probability-of-superiority when stat='ps'."""
    E = np.zeros((12, 4))
    for j, m in enumerate(MODELS):
        by = {v: [] for v in VARIANTS}
        for r in pm[m]:
            by[r["variant"]].append(r["white_margin"] - r["black_margin"])
        for i, v in enumerate(VARIANTS):
            x = np.array(by[v])
            E[i, j] = x.mean() if stat == "mean" else \
                      np.mean(np.where(x > 0, 1.0, np.where(x < 0, 0.0, 0.5)))
    return E

def zscore(E):      return (E - E.mean(0)) / E.std(0, ddof=1)
def arm_center(E):
    out = E.copy()
    for a in (0, 1):
        out[ARM == a] -= out[ARM == a].mean(0)
    return out / out.std(0, ddof=1)
def corr(x, y):
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return float(np.mean(x * y))

def block_perm_p(Z, a, b, n=20000):
    """null: shuffle wording labels within each 12-row context block."""
    xs = (Z[:, a] - Z[:, a].mean()) / Z[:, a].std()
    ys = (Z[:, b] - Z[:, b].mean()) / Z[:, b].std()
    obs = float(np.mean(xs * ys))
    nb = Z.shape[0] // 12
    idx = np.arange(Z.shape[0]).reshape(nb, 12)
    perms = np.empty((n, Z.shape[0]), dtype=int)
    for i in range(n):
        perms[i] = np.concatenate([blk[rng.permutation(12)] for blk in idx])
    null = (ys[perms] @ xs) / Z.shape[0]
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n + 1)

for label, stat, center in [("margin-gap scale, grand-centered", "mean", "grand"),
                            ("margin-gap scale, ARM-centered",   "mean", "arm"),
                            ("prob-of-superiority scale",        "ps",   "grand")]:
    Z = np.vstack([(zscore if center == "grand" else arm_center)(matrix(CONTEXTS[c], stat))
                   for c in CTX])
    print(f"\n### pooled 60-wording profiles ({label})")
    rs = {}
    fam_p = None
    for a, b in PAIRS:
        obs, p = block_perm_p(Z, a, b)
        rs[(a, b)] = obs
        if (a, b) == FAM:
            fam_p = float(p)
        tag = "  <-- same family" if (a, b) == FAM else ""
        print(f"  {MODELS[a][:22]:22s} x {MODELS[b][:22]:22s} r={obs:+.3f}  perm-p={p:.4f}{tag}")
    cross = [r for k, r in rs.items() if k != FAM]
    print(f"  same-family r = {rs[FAM]:+.3f}; cross-family mean = {np.mean(cross):+.3f} "
          f"(range {min(cross):+.3f} to {max(cross):+.3f}); same-family rank "
          f"{1 + sum(c > rs[FAM] for c in cross)} of 6")
    share = np.sum(Z.mean(1)**2) * 4 / np.sum(Z**2)
    print(f"  naive shared-wording variance share = {share:.3f} "
          f"(null expectation with 4 models = 0.250); implied mean pairwise r = {(share*4-1)/3:+.3f}")
    RESULTS["specifications"][label] = dict(
        family_r=rs[FAM], family_perm_p=fam_p,
        cross_family_mean=float(np.mean(cross)),
        cross_family_min=float(min(cross)), cross_family_max=float(max(cross)),
        family_rank_of_6=1 + sum(c > rs[FAM] for c in cross),
        shared_variance_share=float(share),
        idiosyncratic_share=float(1 - share))

print("\n### per-context same-family pair vs the five cross-family pairs (margin, grand-centered)")
Zs = {c: zscore(matrix(CONTEXTS[c])) for c in CTX}
ranks, beats = [], 0
for c in CTX:
    rs = {p: corr(Zs[c][:, p[0]], Zs[c][:, p[1]]) for p in PAIRS}
    rank = 1 + sum(1 for k, r in rs.items() if k != FAM and r > rs[FAM])
    xf = np.mean([r for k, r in rs.items() if k != FAM])
    ranks.append(rank); beats += rs[FAM] > xf
    RESULTS["per_context"][c] = dict(family_r=rs[FAM], rank_of_6=rank,
                                     cross_family_mean=float(xf))
    print(f"  {c:12s} same-family r={rs[FAM]:+.3f}  rank {rank}/6  > cross-family mean ({xf:+.3f}): {rs[FAM] > xf}")
print(f"  mean rank {np.mean(ranks):.2f} (chance 3.5); beats cross-family mean in {beats}/5 contexts")
RESULTS["per_context_summary"] = dict(mean_rank=float(np.mean(ranks)),
                                      beats_cross_mean=int(beats),
                                      n_contexts=len(CTX))
print("  NOTE: with 4 models any 'which pair is special' exchangeability test is floor-bounded at p = 1/6.")

print("\n### leave-one-context-out pooled same-family r")
full = np.vstack([Zs[c] for c in CTX])
print(f"  all 5 contexts: r={corr(full[:, 2], full[:, 3]):+.3f}")
for drop in CTX:
    Z = np.vstack([Zs[c] for c in CTX if c != drop])
    print(f"  drop {drop:12s}: r={corr(Z[:, 2], Z[:, 3]):+.3f}")

print("\n### reliability floor from the S1==N1 byte-identical replicate")
worst, worst_at = 1.0, None
for c in CTX:
    for j, m in enumerate(MODELS):
        key = lambda r: (r.get("template", r.get("level")), r["pair"])
        s1 = {key(r): r["white_margin"] - r["black_margin"] for r in CONTEXTS[c][m] if r["variant"] == "S1"}
        n1 = {key(r): r["white_margin"] - r["black_margin"] for r in CONTEXTS[c][m] if r["variant"] == "N1"}
        common = sorted(set(s1) & set(n1))
        d = np.array([s1[k] - n1[k] for k in common])
        sig_wm2 = np.var(d, ddof=1) / 2.0 / len(common)   # noise var of a wording-mean effect
        rel = 1 - sig_wm2 / matrix(CONTEXTS[c])[:, j].var(ddof=1)
        if rel < worst: worst, worst_at = rel, (c, m)
print(f"  minimum per-(context, model) reliability of a wording-effect profile = {worst:.4f} at {worst_at}")

RESULTS["reliability_floor"] = dict(value=float(worst),
                                    at=list(worst_at) if worst_at else None)
RESULTS["_what"] = ("Cross-model correlation of per-wording race-effect "
                    "profiles over five task contexts; the same-family "
                    "Mistral pair against the five cross-family pairs.")
RESULTS["_provenance"] = ("Computed inside an adversarial audit workflow, "
                          "25 Aug 2026; independently reproduced before "
                          "adoption.")
RESULTS["models"] = MODELS
RESULTS["family_pair"] = [MODELS[FAM[0]], MODELS[FAM[1]]]
OUT.write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")
print(f"wrote {OUT}")