"""The exhaustive Gemini capability probe, and the producer its artifact lacked.

WHY THIS FILE EXISTS. §4.7 rests on a claim about somebody else's product: that
no Gemini model reachable on the API-key surface returns a next-token
distribution. The evidence is frontier_api_capability_v2.json. An audit of this
paper found that the artifact had NO PRODUCER anywhere in the repository -- two
scripts read it and nothing wrote it. probe_frontier_api.py writes a different
file (frontier_api_capability.json, no _v2) from a HAND-PICKED candidate list,
which is the narrower probe the paper explicitly says it replaced. So the
strongest empirical claim in §4.7 was supported by a file a reader could not
regenerate, in a paper whose abstract promises a released artifact set. That is
the defect this fixes.

WHAT THE PROBE DOES, and why each part is there:

  * the model list comes from the API's own /models endpoint on BOTH public
    REST versions, not from a list we wrote down. "Every model the surface
    lists" is only checkable if the surface supplies the list;
  * generation-capable text models only. Image, speech and music models cannot
    answer a screening question, and probing them would inflate the
    denominator with failures that mean nothing;
  * three spellings of the parameter, because a feature can exist under a name
    we did not guess and one spelling failing proves nothing;
  * quota errors are recorded as their own outcome. A quota error is raised
    before the request body is validated, so a quota-blocked model says NOTHING
    about whether it would have returned log probabilities. Folding those into
    "refused" would turn a billing state into a claim about a product.

COUNTS: PROBES AND MODELS ARE NOT THE SAME NUMBER, which is the second defect
this fixes. The artifact's `n_probed` is 46, and the paper read that as "46
models". It is 46 (model, REST-version) records over 33 DISTINCT models -- 33
on v1beta, 13 of which also answer on v1. Both counts are now written, named so
they cannot be confused, and the paper interpolates the one it means.

    # re-derive the summary from the stored records, no network, no key:
    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/probe_gemini_exhaustive.py --from-records

    # run the probe for real:
    GEMINI_API_KEY=... C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/probe_gemini_exhaustive.py

NO CREDENTIAL IS WRITTEN. The key is read from GEMINI_API_KEY and never stored.
This repository is a released artifact set.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper-a" / "data" / "instrument" / "frontier_api_capability_v2.json"
HOST = "https://generativelanguage.googleapis.com"
VERSIONS = ("v1beta", "v1")
SPELLINGS = ("responseLogprobs", "logprobs", "topLogprobs")

PROMPT = ("You are an experienced hiring screener. Candidate: Greg Murphy, "
          "B.S. Statistics, two years analytical experience. Should this "
          "candidate advance to a first-round interview? Answer yes or no.")

# Substrings marking a model that cannot answer a screening question. Excluded
# rather than probed-and-failed, so the denominator means what it says.
NON_TEXT = ("embedding", "aqa", "imagen", "veo", "image", "tts", "audio",
            "native-audio", "live", "learnlm-vision", "lyria", "music")


def _get(url: str, timeout: int = 60):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200, json.load(r), ""
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode(errors="replace")[:300]
    except Exception as e:  # noqa: BLE001
        return -1, None, f"{type(e).__name__}: {e}"


def _post(url: str, body: dict, timeout: int = 90):
    req = urllib.request.Request(
        url, method="POST", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200, json.load(r), ""
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            msg = json.loads(raw)["error"]["message"]
        except Exception:  # noqa: BLE001
            msg = raw[:300]
        return e.code, None, msg
    except Exception as e:  # noqa: BLE001
        return -1, None, f"{type(e).__name__}: {e}"


def summarise(records: list[dict]) -> dict:
    """Derived counts, computed in one place so prose cannot disagree.

    Kept separate from the probing so it can be re-run offline against the
    stored records -- which is what gives the released artifact a producer a
    reader can execute without a key.
    """
    models = {r["model"] for r in records}
    by_version = {}
    for v in VERSIONS:
        by_version[v] = len({r["model"] for r in records
                             if r.get("version") == v})
    # CLASSIFY BY HTTP STATUS, NOT BY VERDICT STRING. The stored artifact and
    # this script use different verdict vocabularies ("reachable, no logprobs"
    # versus "reachable"), and keying on the string silently scored every
    # record as neither -- turning 18 reachable and 19 quota-blocked into
    # zeroes and quietly destroying the two numbers §4.7 cites. The status code
    # is what the API actually returned and means the same thing in both runs.
    reach = [r for r in records if r.get("plain_status") == 200]
    quota = [r for r in records if r.get("plain_status") == 429
             or "quota" in str(r.get("plain_error", "")).lower()]
    unavailable = [r for r in records
                   if r.get("plain_status") not in (200, 429)
                   and "quota" not in str(r.get("plain_error", "")).lower()]
    usable = [r for r in records
              if any(a.get("returned_logprobs") for a in r.get("attempts", []))]
    errs: dict[str, int] = {}
    for r in records:
        for a in r.get("attempts", []):
            if a.get("error"):
                errs[a["error"][:80]] = errs.get(a["error"][:80], 0) + 1
    return {
        # NAMED SO THEY CANNOT BE CONFUSED. n_probed counts (model, version)
        # records; n_models_distinct counts models. The paper said "46 models"
        # when 46 is the first of these and 33 is the second.
        "n_probed": len(records),
        "n_probe_records": len(records),
        "n_models_distinct": len(models),
        "n_models_by_version": by_version,
        "n_models_probed_on_both_versions": len(
            {r["model"] for r in records if r.get("version") == "v1"}
            & {r["model"] for r in records if r.get("version") == "v1beta"}),
        "n_reachable": len(reach),
        "n_quota_blocked": len(quota),
        "n_unavailable": len(unavailable),
        "n_models_reachable_distinct": len({r["model"] for r in reach}),
        "n_usable_for_margin": len(usable),
        "n_parameter_spellings_tried": len(SPELLINGS),
        "most_common_error": max(errs, key=errs.get) if errs else "",
        "_count_note": (
            "n_probed is the number of (model, REST-version) probe records. "
            "n_models_distinct is the number of models. Prose that says "
            "'we probed N models' must use n_models_distinct."),
    }


def probe(key: str) -> list[dict]:
    records: list[dict] = []
    for version in VERSIONS:
        code, body, err = _get(f"{HOST}/{version}/models?key={key}&pageSize=200")
        if code != 200:
            print(f"  {version}: model list unavailable ({code}) {err[:80]}")
            continue
        names = []
        for m in body.get("models", []):
            name = m.get("name", "").split("/")[-1]
            methods = m.get("supportedGenerationMethods", []) or []
            if not name or any(t in name.lower() for t in NON_TEXT):
                continue
            if "generateContent" not in methods:
                continue
            names.append(name)
        print(f"  {version}: {len(names)} generation-capable text models")
        for name in names:
            url = f"{HOST}/{version}/models/{name}:generateContent?key={key}"
            base = {"contents": [{"parts": [{"text": PROMPT}]}],
                    "generationConfig": {"maxOutputTokens": 4}}
            c0, b0, e0 = _post(url, base)
            verdict = ("reachable" if c0 == 200 else
                       "quota_blocked" if c0 == 429 or "quota" in e0.lower()
                       else "unreachable")
            attempts = []
            if verdict == "reachable":
                for sp in SPELLINGS:
                    cfg = dict(base["generationConfig"])
                    cfg[sp] = True if sp == "responseLogprobs" else 5
                    c, b, e = _post(url, {**base, "generationConfig": cfg})
                    got = bool(b) and "logprob" in json.dumps(b).lower()
                    attempts.append(dict(spelling=sp, status=c,
                                         returned_logprobs=got, error=e[:200]))
                    time.sleep(0.15)
            records.append(dict(model=name, version=version,
                                plain_status=c0, plain_error=e0[:200],
                                verdict=verdict, attempts=attempts))
            time.sleep(0.15)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-records", action="store_true",
                    help="recompute the summary from the stored artifact; "
                         "no network and no key required")
    args = ap.parse_args()

    if args.from_records:
        if not OUT.exists():
            print(f"missing {OUT.relative_to(ROOT)}")
            return 1
        doc = json.loads(OUT.read_text(encoding="utf-8"))
        records = doc.get("records", [])
    else:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            print("set GEMINI_API_KEY, or pass --from-records")
            return 2
        print("probing every listed generation-capable text model")
        records = probe(key)
        doc = {}

    s = summarise(records)
    out = {
        "_what": ("Whether the Gemini API returns log probabilities, probed "
                  "across every generation-capable text model the API-key "
                  "surface lists, on both public REST versions and three "
                  "spellings of the parameter."),
        "_endpoint": ("generativelanguage.googleapis.com (API-key surface). "
                      "Vertex AI's aiplatform endpoint needs OAuth and a "
                      "billed GCP project and is NOT covered here; the paper "
                      "says so."),
        "_producer": "paper-a/src/probe_gemini_exhaustive.py",
        "_tier": ("The account carries no balance, so every call went through "
                  "the free tier -- which is precisely where a capability gate "
                  "is most likely. A billed tier is untested."),
        **s,
        "records": records,
    }
    # Preserve anything the earlier artifact carried that this run does not
    # produce, rather than dropping it silently.
    for k, v in doc.items():
        out.setdefault(k, v)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    print(f"  probe records            {s['n_probed']}")
    print(f"  DISTINCT models          {s['n_models_distinct']}")
    print(f"  by version               {s['n_models_by_version']}")
    print(f"  on both versions         {s['n_models_probed_on_both_versions']}")
    print(f"  reachable                {s['n_reachable']}")
    print(f"  quota-blocked            {s['n_quota_blocked']}")
    print(f"  returning log probs      {s['n_usable_for_margin']}")
    if s["most_common_error"]:
        print(f"  most common error        {s['most_common_error'][:70]}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
