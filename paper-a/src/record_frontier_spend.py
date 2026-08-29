"""What the frontier arm actually cost, parsed from the run's own output.

WHY THIS IS A SCRIPT AND NOT A SENTENCE. §4.7 says the frontier run cost less
than a dollar, and that claim is doing work: the paper's argument is that the
barrier to auditing frontier models on a graded outcome is the vendor's product
surface, not the researcher's budget. A reader is entitled to the number, and
this paper's rule is that no number is typed.

`experiment_frontier_margin.py` prints its MEASURED input-token usage per model
-- the `usage.prompt_tokens` the API returned, summed, not an estimate from
character counts -- along with the list price the spend guard used. This reads
those lines back and writes them as an artifact.

    C:/research-toolchain/venv/Scripts/python.exe \
        paper-a/src/record_frontier_spend.py <run-log> [<run-log> ...]
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper-a" / "data" / "frontier" / "frontier_spend.json"

MODEL = re.compile(r"^===\s*(\S+)\s*===")
SPEND = re.compile(r"input tokens ([\d,]+)\s+estimated spend "
                   r"\$([\d.]+) at \$([\d.]+)/M")


def main() -> int:
    logs = [pathlib.Path(a) for a in sys.argv[1:]]
    if not logs:
        sys.exit("give at least one run log")

    models, cur = {}, None
    for log in logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = MODEL.match(line.strip())
            if m:
                cur = m.group(1)
                continue
            s = SPEND.search(line)
            if s and cur:
                models[cur] = dict(
                    input_tokens=int(s.group(1).replace(",", "")),
                    usd=float(s.group(2)),
                    usd_per_million_input=float(s.group(3)))
                cur = None

    if not models:
        sys.exit("no spend lines found; was the run log the right file?")

    # ------------------------------------------------------------------
    # THE COUNTER IS PER-INVOCATION, AND THE RUNNER RESUMES.
    # experiment_frontier_margin.py rebuilds `done` from the existing JSONL and
    # `continue`s past recorded cells, accumulating in_tok only for calls it
    # actually issues. A model whose file was partly written in an earlier
    # session therefore prints a token count for the LAST invocation, not for
    # the run -- and the paper quoted the sum as the cost of the whole arm.
    #
    # It is detectable because every model sends the SAME prompts and all four
    # checkpoints share the o200k_base tokenizer, so the counts should agree
    # exactly. Three of four report 134,532 and one reports 77,333: that one
    # resumed. The modal count is the per-model truth, and a model below it is
    # corrected up to it rather than left understating the bill.
    counts = [v["input_tokens"] for v in models.values()]
    modal = max(set(counts), key=counts.count) if counts else 0
    n_modal = counts.count(modal)
    resumed = []
    for name, v in models.items():
        v["input_tokens_recorded"] = v["input_tokens"]
        v["usd_recorded"] = v["usd"]
        if n_modal >= 2 and v["input_tokens"] < modal:
            resumed.append(name)
            v["resumed_mid_run"] = True
            v["input_tokens"] = modal
            v["usd"] = round(modal / 1e6 * v["usd_per_million_input"], 4)
        else:
            v["resumed_mid_run"] = False

    total = sum(v["usd"] for v in models.values())
    out = {
        "_what": "Measured spend for the frontier arm, parsed from the run's "
                 "own printed usage.",
        "_source": "usage.prompt_tokens returned by the API, summed per model "
                   "by experiment_frontier_margin.py; output is one token per "
                   "call and is not counted.",
        "_logs": [str(p.name) for p in logs],
        "_caveat": "List price at the time of the run. Prices change; the "
                   "token counts do not.",
        "_resumption_note": (
            "The runner resumes: it skips cells already in the JSONL and "
            "counts tokens only for calls it issues, so a model whose file was "
            "partly written earlier prints the last invocation's usage rather "
            "than the run's. Every model sends the same prompts on the same "
            "tokenizer, so the counts must agree; any model below the modal "
            "count is corrected up to it. input_tokens_recorded and "
            "usd_recorded preserve what the log actually said."),
        "models": models,
        "total_usd": total,
        "total_input_tokens": sum(v["input_tokens"] for v in models.values()),
        "total_cents": int(round(total * 100)),
        "n_models": len(models),
        "modal_input_tokens_per_model": modal,
        "n_models_resumed_mid_run": len(resumed),
        "models_resumed_mid_run": resumed,
        "total_input_tokens_as_logged": sum(
            v["input_tokens_recorded"] for v in models.values()),
        "total_usd_as_logged": round(
            sum(v["usd_recorded"] for v in models.values()), 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for m, v in models.items():
        print(f"  {m:<16}{v['input_tokens']:>10,} tok   ${v['usd']:.4f}")
    print(f"  {'TOTAL':<16}{out['total_input_tokens']:>10,} tok   ${total:.4f} "
          f"({out['total_cents']} cents)")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
