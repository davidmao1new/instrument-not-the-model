"""Does each mechanism condition do what its docstring says, in token space?

An adversarial audit raised three objections to Study 3 that cannot be settled
by reading the code, only by tokenising the actual strings with the actual
model's tokenizer:

  A. D7 IS SUPPOSED TO BE THE DECISIVE CONTROL. It substitutes the paragraph
     break with a triple newline, and the whole of contrast C2 rests on the
     claim that this leaves the structural delimiter INTACT while making an
     edit of the same size. But the delimiter H_delim actually names is the
     MERGED token spelling a sentence-final period followed by a paragraph
     break. If '.\\n\\n\\n' does not tokenise with that merged token still
     present, then D7 destroys the delimiter too, C2 is destruction versus
     destruction, and Study 3's decisive contrast means nothing.

  B. SEQUENCE LENGTH may be confounded with delimiter destruction. If the
     destruction conditions are systematically longer than the controls, then
     "delimiter destroyed" and "prompt got longer" are the same variable.

  C. THE NAME'S TOKEN INDEX may shift under the destruction conditions and not
     under the controls, which would make the effect a position effect on the
     name rather than a delimiter effect.

There is also a fourth question the audit raised about the tokenizers
themselves: whether every model in the panel even HAS a merged period-break
token. If some do not, `n_delims_destroyed` is not a meaningful dose for those
models and the dose-response cannot be pooled across the panel.

This probe answers all four against a live server, per model. It reports what it
finds and draws no conclusion; the conclusions belong in the analysis.

WHY THE ARTIFACT NOW CARRIES A MEASURED DELIMITER COLUMN AS WELL AS THE
DECLARED ONE. The only delimiter number this file used to export per condition
was `n_delims_destroyed`, and that field was never a measurement: it was copied
straight out of `CONDITIONS[cond][1]`, the experimenter's own declaration of
what the edit was supposed to do. The paper's Table 2 reads that field, so the
column headed "delimiters destroyed" certified the design against the intent
that built it rather than against the tokenizer. The probe did measure the real
quantity -- `n_merged_delims`, the count of the merged period-plus-break token
actually surviving in the token stream -- and printed it, and then no artifact
consumer ever read it.

That mattered in exactly one place, which is the place it could least afford to
matter. D7 is declared to destroy nothing; measured, it removes BOTH merged
delimiter tokens, because '.\\n\\n\\n' is its own single vocabulary item and not
the one H_delim names. Objection A above anticipated this and the ANSWERS block
already printed it. The design intent survives -- D7 substitutes one merged
delimiter for another rather than fragmenting it, which is what it was built to
do -- but "destroyed 0" and "destroyed 2" are different table entries, and a
reader deciding which conditions are the null controls needs the measured one.

So every condition now carries both numbers plus the disposition that
reconciles them (intact / fragmented / substituted), and the declared field is
kept untouched beside them so the superseded column can still be reported.

    .venv/Scripts/python.exe paper-a/src/probe_condition_tokens.py --model-label <id>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402
from experiment_mechanism import CONDITIONS, build  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "instrument"


def tokenize(api: str, text: str):
    r = requests.post(f"{api}/tokenize", json={"content": text, "with_pieces": True},
                      timeout=120)
    r.raise_for_status()
    j = r.json()
    toks = j.get("tokens", [])
    if toks and isinstance(toks[0], dict):
        return [(t["id"], t.get("piece", "")) for t in toks]
    # server built without with_pieces: fall back to ids, then detokenise each
    out = []
    for tid in toks:
        d = requests.post(f"{api}/detokenize", json={"tokens": [tid]}, timeout=60)
        out.append((tid, d.json().get("content", "")))
    return out


def count_adjacent(ids: list[int], pair) -> int:
    """Occurrences of a two-token sequence, used to count FRAGMENTED boundaries.

    A destroyed delimiter is not merely an absent token; on this vocabulary it
    is '.\\n' followed by ' \\n'. Counting the pair rather than either member
    alone keeps ordinary sentence-final '.\\n' tokens elsewhere in the prompt
    from being miscounted as damage.
    """
    if not pair:
        return 0
    a, b = pair
    return sum(1 for k in range(len(ids) - 1) if ids[k] == a and ids[k + 1] == b)


def disposition(destroyed: int, n_frag: int, n_alt: int) -> str:
    """What happened to the merged delimiters, in one word.

    The paper's C2 contrast turns on the difference between a boundary that was
    FRAGMENTED into two tokens and one that was SUBSTITUTED by a different
    single delimiter token. Both remove the token H_delim names, so a bare
    "destroyed" count cannot tell them apart, and reporting only the count would
    make D6 and D7 look identical when the whole point of D7 is that they are
    not.
    """
    if destroyed == 0:
        return "intact"
    if n_frag == destroyed and n_alt == 0:
        return "fragmented"
    if n_alt == destroyed and n_frag == 0:
        return "substituted"
    return "mixed"


def norm(piece) -> str:
    if isinstance(piece, list):
        try:
            return bytes(piece).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return str(piece)
    return str(piece).replace("▁", " ").replace("Ċ", "\n").replace("Ġ", " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    api = f"http://127.0.0.1:{args.port}"
    served = st.assert_serving(args.port, args.model_label)
    print(f"  [guard] port {args.port} serving {served}\n", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    name = st.NAME_GRID[0]["white"]
    body = st.TEMPLATES["T1_strong"]

    # --- does this tokenizer have the tokens the hypothesis names? ----------
    probes = {"\\n\\n": "\n\n", "\\n \\n": "\n \n", "\\n\\n\\n": "\n\n\n",
              ".\\n\\n": ".\n\n", ".\\n\\n\\n": ".\n\n\n", ".\\n \\n": ".\n \n",
              ".\\n": ".\n", "\\n": "\n"}
    print("TOKENISATION OF THE DELIMITER STRINGS THEMSELVES")
    print(f"  {'string':<12}{'n tokens':>9}   pieces")
    vocab = {}
    for label, s in probes.items():
        t = tokenize(api, s)
        # a leading BOS is added by some servers; drop a first token that
        # detokenises to empty or to the BOS marker
        if t and norm(t[0][1]).strip() in ("", "<s>", "<|begin_of_text|>"):
            t = t[1:]
        vocab[label] = dict(n=len(t), pieces=[norm(p) for _, p in t],
                            ids=[i for i, _ in t])
        print(f"  {label:<12}{len(t):>9}   {[norm(p) for _, p in t]}")

    merged_exists = vocab[".\\n\\n"]["n"] == 1
    print(f"\n  merged period-plus-break is a SINGLE token: {merged_exists}")

    # --- per-condition token analysis --------------------------------------
    print("\nPER-CONDITION TOKEN ANALYSIS")
    base_toks = tokenize(api, build("D0", name, body))
    base_ids = [i for i, _ in base_toks]
    merged_id = vocab[".\\n\\n"]["ids"][0] if merged_exists else None
    # The other two shapes a paragraph boundary takes in this vocabulary, read
    # off the tokenizer rather than assumed: the alternative merged delimiter
    # D7 substitutes in, and the two-token fragment D4/D5/D6 split the boundary
    # into. Both are None when this tokenizer does not produce them, and the
    # counts below then fall through to zero rather than guessing.
    alt_merged_id = (vocab[".\\n\\n\\n"]["ids"][0]
                     if vocab[".\\n\\n\\n"]["n"] == 1 else None)
    frag_ids = (tuple(vocab[".\\n \\n"]["ids"])
                if vocab[".\\n \\n"]["n"] == 2 else None)

    rows = {}
    print(f"  {'cond':<5}{'n_tok':>7}{'d_len':>7}{'merged left':>13}"
          f"{'name idx':>10}{'d_name':>8}   note")
    for cond in CONDITIONS:
        toks = tokenize(api, build(cond, name, body))
        ids = [i for i, _ in toks]
        pieces = [norm(p) for _, p in toks]
        n_merged = sum(1 for i in ids if merged_id is not None and i == merged_id)
        n_alt = sum(1 for i in ids if alt_merged_id is not None and i == alt_merged_id)
        n_frag = count_adjacent(ids, frag_ids)
        # index of the first token whose piece contains the given name's first word
        first_word = name.split()[0]
        nidx = next((k for k, p in enumerate(pieces)
                     if first_word.startswith(p.strip()) and p.strip()), None)
        for k, p in enumerate(pieces):
            if p.strip() and p.strip() in first_word and len(p.strip()) > 1:
                nidx = k
                break
        rows[cond] = dict(n_tokens=len(ids), delta_len=len(ids) - len(base_ids),
                          n_merged_delims=n_merged,
                          n_alt_merged_delims=n_alt,
                          n_frag_delims=n_frag, name_token_index=nidx,
                          delta_name_index=(nidx - rows["D0"]["name_token_index"])
                          if cond != "D0" and rows.get("D0") else 0,
                          # KEPT, NOT REPLACED. This is the experimenter's
                          # declaration, the field Table 2 has always reported.
                          # The measured column is added beside it below so the
                          # paper can report both and say which is which.
                          n_delims_destroyed=CONDITIONS[cond][1],
                          n_delims_destroyed_declared=CONDITIONS[cond][1],
                          note=CONDITIONS[cond][0])
        r = rows[cond]
        print(f"  {cond:<5}{r['n_tokens']:>7}{r['delta_len']:>+7}"
              f"{n_merged:>13}{str(nidx):>10}{r['delta_name_index']:>+8}   "
              f"{CONDITIONS[cond][0][:38]}")

    # --- declared versus measured, reconciled -------------------------------
    # The measured dose is defined against the BASELINE's own delimiter count,
    # not against a constant, because "how many of the merged delimiters this
    # prompt actually had did this edit remove" is the quantity H_delim is about
    # and it is the only definition that stays meaningful on a tokenizer that
    # merges differently or not at all.
    base_merged = rows["D0"]["n_merged_delims"]
    disagree = {}
    for cond, r in rows.items():
        destroyed = base_merged - r["n_merged_delims"] if merged_exists else None
        r["n_merged_delims_baseline"] = base_merged
        r["n_delims_destroyed_measured"] = destroyed
        r["n_delims_fragmented_measured"] = r["n_frag_delims"]
        r["n_delims_substituted_measured"] = r["n_alt_merged_delims"]
        r["delimiter_disposition"] = (
            disposition(destroyed, r["n_frag_delims"], r["n_alt_merged_delims"])
            if merged_exists else "no_merged_token_in_vocabulary")
        # Undecidable rather than False when the vocabulary has no merged token:
        # there is nothing for the declared dose to be checked against, and
        # calling that a disagreement would invent eleven of them.
        r["declared_matches_measured"] = (
            (destroyed == r["n_delims_destroyed_declared"]) if merged_exists else None)
        if r["declared_matches_measured"] is False:
            disagree[cond] = dict(declared=r["n_delims_destroyed_declared"],
                                  measured=destroyed,
                                  disposition=r["delimiter_disposition"],
                                  note=r["note"])

    # Which conditions are the null controls is a CONSEQUENCE of the column
    # above, not an independent fact, so both versions of the answer are derived
    # here rather than being asserted anywhere downstream.
    controls_declared = sorted(c for c, r in rows.items()
                               if r["n_delims_destroyed_declared"] == 0 and c != "D0")
    controls_measured = sorted(c for c, r in rows.items()
                               if r["n_delims_destroyed_measured"] == 0 and c != "D0")
    delimiter_audit = dict(
        merged_delim_id=merged_id, alt_merged_delim_id=alt_merged_id,
        frag_pair_ids=list(frag_ids) if frag_ids else None,
        baseline_merged_delims=base_merged,
        measured_definition=("n_delims_destroyed_measured = "
                             "D0's count of the merged '.\\n\\n' token minus "
                             "this condition's count of it, in the tokenised "
                             "prompt"),
        declared_source="experiment_mechanism.CONDITIONS[cond][1]",
        n_conditions=len(rows), n_disagreements=len(disagree),
        disagreements=disagree,
        null_controls_declared=controls_declared,
        null_controls_measured=controls_measured,
        reclassified=sorted(set(controls_declared) ^ set(controls_measured)))

    print("\nDECLARED VERSUS MEASURED DELIMITER DOSE")
    print(f"  {'cond':<5}{'declared':>10}{'measured':>10}{'frag':>6}{'subst':>7}"
          f"   {'disposition':<13}agree")
    for cond, r in rows.items():
        print(f"  {cond:<5}{r['n_delims_destroyed_declared']:>10}"
              f"{r['n_delims_destroyed_measured']:>10}"
              f"{r['n_delims_fragmented_measured']:>6}"
              f"{r['n_delims_substituted_measured']:>7}   "
              f"{r['delimiter_disposition']:<13}"
              f"{'yes' if r['declared_matches_measured'] else 'NO'}")
    print(f"  disagreements: {sorted(disagree) or 'none'}")
    print(f"  null controls, declared: {controls_declared}")
    print(f"  null controls, measured: {controls_measured}")

    # --- the three questions, answered -------------------------------------
    print("\nANSWERS")
    if merged_exists:
        b = rows["D0"]["n_merged_delims"]
        print(f"  A. merged delimiters surviving:  D0={b}  "
              f"D6(destroy both)={rows['D6']['n_merged_delims']}  "
              f"D7(substitute)={rows['D7']['n_merged_delims']}")
        if rows["D7"]["n_merged_delims"] == b:
            print("     -> D7 LEAVES THE MERGED DELIMITER INTACT. C2 is a valid "
                  "destruction-vs-substitution contrast.")
        else:
            print("     -> D7 ALSO DESTROYS THE MERGED DELIMITER. C2 is NOT "
                  "destruction vs substitution and the paper must say so.")
    else:
        print("  A. this tokenizer has no merged period-plus-break token, so "
              "H_delim as stated does not apply to it")
    ctrl = [rows[c]["delta_len"] for c in ("D1", "D2", "D3")]
    dest = [rows[c]["delta_len"] for c in ("D4", "D5", "D6")]
    print(f"  B. sequence-length change: controls {ctrl}, destruction {dest}, "
          f"substitution [{rows['D7']['delta_len']}]")
    print(f"     -> length is {'CONFOUNDED with' if set(ctrl).isdisjoint(dest) else 'separable from'} destruction")
    nm = {c: rows[c]["delta_name_index"] for c in CONDITIONS}
    print(f"  C. name token index shift: {nm}")

    path = OUT_DIR / f"condition_tokens_{args.model_label}.json"
    path.write_text(json.dumps(dict(model=args.model_label, served=served,
                                    delimiter_vocab=vocab,
                                    merged_token_exists=merged_exists,
                                    delimiter_audit=delimiter_audit,
                                    conditions=rows), indent=2),
                    encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
