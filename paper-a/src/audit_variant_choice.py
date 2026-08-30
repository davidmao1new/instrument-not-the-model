r"""Which VARIANT of a quantity the paper reads, and whether anyone chose it.

THE DEFECT THIS EXISTS FOR. Every number in this paper is interpolated from
a released artifact, and check_iclr.py's typed-measurement gate enforces
that: a numeral typed into the prose fails the build. That gate answers
"did this number come from an artifact". It does not answer "did it come
from the RIGHT FIELD of that artifact", and several artifacts here carry the
same quantity twice -- once as measured and once after a correction, or once
over all units and once over the subset a stated rule admits.

Reading the wrong sibling produces a number that is fully provenanced, passes
every existing gate, and contradicts what the paper says about it. That
happened twice, and both were found by reading, not by a gate:

  * frontier_noise_floor.json carries published_ratio_sd_to_effect beside
    ratio_sd_to_effect_corrected. The headline dispersion range was built
    from the published values while the paper's own section on the noise
    floor printed the corrected ones, so one checkpoint appeared as 0.64 in
    the abstract and 0.55 six pages later. In a paper whose thesis is that
    undisclosed analysis choices move headline numbers, the undisclosed
    choice was on the headline number.

  * reporting_scale.json carries jacobian_error_min/max beside
    predicted_for_distinguishable. Here the unfiltered read was CORRECT --
    the artifact states that jacobian_error is independent of the effect
    size, so the separability rule that governs its sibling does not apply --
    but nothing recorded that, and the paper had a sentence that read as
    though it required the filtered form. The risk is identical whichever
    way the choice went: two siblings, one read, no record of why.

WHAT THIS CHECKS. For every artifact the builders read, it finds families of
keys that are variants of one quantity, works out which member the builder
source actually names, and requires a recorded decision saying which variant
the paper uses and why. An unrecorded choice is a finding. A recorded choice
that no longer matches anything is also a finding, because a stale record is
how a register stops describing the thing it registers.

MEMBERSHIP IS DECIDED FROM THE BUILDER SOURCE, NOT FROM VALUES. Matching a
macro's value against artifact values would be value-coincidence matching:
0.55 appears in a dozen artifacts and two of them mean different things. The
builders name the key they read, so that name is the evidence.

    .venv/Scripts/python.exe paper-a/src/audit_variant_choice.py
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
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"

# The builders whose source names the key that is read. Any file that
# interpolates a number into prose belongs here.
BUILDERS = ("build_paper_v3.py", "build_iclr.py", "build_facct_tex.py",
            "build_reporting_matrix.py", "build_methods_supplement.py")

# A quantity written twice, once each side of a correction. Stripping any of
# these from a key name leaves the stem two siblings share.
CORRECTION_MARKERS = (
    ("prefix", "published_"), ("prefix", "raw_"), ("prefix", "uncorrected_"),
    ("prefix", "observed_"), ("prefix", "nominal_"), ("prefix", "measured_"),
    ("suffix", "_published"), ("suffix", "_corrected"),
    ("suffix", "_uncorrected"), ("suffix", "_raw"), ("suffix", "_adjusted"),
    ("suffix", "_debiased"), ("suffix", "_net"),
)

# A quantity written twice, once over everything and once over the subset
# some stated rule admits. These do not share a stem with their sibling, so
# they are recognised by the restriction they name rather than by grouping.
SUBSET_MARKERS = re.compile(
    r"_for_|_excluding|_where_|_restricted|_only_|_among_", re.I)

# The clause that states the restriction. Removing it leaves the stem the
# restricted key would share with its unrestricted sibling.
RESTRICTION_CLAUSE = re.compile(
    r"(?:_for_|_excluding_|_where_|_restricted_to_|_only_|_among_).*$", re.I)


def brackets(value, holder: dict, src: str) -> str | None:
    """A read min/max pair in this block that contains a two-element range.

    Corroboration only, for a key whose NAME already declares a restriction.
    Two numbers matching by value alone is coincidence; a declared subset
    whose range sits inside a read min/max pair in the same block is not.
    """
    if not (isinstance(value, list) and len(value) == 2
            and all(isinstance(v, (int, float)) for v in value)):
        return None
    lo_k = [k for k in holder
            if k.endswith("_min") and named(src, k)
            and isinstance(holder[k], (int, float))]
    hi_k = [k for k in holder
            if k.endswith("_max") and named(src, k)
            and isinstance(holder[k], (int, float))]
    for a in lo_k:
        for b in hi_k:
            if a[:-4] != b[:-4]:
                continue
            if holder[a] <= min(value) and max(value) <= holder[b]:
                return f"{a}/{b}"
    return None


# The words an entry uses to name a form, and the key marker each implies.
FORM_WORDS = {
    "corrected": ("_corrected",),
    "adjusted": ("_adjusted",),
    "debiased": ("_debiased",),
    "published": ("published_", "_published"),
    "raw": ("raw_", "_raw"),
    "uncorrected": ("uncorrected_", "_uncorrected"),
    "observed": ("observed_",),
    "measured": ("measured_",),
}


def marker_of(key: str) -> str | None:
    """The form word a key name implies, if any."""
    for word, marks in FORM_WORDS.items():
        for m in marks:
            if (m.startswith("_") and key.endswith(m)) or \
               (not m.startswith("_") and key.startswith(m)):
                return word
    return None


def forms_named(entry) -> set[str]:
    """The form words an entry claims, read from its first field."""
    text = (entry[0] if isinstance(entry, (tuple, list)) and entry
            else str(entry)).lower()
    return {w for w in FORM_WORDS if w in text}


def stem(key: str) -> str | None:
    """The shared name under a correction marker, or None if unmarked."""
    for kind, mark in CORRECTION_MARKERS:
        if kind == "prefix" and key.startswith(mark):
            return key[len(mark):]
        if kind == "suffix" and key.endswith(mark):
            return key[:-len(mark)]
    return None


def walk(node, path=()):
    """Every dict in the artifact, with the path that reached it."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, path + (f"[{i}]",))


def families(doc) -> dict[str, set[str]]:
    """Variant families in one artifact: stem -> the key names that share it."""
    out: dict[str, set[str]] = {}
    for _path, d in walk(doc):
        keys = list(d)
        for k in keys:
            s = stem(k)
            if s is None:
                continue
            # The bare stem, or another marked spelling of it, must also be
            # present in the same dict for this to be a family rather than a
            # lone name that happens to carry a word like "raw".
            siblings = {o for o in keys
                        if o != k and (o == s or stem(o) == s)}
            if siblings:
                out.setdefault(s, set()).update({k} | siblings)
    return out


def builder_text() -> str:
    parts = []
    for name in BUILDERS:
        p = SRC / name
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def named(src: str, key: str) -> bool:
    """Does a builder actually name this key?"""
    return re.search(r"[\"']" + re.escape(key) + r"[\"']", src) is not None


def load_registry() -> dict:
    p = SRC / "variant_choices.py"
    if not p.is_file():
        return {}
    ns: dict = {}
    exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)  # noqa: S102
    return ns.get("CHOICES", {})


def main() -> int:
    print("=" * 78)
    print("VARIANT-CHOICE AUDIT  --  which form of a quantity the paper reads")
    print("=" * 78)

    if not DATA.is_dir():
        print("  no data tree on this checkout")
        return 0

    src = builder_text()
    if not src.strip():
        print("  REFUSING: no builder source found, so no key could be shown "
              "to be read. This audit must not pass by being unable to look.")
        return 1

    registry = load_registry()
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    n_art = n_fam = 0

    for art in sorted(DATA.rglob("*.json")):
        try:
            doc = json.loads(art.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        n_art += 1
        rel = art.name

        for s, members in sorted(families(doc).items()):
            read = sorted(m for m in members if named(src, m))
            if not read:
                continue          # the paper does not read this family
            n_fam += 1
            seen.add((rel, s))
            if (rel, s) not in registry:
                problems.append(
                    f"{rel}: the builders read {read[0]!r} from a family of "
                    f"{len(members)} variants ({', '.join(sorted(members))}) "
                    f"with no recorded reason. Add ({rel!r}, {s!r}) to "
                    "variant_choices.py naming the variant and why.")
                continue
            # THE RECORD IS CHECKED AGAINST THE CODE. Without this the entry
            # is a paragraph beside a read that can flip under it, which is
            # the defect this whole audit exists for, one level up.
            claimed = forms_named(registry[(rel, s)])
            if not claimed:
                continue          # an entry that names no form claims nothing
            actual = {m for m in (marker_of(r) for r in read) if m}
            rogue = actual - claimed
            if rogue:
                problems.append(
                    f"{rel}: variant_choices.py records {s!r} as "
                    f"{'/'.join(sorted(claimed))}, but the builders read "
                    f"{', '.join(sorted(r for r in read if marker_of(r) in rogue))}"
                    f" -- a {'/'.join(sorted(rogue))} form. Either the record "
                    "or the read is wrong.")

        for _path, holder in walk(doc):
            for k in sorted(holder):
                if not SUBSET_MARKERS.search(k) or named(src, k):
                    continue
                # (a) the stem it would share with an unrestricted sibling
                st = RESTRICTION_CLAUSE.sub("", k)
                unrestricted = (st if st != k and st in holder
                                and named(src, st) else None)
                # (b) or a read min/max pair in this block that contains it
                if unrestricted is None:
                    unrestricted = brackets(holder[k], holder, src)
                if unrestricted is None:
                    continue
                n_fam += 1
                seen.add((rel, k))
                if (rel, k) in registry:
                    continue
                problems.append(
                    f"{rel}: {k!r} restricts a quantity the builders read "
                    f"unrestricted as {unrestricted!r}, with no recorded "
                    f"reason. Add ({rel!r}, {k!r}) to variant_choices.py "
                    "naming which form the paper uses and why, or read the "
                    "restricted form.")

    stale = [k for k in registry if k not in seen]
    for rel, s in sorted(stale):
        problems.append(
            f"{rel}: variant_choices.py records a decision for {s!r} that no "
            "longer matches any family the builders read -- a stale record "
            "stops describing what it registers.")

    print(f"  {n_art} artifact(s) scanned, {n_fam} variant choice(s) reached "
          f"by the builders, {len(registry)} recorded")
    if problems:
        print("-" * 78)
        for p in problems:
            print(f"  {p}")
        print("=" * 78)
        print(f"{len(problems)} unrecorded or stale variant choice(s)")
        return 1
    print("  every variant the paper reads has a recorded reason")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
