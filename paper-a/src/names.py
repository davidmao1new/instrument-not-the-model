"""
Name list construction with matched demographic signal.

The point of this module: race-associated name lists must be MATCHED on
length, syllable count, and frequency, or the demographic signal is
confounded with name familiarity/rarity. Reviewers will raise this.

Primary data source (download yourself, do not commit):
  Rosenman, Olivella & Imai (2023), "Race and ethnicity data for first,
  middle, and last names", Scientific Data. Built from NC voter files.
  Expected at: data/raw/rosenman_firstnames.csv, rosenman_surnames.csv

Secondary/robustness source:
  Tzioumis (2018), "Demographic aspects of first names", Scientific Data.

Historical anchor (hardcoded below, small + Black/White only):
  Bertrand & Mullainathan (2004). Included so results are comparable to
  the field-experiment literature. NOT sufficient on its own.

VERIFY all citations before submission. Do not cite from this file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import pandas as pd

Race = Literal["white", "black", "hispanic", "asian"]
Gender = Literal["male", "female"]

RACES: tuple[Race, ...] = ("white", "black", "hispanic", "asian")
GENDERS: tuple[Gender, ...] = ("male", "female")

# Minimum P(race | name) for a name to be usable as a signal.
# Pre-specified in PROTOCOL.md §8. Do not tune this after seeing results.
PROB_THRESHOLD = 0.80

DATA_RAW = Path("data/raw")
DATA_OUT = Path("data/processed")


# --------------------------------------------------------------------------
# Bertrand & Mullainathan (2004) anchor set.
# Source: "Are Emily and Greg More Employable than Lakisha and Jamal?"
# AER 94(4). Verify spelling against the published appendix before use.
# --------------------------------------------------------------------------
BM2004 = {
    ("white", "female"): [
        "Allison", "Anne", "Carrie", "Emily", "Jill",
        "Laurie", "Kristen", "Meredith", "Sarah",
    ],
    ("white", "male"): [
        "Brad", "Brendan", "Geoffrey", "Greg", "Brett",
        "Jay", "Matthew", "Neil", "Todd",
    ],
    ("black", "female"): [
        "Aisha", "Ebony", "Keisha", "Kenya", "Latonya",
        "Lakisha", "Latoya", "Tamika", "Tanisha",
    ],
    ("black", "male"): [
        "Darnell", "Hakim", "Jermaine", "Kareem", "Jamal",
        "Leroy", "Rasheed", "Tremayne", "Tyrone",
    ],
}

BM2004_SURNAMES = {
    "white": ["Baker", "Kelly", "McCarthy", "Murphy", "Murray",
              "O'Brien", "Ryan", "Sullivan", "Walsh"],
    "black": ["Jackson", "Jones", "Robinson", "Washington", "Williams"],
}


@dataclass(frozen=True)
class NameRecord:
    first: str
    last: str
    race: str
    gender: str
    p_race_first: float
    p_race_last: float
    n_chars: int
    n_syllables: int
    freq_pctile: float
    source: str

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"


_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.I)


def count_syllables(name: str) -> int:
    """Crude vowel-group syllable heuristic.

    Good enough for MATCHING (we only need comparability across lists),
    not good enough to report as linguistic fact. Documented as a
    heuristic in the paper's methods section.
    """
    groups = _VOWEL_GROUP.findall(name)
    n = len(groups)
    if name.lower().endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def load_rosenman_firstnames() -> pd.DataFrame:
    """Load first-name race probabilities.

    Expected columns (rename in the mapping below if the release differs):
      name, pctwhite, pctblack, pcthispanic, pctapi
    Percentages may be 0-100 or 0-1 depending on release; normalized here.
    """
    path = DATA_RAW / "rosenman_firstnames.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Rosenman et al. (2023) Scientific Data "
            "release and place it there. See PROTOCOL.md section 8."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    colmap = {
        "pctwhite": "white", "pctblack": "black",
        "pcthispanic": "hispanic", "pctapi": "asian",
    }
    missing = [c for c in colmap if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {list(colmap)} in {path}; missing {missing}. "
            f"Found: {list(df.columns)}. Adjust colmap for your release."
        )
    df = df.rename(columns=colmap)

    for r in RACES:
        if df[r].max() > 1.0001:
            df[r] = df[r] / 100.0
    return df


def _match_on_shape(
    candidates: dict[str, pd.DataFrame],
    per_cell: int,
) -> dict[str, pd.DataFrame]:
    """Select `per_cell` names per race, matched on length and syllables.

    Strategy: build the (n_chars, n_syllables) strata that every race can
    populate, then draw evenly from the shared strata. This guarantees the
    marginal distribution of name shape is identical across races, which
    kills the familiarity/rarity confound.
    """
    for r, df in candidates.items():
        df["n_chars"] = df["name"].str.len()
        df["n_syllables"] = df["name"].map(count_syllables)

    shared = None
    for df in candidates.values():
        strata = set(zip(df["n_chars"], df["n_syllables"]))
        shared = strata if shared is None else (shared & strata)
    if not shared:
        raise ValueError(
            "No shared (n_chars, n_syllables) strata across races. "
            "Lower PROB_THRESHOLD or widen the candidate pool."
        )

    shared_sorted = sorted(shared)

    # Build a single allocation plan shared by every race BEFORE drawing any
    # names: for each stratum, how many names each race will contribute.
    #
    # This is the part that makes the guarantee real. An earlier implementation
    # cycled through strata per race independently, advancing its cursor whether
    # or not a draw succeeded. When one race had fewer names available in a
    # stratum than another, the races silently ended up with DIFFERENT marginal
    # shape distributions -- for example [(5,1),(5,1),(5,2)] against
    # [(5,1),(5,2),(5,2)]. That is precisely the length/familiarity confound
    # PROTOCOL.md section 8 exists to eliminate, and Wilson & Caliskan
    # (ledger A-24) show that name-shape and frequency artifacts can reverse the
    # direction of a measured effect. Caught by
    # tests/test_names.py::test_matched_lists_share_shape_distribution.
    capacity = {
        stratum: min(
            int(((df["n_chars"] == stratum[0]) & (df["n_syllables"] == stratum[1])).sum())
            for df in candidates.values()
        )
        for stratum in shared_sorted
    }

    plan: dict[tuple[int, int], int] = {s: 0 for s in shared_sorted}
    remaining = per_cell
    # Round-robin over strata so the draw is spread across shapes rather than
    # exhausting one stratum first, which would narrow the shape distribution.
    while remaining > 0:
        progressed = False
        for stratum in shared_sorted:
            if remaining == 0:
                break
            if plan[stratum] < capacity[stratum]:
                plan[stratum] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    drawn = sum(plan.values())
    if drawn < per_cell:
        raise ValueError(
            f"Only {drawn} names per race can be matched on shape, but per_cell="
            f"{per_cell} was requested. Shared strata capacities: "
            f"{ {s: c for s, c in capacity.items() if c} }. "
            "Lower per_cell, lower PROB_THRESHOLD, or widen the candidate pool. "
            "Do NOT proceed with unequal lists: unmatched name shape confounds "
            "the race signal with familiarity and rarity (PROTOCOL.md section 8)."
        )

    out: dict[str, pd.DataFrame] = {}
    for r, df in candidates.items():
        picks: list[dict] = []
        for stratum, k in plan.items():
            if k == 0:
                continue
            pool = df[
                (df["n_chars"] == stratum[0]) & (df["n_syllables"] == stratum[1])
            ].sort_values("name")  # deterministic; no RNG in the name set
            picks.extend(pool.iloc[:k].to_dict("records"))
        out[r] = pd.DataFrame(picks)
    return out


def build_matched_first_names(per_cell: int = 12) -> pd.DataFrame:
    """Build a matched first-name set across race x gender.

    NOTE ON GENDER: the Rosenman first-name release does not carry a gender
    field. You must join a gender-frequency source (SSA baby-name data is the
    standard choice and is public) and pre-specify a P(gender|name) threshold
    the same way you did for race. This function raises until you wire that in,
    deliberately, so gender assignment never happens by guesswork.
    """
    raise NotImplementedError(
        "Wire in a gender source before building the name set.\n"
        "Recommended: SSA national baby-name data, threshold P(gender|name) >= 0.90.\n"
        "Then: filter df by race threshold, join gender, call _match_on_shape "
        "per (race, gender) cell.\n"
        "This is intentionally unimplemented so gender is never assigned by "
        "intuition. See PROTOCOL.md section 8."
    )


def bm2004_records() -> list[NameRecord]:
    """The Bertrand & Mullainathan anchor set as NameRecords.

    Uses their surname lists where available. Probabilities are recorded as
    NaN because BM did not publish P(race|name); they curated by construction.
    Report this honestly in the paper.
    """
    recs: list[NameRecord] = []
    for (race, gender), firsts in BM2004.items():
        lasts = BM2004_SURNAMES.get(race, ["Smith"])
        for i, first in enumerate(firsts):
            last = lasts[i % len(lasts)]
            recs.append(
                NameRecord(
                    first=first, last=last, race=race, gender=gender,
                    p_race_first=float("nan"), p_race_last=float("nan"),
                    n_chars=len(first), n_syllables=count_syllables(first),
                    freq_pctile=float("nan"), source="bertrand_mullainathan_2004",
                )
            )
    return recs


def write_name_set(records: list[NameRecord], filename: str) -> Path:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    out = DATA_OUT / filename
    pd.DataFrame([asdict(r) for r in records]).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    anchor = bm2004_records()
    p = write_name_set(anchor, "names_bm2004_anchor.csv")
    print(f"Wrote {len(anchor)} anchor names -> {p}")
    print("Next: download Rosenman + SSA data, then implement "
          "build_matched_first_names(). See PROTOCOL.md section 8.")
