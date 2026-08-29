"""Config-to-artifact integrity.

Why this test exists: `config.yaml` is the file the paper's reproducibility
statement will point at. A wrong SHA256 there is worse than no SHA256, because it
looks like provenance while providing none. This test was written immediately
after a near-miss in which truncated hashes were pasted into `config.yaml` and
padded out to look like full ones.

Every hash and byte count in `config.yaml` must match `models/MANIFEST.tsv`,
which is written only by the download script from the bytes actually on disk.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper-a" / "config.yaml"
MANIFEST = ROOT / "models" / "MANIFEST.tsv"

SHA256_LEN = 64


def load_manifest() -> dict[str, tuple[int, str]]:
    rows: dict[str, tuple[int, str]] = {}
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 5:
            rows[parts[0]] = (int(parts[3]), parts[4])
    return rows


def config_entries() -> list[dict]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return list(cfg["models"]) + list(cfg["robustness_quantization"])


@pytest.fixture(scope="module")
def manifest() -> dict[str, tuple[int, str]]:
    if not MANIFEST.exists():
        pytest.skip("models/MANIFEST.tsv absent; run scripts/fetch_anchor_weights.sh")
    return load_manifest()


@pytest.mark.parametrize("entry", config_entries(), ids=lambda e: e["file"])
def test_hash_matches_manifest(entry: dict, manifest: dict[str, tuple[int, str]]) -> None:
    name = pathlib.Path(entry["file"]).name
    assert name in manifest, f"{name} is in config.yaml but not in MANIFEST.tsv"
    _, sha = manifest[name]
    assert entry["sha256"] == sha, (
        f"{name}: config.yaml sha256 does not match the manifest.\n"
        f"  config:   {entry['sha256']}\n  manifest: {sha}"
    )


@pytest.mark.parametrize("entry", config_entries(), ids=lambda e: e["file"])
def test_byte_count_matches_manifest(entry: dict, manifest: dict[str, tuple[int, str]]) -> None:
    name = pathlib.Path(entry["file"]).name
    expected, _ = manifest[name]
    assert entry.get("bytes", expected) == expected


@pytest.mark.parametrize("entry", config_entries(), ids=lambda e: e["file"])
def test_hash_is_a_full_sha256(entry: dict) -> None:
    """Catches truncated or hand-typed hashes before they reach the paper."""
    h = entry["sha256"]
    assert len(h) == SHA256_LEN, f"{entry['file']}: sha256 is {len(h)} chars, expected {SHA256_LEN}"
    assert all(c in "0123456789abcdef" for c in h), f"{entry['file']}: non-hex characters in sha256"


def test_generation_labels_follow_the_stated_rule() -> None:
    """RQ1 turns entirely on this split, so it must be mechanical, not a judgement call."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cutoff = cfg["generation_rule"]["cutoff"][:7]  # YYYY-MM
    for m in cfg["models"]:
        released, label = m["released"], m["generation"]
        expected = "pre_2024" if released < cutoff else "post_2024"
        assert label == expected, (
            f"{m['id']}: released {released} but labelled {label}; "
            f"the rule in generation_rule gives {expected}"
        )


def test_panel_spans_both_generations() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    labels = {m["generation"] for m in cfg["models"]}
    assert labels == {"pre_2024", "post_2024"}, (
        "RQ1 is a generational comparison and needs models on both sides of the cutoff"
    )


def test_primary_panel_has_one_quantization_level() -> None:
    """Mixing quantization within the primary panel would confound generation
    with numerical precision. Q8_0 copies live in robustness_quantization."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    quants = {m["quantization"] for m in cfg["models"]}
    assert quants == {"Q4_K_M"}, f"primary panel mixes quantization levels: {quants}"


def test_paid_apis_are_off() -> None:
    """A hard stop, not a threshold. Flipping this needs David's approval."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["cost"]["paid_apis_enabled"] is False
    assert cfg["cost"]["local_only"] is True


def test_temperature_is_zero_for_the_primary_run() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["inference"]["temperature"] == 0.0


def test_h2b_is_configured_as_an_interaction() -> None:
    """CLAUDE.md: comparing two separate significance tests is the single most
    common error in this literature. Guard it in config, not just in prose."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert "interaction" in cfg["analysis"]["h2b_test"].lower()


def test_order_reversal_is_required_for_forced_choice() -> None:
    """4 of 5 pairs were order-dependent even on a post-2024 model."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["outcomes"]["forced_choice"]["order_reversal"] == "required"


def test_pending_approval_items_are_not_silently_active() -> None:
    """Anything touching a pre-registered section must stay flagged until David
    signs off. This test fails loudly if a flag is removed without the
    corresponding CHANGELOG entry."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    # SKIPS IN THE PUBLIC RELEASE. CHANGELOG.md is a development diary and is
    # deliberately not redistributed -- part of it records private
    # correspondence. The config invariant it guards is a property of the dev
    # tree, so its absence is expected rather than a failure.
    if not (ROOT / "CHANGELOG.md").exists():
        pytest.skip("CHANGELOG.md not redistributed with the public release")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    proposed = cfg["sample_size"]["proposed"]
    if not proposed.get("pending_approval", False):
        assert "power_analysis" in changelog or "S6" in changelog or "§6" in changelog, (
            "sample_size.proposed is no longer pending_approval but CHANGELOG.md "
            "records no approval of the PROTOCOL section 6 change"
        )
