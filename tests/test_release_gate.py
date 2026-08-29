r"""The release gate's byte-fidelity contract, tested for the first time.

copy_sanitised() reads and writes with newline="" because Path.write_text on
Windows translates every \n -- which once left all 286 shipped files
byte-different from their sources, in an artifact whose entire purpose is
that a reader can check the files against the paper. That contract had no
test: nothing in tests/ imported build_release_repo at all, so a well-meaning
simplification to read_text/write_text would have silently corrupted the
public artifact again.
"""

import filecmp
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
sys.path.insert(0, str(SRC))

GATE = SRC / "build_release_repo.py"
pytestmark = pytest.mark.skipif(not GATE.exists(),
                                reason="gate not shipped (public clone)")


def _gate():
    import importlib
    import build_release_repo as g
    return importlib.reload(g)


def test_crlf_and_lf_files_survive_byte_identical(tmp_path):
    g = _gate()
    for name, body in [("crlf.md", b"line one\r\nline two\r\n"),
                       ("lf.md", b"line one\nline two\n"),
                       ("mixed.md", b"a\r\nb\nc\r\n")]:
        src = tmp_path / name
        src.write_bytes(body)
        dst = tmp_path / ("out_" + name)
        g.copy_sanitised(src, dst)
        assert dst.read_bytes() == body, (
            f"{name}: newline translation corrupted the copy")


def test_a_clean_file_ships_byte_identical(tmp_path):
    g = _gate()
    src = tmp_path / "clean.py"
    src.write_bytes(b"x = 1\r\n# nothing to sanitise\r\n")
    dst = tmp_path / "out.py"
    hits = g.copy_sanitised(src, dst)
    assert hits == 0
    assert filecmp.cmp(src, dst, shallow=False)


def test_a_sanitised_path_changes_only_that_span(tmp_path):
    g = _gate()
    body = ('before\r\npath = "x"'
            "\r\nafter\r\n")
    src = tmp_path / "hit.py"
    src.write_bytes(body.encode())
    dst = tmp_path / "out.py"
    hits = g.copy_sanitised(src, dst)
    out = dst.read_bytes().decode()
    assert hits >= 1
    assert "Stanford" not in out
    assert out.startswith("before\r\n") and out.endswith("after\r\n"), (
        "sanitising rewrote more than the matched span")


def test_a_non_utf8_text_file_is_refused_not_shipped(tmp_path):
    """The UTF-16 hole: such a file used to be copied VERBATIM, skipping the
    sanitiser and every DENY_TEXT rule."""
    g = _gate()
    # The fixture text is deliberately bland: this test is about the ENCODING
    # refusal, and the first version put a deny-listed term here -- whereupon
    # the gate refused to ship this test file, which was the gate working.
    src = tmp_path / "utf16.md"
    src.write_bytes("any text at all".encode("utf-16"))
    dst = tmp_path / "out.md"
    with pytest.raises(SystemExit):
        g.copy_sanitised(src, dst)
    assert not dst.exists()
