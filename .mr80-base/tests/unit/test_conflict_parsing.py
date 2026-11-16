import pytest

from conciliator.integrations.git.conflicts import (
    ConflictParseError,
    parse_conflict_file,
)
from tests.unit.fixtures.conflict_samples import (
    SAMPLE_EMPTY_SIDES,
    SAMPLE_MALFORMED,
    SAMPLE_NO_CONFLICT,
    SAMPLE_SINGLE_CONFLICT,
    SAMPLE_TWO_CONFLICTS,
)


def test_no_conflict_returns_empty():
    assert parse_conflict_file("a.py", SAMPLE_NO_CONFLICT) == []


def test_single_conflict():
    blocks = parse_conflict_file("a.py", SAMPLE_SINGLE_CONFLICT)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.file_path == "a.py"
    assert b.block_idx == 0
    assert b.start_line == 2  # <<< 是第 2 行
    assert b.end_line == 6
    assert b.ours_text == "ours line\n"
    assert b.theirs_text == "theirs line\n"


def test_two_conflicts_increment_idx():
    blocks = parse_conflict_file("a.py", SAMPLE_TWO_CONFLICTS)
    assert len(blocks) == 2
    assert blocks[0].block_idx == 0
    assert blocks[1].block_idx == 1
    assert blocks[0].ours_text == "ours1\n"
    assert blocks[1].theirs_text == "theirs2\n"


def test_empty_ours():
    blocks = parse_conflict_file("a.py", SAMPLE_EMPTY_SIDES)
    assert len(blocks) == 1
    assert blocks[0].ours_text == ""
    assert blocks[0].theirs_text == "theirs only\n"


def test_malformed_raises():
    with pytest.raises(ConflictParseError) as exc_info:
        parse_conflict_file("a.py", SAMPLE_MALFORMED)
    assert "missing" in exc_info.value.detail or "unexpected" in exc_info.value.detail


def test_real_python_file():
    content = """\
import os

class Foo:
<<<<<<< HEAD
    def new_method(self):
        return 1
=======
    def old_method(self):
        return 2
>>>>>>> main

    def shared(self):
        return 3
"""
    blocks = parse_conflict_file("foo.py", content)
    assert len(blocks) == 1
    assert "def new_method" in blocks[0].ours_text
    assert "def old_method" in blocks[0].theirs_text
