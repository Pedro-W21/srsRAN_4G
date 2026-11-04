"""Tests de la lecture des lignes SCHED: DL."""

from __future__ import annotations

from pathlib import Path

import pytest

from sched_detector.log_parser import (
    TTI_WRAP_PERIOD,
    find_declared_policy,
    parse_line,
    parse_lines,
    read_log,
)

REAL_FORMAT_LINE = (
    "2026-10-14T14:02:11.123456 [MAC    ] [I] [ 5119] SCHED: DL tx rnti=0x46, cc=0, pid=2, "
    "mask=0x1ffff, dci=(2,8), n_rtx=0, cfi=1, tbs=4584, buffer=12000/7400, tti_tx_dl=5123"
)


def _line(tti: int, kind: str = "tx", cc: int = 0, rnti: int = 0x46) -> str:
    return (
        f"SCHED: DL {kind} rnti=0x{rnti:x}, cc={cc}, pid=0, mask=0x7, dci=(2,0), "
        f"n_rtx=0, cfi=1, tbs=300, buffer=1000/700, tti_tx_dl={tti}"
    )


def test_parses_every_field_of_a_real_line() -> None:
    allocation = parse_line(REAL_FORMAT_LINE)
    assert allocation is not None
    assert allocation.rnti == 0x46
    assert allocation.carrier_index == 0
    assert not allocation.is_retransmission
    assert allocation.nof_rbg == 17
    assert allocation.tbs_bytes == 4584
    assert (allocation.buffer_before_bytes, allocation.buffer_after_bytes) == (12000, 7400)
    assert allocation.tti_tx_dl == 5123


def test_flags_retransmissions() -> None:
    allocation = parse_line(_line(10, kind="retx"))
    assert allocation is not None
    assert allocation.is_retransmission


@pytest.mark.parametrize(
    "line",
    [
        "",
        "SCHED: UL tx rnti=0x46, cc=0, pid=5, dci=(2,4), prb=[2, 48), tbs=2216, tti_tx_ul=5127",
        "SCHED: DL tx rnti=0x46, cc=0, pid=2, mask=0x1ffff",  # tronquée
        "[RRC    ] [I] User 0x46 connected",
    ],
)
def test_ignores_lines_that_are_not_complete_dl_allocations(line: str) -> None:
    assert parse_line(line) is None


def test_filters_other_carriers() -> None:
    lines = [_line(1, cc=0), _line(1, cc=1), _line(2, cc=0)]
    assert [a.tti_tx_dl for a in parse_lines(lines, carrier_index=0)] == [1, 2]


def test_unwraps_tti_counter_at_10240() -> None:
    lines = [_line(10238), _line(10239), _line(0), _line(1)]
    unwrapped = [a.tti_unwrapped for a in parse_lines(lines)]
    assert unwrapped == [10238, 10239, TTI_WRAP_PERIOD, TTI_WRAP_PERIOD + 1]


def test_small_backward_step_is_not_a_wrap() -> None:
    lines = [_line(500, rnti=0x46), _line(499, rnti=0x47)]
    assert [a.tti_unwrapped for a in parse_lines(lines)] == [500, 499]


def test_read_log_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="introuvable"):
        read_log(tmp_path / "absent.log")


def test_finds_declared_policy() -> None:
    lines = ["[MAC    ] [I] Using time-domain RR scheduling policy for cc=0", _line(1)]
    assert find_declared_policy(lines) == "time-domain RR"


def test_declared_policy_absent_returns_none() -> None:
    assert find_declared_policy([_line(1)]) is None
