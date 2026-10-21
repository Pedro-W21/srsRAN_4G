"""Tests de l'interface en ligne de commande et de ses codes de sortie."""

from __future__ import annotations

from pathlib import Path

import pytest

from sched_detector.cli import EXIT_INPUT_ERROR, EXIT_OK, EXIT_UNEXPECTED_POLICY, main
from sched_detector.simulator import SimulatedPolicy, SimulatedUe, SimulationConfig, simulate


@pytest.fixture
def round_robin_log(tmp_path: Path) -> Path:
    ues = (
        SimulatedUe(0x46, 300, 7_000, on_duration_tti=1_000, off_duration_tti=500),
        SimulatedUe(0x47, 150, 7_000),
    )
    config = SimulationConfig(SimulatedPolicy.ROUND_ROBIN, ues, nof_tti=8_000)
    path = tmp_path / "enb.log"
    header = "[MAC    ] [I] Using time-domain RR scheduling policy for cc=0\n"
    path.write_text(header + "\n".join(simulate(config)) + "\n", encoding="utf-8")
    return path


def test_reports_round_robin_and_declared_policy(
    round_robin_log: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(round_robin_log)]) == EXIT_OK
    output = capsys.readouterr().out
    assert "time_rr (Round Robin)" in output
    assert "time-domain RR" in output


def test_expect_matching_policy_exits_zero(round_robin_log: Path) -> None:
    assert main([str(round_robin_log), "--expect", "rr"]) == EXIT_OK


def test_expect_other_policy_exits_two(round_robin_log: Path) -> None:
    assert main([str(round_robin_log), "--expect", "pf"]) == EXIT_UNEXPECTED_POLICY


def test_missing_file_exits_one(tmp_path: Path) -> None:
    assert main([str(tmp_path / "absent.log")]) == EXIT_INPUT_ERROR


def test_log_without_sched_lines_exits_one(tmp_path: Path) -> None:
    path = tmp_path / "enb.log"
    path.write_text("[RRC    ] [I] nothing here\n", encoding="utf-8")
    assert main([str(path)]) == EXIT_INPUT_ERROR


def test_custom_policy_report_and_expect_custom(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ues = (
        SimulatedUe(0x46, 300, 7_000, on_duration_tti=1_000, off_duration_tti=500),
        SimulatedUe(0x47, 150, 7_000),
    )
    config = SimulationConfig(SimulatedPolicy.LONGEST_QUEUE_FIRST, ues, nof_tti=8_000)
    path = tmp_path / "enb.log"
    path.write_text("\n".join(simulate(config)) + "\n", encoding="utf-8")
    assert main([str(path), "--expect", "custom"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "Scheduler personnalisé" in output
    assert "Modèle connu le plus proche" in output
