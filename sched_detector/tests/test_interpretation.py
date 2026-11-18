"""Tests des chiffres « sans piège » du rapport (section En bref, tableau, Jain).

Le scénario de référence reproduit un vrai rapport UL : un UE favori qui gagne
90 % des TTI disputés, deux canaux identiques, et un troisième UE connecté qui
n'émet presque rien. La précision brute du RR y vaut 90 % : le rapport doit
montrer qu'elle ne vaut pas mieux que « toujours le favori ».
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from sched_detector.cli import EXIT_OK, main
from sched_detector.detector import Detection, detect
from sched_detector.interpretation import (
    ModelStatus,
    describe_closest_model,
    effective_ues,
    favourite_rnti,
    format_points,
    gain_points,
    model_status,
    status_reason,
    summarize,
)
from sched_detector.log_parser import Direction, parse_lines
from sched_detector.simulator import SimulatedPolicy, SimulatedUe, SimulationConfig, simulate

FAVOURITE, OTHER, IDLE = 0x4C, 0x4F, 0x4B


def _ul_line(rnti: int, tti: int, start: int, stop: int, tbs: int, bsr: int) -> str:
    return (
        f"SCHED: UL tx rnti=0x{rnti:x}, cc=0, pid={tti % 8}, dci=(1,0), prb=[{start}, {stop}), "
        f"n_rtx=0, cfi=1, tbs={tbs}, bsr={bsr} ({bsr + tbs}-{tbs}), tti_tx_ul={tti % 10_240}"
    )


def _favourite_log(nof_tti: int = 6_000) -> list[str]:
    """0x4c servi en premier 9 fois sur 10, canaux identiques, 0x4b quasi muet."""
    generator = random.Random(7)
    lines = []
    for tti in range(nof_tti):
        first, second = (FAVOURITE, OTHER) if generator.random() < 0.9 else (OTHER, FAVOURITE)
        lines.append(_ul_line(first, tti, 2, 47, 2_865, 140_000))
        lines.append(_ul_line(second, tti, 47, 48, 64, 140_000))
        if tti % 997 == 0:
            lines.append(_ul_line(IDLE, tti, 0, 1, 7, 0))
    return lines


@pytest.fixture(scope="module")
def favourite_detection() -> Detection:
    return detect(list(parse_lines(_favourite_log(), direction=Direction.UL)))


@pytest.fixture(scope="module")
def round_robin_detection() -> Detection:
    ues = (
        SimulatedUe(0x46, 100, 7_000, on_duration_tti=1_000, off_duration_tti=500),
        SimulatedUe(0x47, 50, 7_000),
    )
    config = SimulationConfig(
        SimulatedPolicy.ROUND_ROBIN, ues, nof_tti=8_000, direction=Direction.UL
    )
    return detect(list(parse_lines(simulate(config), direction=Direction.UL)))


# --- Le piège du 90 % --------------------------------------------------------------


def test_round_robin_score_is_shown_next_to_the_favourite(
    favourite_detection: Detection,
) -> None:
    rr = favourite_detection.evidence.round_robin
    assert rr.decisive_accuracy == pytest.approx(0.9, abs=0.02)
    assert rr.judged_naive_accuracy == pytest.approx(rr.decisive_accuracy, abs=0.01)
    assert abs(gain_points(rr)) < 1.0
    assert model_status(rr) is ModelStatus.REJECTED
    assert status_reason(rr) == "rejeté : pas mieux que le favori"


def test_no_closest_model_is_claimed_when_none_beats_the_favourite(
    favourite_detection: Detection,
) -> None:
    assert favourite_rnti(favourite_detection) == FAVOURITE
    line = describe_closest_model(favourite_detection)
    assert "aucun" in line
    assert "Round Robin" not in line


def test_summary_gives_figures_against_the_favourite(favourite_detection: Detection) -> None:
    summary = summarize(favourite_detection)
    rr_line = next(line for line in summary if line.startswith("Round Robin"))
    assert "non" in rr_line
    assert "désignant toujours 0x4c" in rr_line
    assert "gain +0 pt" in rr_line
    assert any("0x4c 90%" in line for line in summary)
    assert any(line.startswith("Max C/I : non testable") for line in summary)


def test_summary_suggests_ignoring_the_idle_ue(favourite_detection: Detection) -> None:
    hint = next(line for line in summarize(favourite_detection) if line.startswith("Astuce"))
    assert "--ignore-rnti 0x4b" in hint


# --- Une vraie politique connue reste lisible ----------------------------------------


def test_validated_round_robin_shows_a_large_gain(round_robin_detection: Detection) -> None:
    rr = round_robin_detection.evidence.round_robin
    assert model_status(rr) is ModelStatus.VALIDATED
    assert gain_points(rr) > 40
    rr_line = next(line for line in summarize(round_robin_detection) if "Round Robin :" in line)
    assert rr_line.startswith("Round Robin : oui")


def test_untestable_model_explains_why(favourite_detection: Detection) -> None:
    max_ci = favourite_detection.evidence.pf_family[0]
    assert max_ci.fairness_alpha == 0.0
    assert model_status(max_ci) is ModelStatus.UNTESTABLE
    assert status_reason(max_ci).startswith("non testable")


# --- Mise en forme -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("points", "expected"),
    [(0.2, "+0 pt"), (-80.4, "-80 pts"), (47.6, "+48 pts"), (math.nan, "-")],
)
def test_format_points(points: float, expected: str) -> None:
    assert format_points(points) == expected


def test_effective_ues_translates_jain() -> None:
    assert effective_ues(0.374, 3) == "≈ 1.1 UE sur 3 servis à égalité"
    assert effective_ues(1.0, 2) == "≈ 2.0 UE sur 2 servis à égalité"
    assert effective_ues(math.nan, 3) == ""


# --- CLI ------------------------------------------------------------------------------


def test_report_table_has_reference_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "enb.log"
    path.write_text("\n".join(_favourite_log()) + "\n", encoding="utf-8")
    assert main([str(path)]) == EXIT_OK
    output = capsys.readouterr().out
    assert "a raison  favori     gain  verdict" in output
    assert "UE sur 3 servis à égalité" in output
    assert "En bref :" in output


def test_ignore_rnti_keeps_rr_modulo_and_fills_jain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "enb.log"
    path.write_text("\n".join(_favourite_log()) + "\n", encoding="utf-8")
    assert main([str(path), "--ignore-rnti", "0x4b"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "UE écartés de l'analyse : 0x4b (N = 3 pour le modulo du RR)" in output
    assert "0x4b" not in output.split("En bref :")[1].split("Équité")[1]
    assert "UE sur 2 servis à égalité" in output
    assert "Astuce" not in output


def test_invalid_rnti_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path / "enb.log"), "--ignore-rnti", "pas_un_rnti"])
