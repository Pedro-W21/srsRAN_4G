"""Tests de l'indice d'équité de Jain.

Les valeurs attendues en pleine contention viennent de la théorie (voir
l'en-tête de ``fairness.py``) avec deux UE dont les canaux sont dans un
rapport 2 (300 et 150 octets par RBG) :

* parts de TTI égales (RR, PF alpha = 1) → J_RBG = 1 et
  J_octets = (2 + 1)² / (2 · (2² + 1²)) = 0,9 ;
* débits égaux (BET) → J_octets = 1 ;
* un seul UE servi (Max C/I) → J = 1/2.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sched_detector.cli import EXIT_OK, main
from sched_detector.fairness import FairnessReport, compute_fairness, jain_index
from sched_detector.features import build_timeline
from sched_detector.log_parser import parse_lines
from sched_detector.simulator import SimulatedPolicy, SimulatedUe, SimulationConfig, simulate

NOF_TTI: int = 8_000
TWO_CHANNELS_JAIN_BYTES: float = 0.9


def _fairness(
    policy: SimulatedPolicy, window_tti: int = 100, **overrides: object
) -> FairnessReport:
    parameters: dict[str, object] = {
        "policy": policy,
        "ues": (
            SimulatedUe(0x46, 300, 7_000, on_duration_tti=1_000, off_duration_tti=500),
            SimulatedUe(0x47, 150, 7_000),
        ),
        "nof_tti": NOF_TTI,
    }
    parameters.update(overrides)
    config = SimulationConfig(**parameters)  # type: ignore[arg-type]
    timeline = build_timeline(list(parse_lines(simulate(config))))
    return compute_fairness(timeline, window_tti=window_tti)


# --- Fonction jain_index ----------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([5.0, 5.0, 5.0], 1.0),
        ([1.0, 0.0], 0.5),
        ([1.0, 0.0, 0.0, 0.0], 0.25),
        ([2.0, 1.0], 0.9),
    ],
)
def test_jain_index_reference_values(values: list[float], expected: float) -> None:
    assert jain_index(values) == pytest.approx(expected)


def test_jain_index_is_scale_invariant() -> None:
    values = [3.0, 7.0, 1.0]
    assert jain_index(values) == pytest.approx(jain_index([1e6 * v for v in values]))


@pytest.mark.parametrize("values", [[], [4.0], [0.0, 0.0]])
def test_jain_index_undefined_cases_are_nan(values: list[float]) -> None:
    assert math.isnan(jain_index(values))


@pytest.mark.parametrize("bad_value", [-1.0, math.inf, math.nan])
def test_jain_index_rejects_invalid_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finies et positives"):
        jain_index([1.0, bad_value])


# --- Politiques simulées ----------------------------------------------------


def test_round_robin_shares_resources_equally_not_throughput() -> None:
    report = _fairness(SimulatedPolicy.ROUND_ROBIN)
    assert report.jain_contention_rbg == pytest.approx(1.0, abs=0.01)
    assert report.jain_contention_throughput == pytest.approx(TWO_CHANNELS_JAIN_BYTES, abs=0.01)


def test_blind_equal_throughput_equalises_throughput() -> None:
    report = _fairness(SimulatedPolicy.BLIND_EQUAL_THROUGHPUT)
    assert report.jain_contention_throughput == pytest.approx(1.0, abs=0.01)
    assert report.jain_contention_rbg < 0.95


def test_max_ci_serves_a_single_ue_under_contention() -> None:
    report = _fairness(SimulatedPolicy.MAX_CI)
    assert report.jain_contention_throughput == pytest.approx(0.5, abs=0.01)
    # Le second UE est servi pendant les pauses du premier : le J global
    # dépend du trafic offert et reste au-dessus du cas pleine contention.
    assert report.jain_throughput > report.jain_contention_throughput


def test_pf_fairness_increases_with_alpha() -> None:
    low = _fairness(SimulatedPolicy.PROPORTIONAL_FAIR, fairness_alpha=1.0)
    high = _fairness(SimulatedPolicy.PROPORTIONAL_FAIR, fairness_alpha=4.0)
    assert high.jain_contention_throughput > low.jain_contention_throughput


def test_one_tti_window_reveals_greedy_allocation() -> None:
    # Allocation gloutonne : un seul UE servi par TTI à saturation, donc J = 1/n
    # sur une fenêtre de 1 TTI (aux rares TTI près où le premier UE vide son
    # buffer et laisse des RBG au second), alors que le partage fréquentiel
    # reste au-dessus.
    greedy = _fairness(SimulatedPolicy.ROUND_ROBIN, window_tti=1)
    shared = _fairness(SimulatedPolicy.FREQUENCY_EQUAL_SHARE, window_tti=1)
    assert greedy.jain_short_term == pytest.approx(0.5, abs=0.01)
    assert shared.jain_short_term > 0.8


def test_report_details_are_consistent() -> None:
    report = _fairness(SimulatedPolicy.ROUND_ROBIN, window_tti=100)
    assert [ue.rnti for ue in report.ues] == [0x46, 0x47]
    assert all(ue.total_bytes >= ue.contention_bytes > 0 for ue in report.ues)
    assert report.nof_windows <= report.nof_contention_tti // 100
    assert report.jain_short_term_min <= report.jain_short_term


def test_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window_tti"):
        _fairness(SimulatedPolicy.ROUND_ROBIN, window_tti=0)


# --- Ligne de commande ------------------------------------------------------


def test_cli_prints_fairness_section(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = SimulationConfig(
        SimulatedPolicy.ROUND_ROBIN,
        (
            SimulatedUe(0x46, 300, 7_000, on_duration_tti=1_000, off_duration_tti=500),
            SimulatedUe(0x47, 150, 7_000),
        ),
        nof_tti=NOF_TTI,
    )
    path = tmp_path / "enb.log"
    path.write_text("\n".join(simulate(config)) + "\n", encoding="utf-8")
    assert main([str(path), "--jain-window", "10"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "indice de Jain" in output
    assert "fenêtres de   10 TTI" in output


def test_cli_rejects_invalid_window(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path / "enb.log"), "--jain-window", "0"])
