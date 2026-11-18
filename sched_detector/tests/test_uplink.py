"""Tests de la voie montante : lecture des lignes ``SCHED: UL``, verdicts, Jain, CLI.

Les logs simulés reproduisent ce que montre un vrai log srsENB en UL : grant
pleine bande ``prb=[2, 47)`` (45 PRB, taille DFT valide), 1 PRB restant
``[47, 48)`` donné à un autre UE, bande réduite ``[10, 46)`` les TTI de
PRACH, et champ ``bsr`` pour savoir quels UE attendaient.

Le fichier ``data/real_ul_excerpt.log`` contient 40 lignes ``SCHED: UL tx``
et une ligne ``SCHED: DL tx`` d'un log srsENB réel (2026-11-13, deux UE en
``iperf3`` montant, politique personnalisée).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pytest

from sched_detector.classifier import MIN_CONTESTED_TTI, Confidence, Policy
from sched_detector.cli import EXIT_INCONCLUSIVE, EXIT_INPUT_ERROR, EXIT_OK, main
from sched_detector.detector import Detection, choose_direction, detect
from sched_detector.fairness import compute_fairness, jain_index
from sched_detector.features import build_timeline, extract_features
from sched_detector.log_parser import (
    TTI_WRAP_PERIOD,
    Direction,
    parse_all_directions,
    parse_line,
    parse_lines,
)
from sched_detector.simulator import (
    SimulatedPolicy,
    SimulatedUe,
    SimulationConfig,
    is_valid_dft_size,
    simulate,
)

REAL_EXCERPT = Path(__file__).parent / "data" / "real_ul_excerpt.log"
REAL_UL_LINE = (
    "2026-11-13T09:25:13.220986 [MAC    ] [I] [ 2616] SCHED: UL tx rnti=0x4c, cc=0, pid=0, "
    "dci=(1,0), prb=[2, 47), n_rtx=0, cfi=1, tbs=2865, bsr=138208 (152053-13845), "
    "tti_tx_ul=2624"
)
#: 1,5 fois le TBS pleine bande du bon canal : BSR toujours non nul.
SATURATING_ARRIVAL_BYTES: int = 7_000
NOF_TTI: int = 12_000
#: Départ proche de 10240 pour traverser un rebouclage du compteur de TTI.
FIRST_TTI: int = 9_000
#: Octets par PRB : deux canaux (rapport 2), trois canaux, canaux identiques.
TWO_CHANNELS: list[float] = [100, 50]
THREE_CHANNELS: list[float] = [100, 50, 33]
IDENTICAL_CHANNELS: list[float] = [64, 64]


def _ues(
    bytes_per_prb: list[float],
    is_first_intermittent: bool = True,
    arrivals: list[int] | None = None,
) -> tuple[SimulatedUe, ...]:
    return tuple(
        SimulatedUe(
            rnti=0x46 + index,
            bytes_per_resource=rate,
            arrival_bytes_per_tti=arrivals[index] if arrivals else SATURATING_ARRIVAL_BYTES,
            on_duration_tti=1_000 if (is_first_intermittent and index == 0) else 0,
            off_duration_tti=500 if (is_first_intermittent and index == 0) else 0,
        )
        for index, rate in enumerate(bytes_per_prb)
    )


def _config(
    policy: SimulatedPolicy, channels: list[float], **overrides: object
) -> SimulationConfig:
    parameters: dict[str, object] = {
        "policy": policy,
        "ues": _ues(channels),
        "nof_tti": NOF_TTI,
        "first_tti": FIRST_TTI,
        "retx_probability": 0.02,
        "direction": Direction.UL,
    }
    parameters.update(overrides)
    return SimulationConfig(**parameters)  # type: ignore[arg-type]


def _run(config: SimulationConfig) -> Detection:
    return detect(list(parse_lines(simulate(config), direction=Direction.UL)))


def _write_log(tmp_path: Path, config: SimulationConfig) -> Path:
    path = tmp_path / "enb.log"
    path.write_text("\n".join(simulate(config)) + "\n", encoding="utf-8")
    return path


# --- Lecture des lignes SCHED: UL ---------------------------------------------


def test_parses_every_field_of_a_real_ul_line() -> None:
    allocation = parse_line(REAL_UL_LINE, Direction.UL)
    assert allocation is not None
    assert allocation.direction is Direction.UL
    assert allocation.rnti == 0x4C
    assert allocation.carrier_index == 0
    assert not allocation.is_retransmission
    assert allocation.nof_resources == 45
    assert allocation.resource_mask == ((1 << 45) - 1) << 2
    assert allocation.tbs_bytes == 2865
    assert (allocation.pending_before_bytes, allocation.pending_after_bytes) == (152053, 138208)
    assert allocation.tti == 2624


def test_ul_and_dl_parsers_ignore_each_other() -> None:
    assert parse_line(REAL_UL_LINE, Direction.DL) is None
    dl_line = (
        "SCHED: DL tx rnti=0x4f, cc=0, pid=0, mask=0x10000, dci=(0, 1), n_rtx=0, cfi=1, "
        "tbs=63, buffer=62/0, tti_tx_dl=2623"
    )
    assert parse_line(dl_line, Direction.UL) is None
    assert parse_line(dl_line, Direction.DL) is not None


def test_parses_ul_retx_and_ignores_msg3() -> None:
    retx = REAL_UL_LINE.replace("UL tx", "UL retx")
    parsed = parse_line(retx, Direction.UL)
    assert parsed is not None
    assert parsed.is_retransmission
    assert parse_line(REAL_UL_LINE.replace("UL tx", "Msg3 tx"), Direction.UL) is None


def test_negative_bsr_is_clamped_to_zero() -> None:
    line = REAL_UL_LINE.replace("bsr=138208 (152053-13845)", "bsr=-120 (0-120)")
    parsed = parse_line(line, Direction.UL)
    assert parsed is not None
    assert parsed.pending_after_bytes == 0


def test_ul_tti_is_unwrapped() -> None:
    lines = [
        REAL_UL_LINE.replace("tti_tx_ul=2624", f"tti_tx_ul={tti}")
        for tti in (TTI_WRAP_PERIOD - 2, TTI_WRAP_PERIOD - 1, 0, 1)
    ]
    unwrapped = [a.tti_unwrapped for a in parse_lines(lines, direction=Direction.UL)]
    assert unwrapped == [TTI_WRAP_PERIOD - 2, TTI_WRAP_PERIOD - 1, TTI_WRAP_PERIOD, 10_241]


def test_real_excerpt_is_split_by_direction() -> None:
    with REAL_EXCERPT.open(encoding="utf-8") as log_file:
        allocations = parse_all_directions(log_file)
    assert len(allocations[Direction.UL]) == 40
    assert len(allocations[Direction.DL]) == 1
    assert {a.rnti for a in allocations[Direction.UL]} == {0x4C, 0x4F}


def test_timeline_rejects_mixed_directions() -> None:
    with REAL_EXCERPT.open(encoding="utf-8") as log_file:
        allocations = parse_all_directions(log_file)
    with pytest.raises(ValueError, match="séparément"):
        build_timeline(allocations[Direction.UL] + allocations[Direction.DL])


# --- Simulateur UL --------------------------------------------------------------


def test_simulated_ul_grants_are_contiguous_valid_dft_sizes() -> None:
    allocations = list(
        parse_lines(
            simulate(_config(SimulatedPolicy.ROUND_ROBIN, TWO_CHANNELS)), direction=Direction.UL
        )
    )
    assert allocations
    for allocation in allocations:
        mask = allocation.resource_mask
        assert mask >> (mask & -mask).bit_length() - 1 == (1 << allocation.nof_resources) - 1
        assert is_valid_dft_size(allocation.nof_resources)
        assert mask & 0b11 == 0 and mask >> 48 == 0  # PUCCH aux bords


# --- Politiques du catalogue ----------------------------------------------------


@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS, IDENTICAL_CHANNELS])
def test_detects_round_robin_in_uplink(channels: list[float]) -> None:
    verdict = _run(_config(SimulatedPolicy.ROUND_ROBIN, channels)).verdict
    assert verdict.policy is Policy.ROUND_ROBIN
    assert verdict.confidence is Confidence.HIGH
    assert any("tti_tx_ul mod N" in reason for reason in verdict.reasons)


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS])
def test_detects_pf_in_uplink_and_estimates_alpha(channels: list[float], alpha: float) -> None:
    verdict = _run(
        _config(SimulatedPolicy.PROPORTIONAL_FAIR, channels, fairness_alpha=alpha)
    ).verdict
    assert verdict.policy is Policy.PROPORTIONAL_FAIR
    assert verdict.confidence is Confidence.HIGH
    assert verdict.fairness_alpha == pytest.approx(alpha, rel=0.15)


@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS])
def test_detects_max_ci_in_uplink(channels: list[float]) -> None:
    verdict = _run(_config(SimulatedPolicy.MAX_CI, channels)).verdict
    assert verdict.policy is Policy.MAX_CI


@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS])
def test_detects_blind_equal_throughput_in_uplink(channels: list[float]) -> None:
    verdict = _run(_config(SimulatedPolicy.BLIND_EQUAL_THROUGHPUT, channels)).verdict
    assert verdict.policy is Policy.BLIND_EQUAL_THROUGHPUT
    assert math.isinf(verdict.fairness_alpha)


# --- Politiques hors catalogue ---------------------------------------------------


@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS])
@pytest.mark.parametrize(
    "policy",
    [
        SimulatedPolicy.RANDOM,
        SimulatedPolicy.LONGEST_QUEUE_FIRST,
        SimulatedPolicy.FREQUENCY_EQUAL_SHARE,
    ],
)
def test_unknown_uplink_policy_is_reported_as_custom(
    policy: SimulatedPolicy, channels: list[float]
) -> None:
    verdict = _run(_config(policy, channels)).verdict
    assert verdict.policy is Policy.CUSTOM
    assert verdict.confidence is Confidence.HIGH


def test_identical_ul_channels_lower_confidence_of_custom_verdict() -> None:
    verdict = _run(_config(SimulatedPolicy.MAX_CI, IDENTICAL_CHANNELS)).verdict
    assert verdict.policy is Policy.CUSTOM
    assert verdict.confidence is Confidence.LOW


# --- Bande partagée : l'arrondi SC-FDMA n'est pas un partage -------------------------


def test_dft_rounding_leftover_is_not_shared_band() -> None:
    # RR glouton : le premier UE prend 45 PRB, l'autre le PRB restant, et
    # tous deux gardent un BSR non nul. Ce n'est pas une politique fréquentielle.
    features = _run(_config(SimulatedPolicy.ROUND_ROBIN, TWO_CHANNELS)).features
    assert features.greedy_violation_rate < 0.01


def test_equal_ul_split_is_shared_band() -> None:
    detection = _run(_config(SimulatedPolicy.FREQUENCY_EQUAL_SHARE, TWO_CHANNELS))
    assert detection.features.greedy_violation_rate > 0.9
    assert any("bande est partagée" in reason for reason in detection.verdict.reasons)
    assert any("PRB restants" in reason for reason in detection.verdict.reasons)


def test_real_excerpt_leftover_grants_are_not_shared_band() -> None:
    with REAL_EXCERPT.open(encoding="utf-8") as log_file:
        timeline = build_timeline(parse_all_directions(log_file)[Direction.UL])
    assert extract_features(timeline).greedy_violation_rate == 0.0


# --- Indices de Jain en UL ------------------------------------------------------------


def test_real_excerpt_jain_matches_hand_computation() -> None:
    with REAL_EXCERPT.open(encoding="utf-8") as log_file:
        allocations = parse_all_directions(log_file)[Direction.UL]
    received: dict[int, int] = defaultdict(int)
    for allocation in allocations:
        received[allocation.rnti] += allocation.tbs_bytes
    expected = sum(received.values()) ** 2 / (2 * sum(v * v for v in received.values()))
    report = compute_fairness(build_timeline(allocations))
    assert report.jain_throughput == pytest.approx(expected)
    assert report.jain_throughput == pytest.approx(jain_index(received.values()))


def test_ul_round_robin_shares_prbs_equally_not_throughput() -> None:
    config = _config(SimulatedPolicy.ROUND_ROBIN, TWO_CHANNELS, ues=_ues(TWO_CHANNELS, False))
    report = _run(config).fairness
    assert report.jain_contention_resources == pytest.approx(1.0, abs=0.02)
    assert report.jain_contention_throughput < 0.95


def test_ul_blind_equal_throughput_equalises_throughput() -> None:
    config = _config(
        SimulatedPolicy.BLIND_EQUAL_THROUGHPUT, TWO_CHANNELS, ues=_ues(TWO_CHANNELS, False)
    )
    report = _run(config).fairness
    assert report.jain_contention_throughput == pytest.approx(1.0, abs=0.02)
    assert report.jain_contention_resources < 0.95


def test_short_term_windows_bridge_ttis_without_grant() -> None:
    # En UL, les grants ne tombent pas à chaque TTI : un TTI sans aucune
    # allocation ne doit pas couper les fenêtres de l'indice à court terme.
    lines = []
    for index in range(400):
        tti = 2 * index  # un TTI sur deux sans grant
        rnti = 0x46 + index % 2
        lines.append(
            f"SCHED: UL tx rnti=0x{rnti:x}, cc=0, pid=0, dci=(1,0), prb=[2, 47), n_rtx=0, "
            f"cfi=1, tbs=2865, bsr=100000 (102865-2865), tti_tx_ul={tti}"
        )
    report = compute_fairness(build_timeline(list(parse_lines(lines, direction=Direction.UL))))
    assert report.nof_windows >= 3
    assert report.jain_short_term == pytest.approx(1.0, abs=0.01)


# --- Choix de la voie et CLI -------------------------------------------------------------


def test_choose_direction_prefers_the_contested_direction() -> None:
    lines = list(simulate(_config(SimulatedPolicy.ROUND_ROBIN, TWO_CHANNELS, nof_tti=2_000)))
    lines.append(
        "SCHED: DL tx rnti=0x46, cc=0, pid=0, mask=0x1, dci=(0,1), n_rtx=0, cfi=1, "
        "tbs=63, buffer=62/0, tti_tx_dl=5"
    )
    choice = choose_direction(parse_all_directions(lines))
    assert choice.direction is Direction.UL
    assert choice.contested_tti[Direction.DL] == 0
    assert choice.contested_tti[Direction.UL] > 0


def test_choose_direction_rejects_log_without_allocations() -> None:
    with pytest.raises(ValueError, match="Aucune nouvelle transmission DL ni UL"):
        choose_direction(parse_all_directions(["[RRC    ] [I] rien"]))


def test_cli_auto_direction_analyses_uplink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_log(tmp_path, _config(SimulatedPolicy.ROUND_ROBIN, TWO_CHANNELS))
    assert main([str(path), "--expect", "rr"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "Voie analysée : montante (UL), choisie automatiquement" in output
    assert "PRB en pleine contention" in output
    assert "octets/PRB" in output


def test_cli_forced_direction_without_lines_is_input_error(tmp_path: Path) -> None:
    path = _write_log(tmp_path, _config(SimulatedPolicy.ROUND_ROBIN, TWO_CHANNELS, nof_tti=500))
    assert main([str(path), "--direction", "dl"]) == EXIT_INPUT_ERROR


def test_cli_on_real_excerpt_reports_jain_and_is_inconclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 40 grants : bien moins que MIN_CONTESTED_TTI, mais les indices de Jain
    # restent calculés et affichés.
    assert main([str(REAL_EXCERPT), "--direction", "ul"]) == EXIT_INCONCLUSIVE
    output = capsys.readouterr().out
    assert f"minimum {MIN_CONTESTED_TTI}" in output
    assert "débit sur tout le log" in output
    assert "0x4c" in output and "0x4f" in output
