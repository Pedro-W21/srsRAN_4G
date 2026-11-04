"""Tests de bout en bout : logs simulés dont la politique est connue → verdict.

Les politiques du catalogue doivent être reconnues ; les politiques absentes
du catalogue (LQF, aléatoire, partage fréquentiel) doivent sortir en
« Scheduler personnalisé », jamais déguisées en politique connue.
"""

from __future__ import annotations

import math

import pytest

from sched_detector.classifier import MIN_CONTESTED_TTI, Confidence, Policy
from sched_detector.detector import Detection, detect
from sched_detector.features import build_timeline
from sched_detector.log_parser import parse_lines
from sched_detector.models import fit_pf_model
from sched_detector.simulator import SimulatedPolicy, SimulatedUe, SimulationConfig, simulate

#: 1,2 fois le TBS pleine bande d'un bon canal : file RLC toujours pleine.
SATURATING_ARRIVAL_BYTES: int = 7_000
NOF_TTI: int = 12_000
#: Départ proche de 10240 pour traverser un rebouclage du compteur de TTI.
FIRST_TTI: int = 9_000
#: Deux canaux différents (rapport 2) et trois canaux différents.
TWO_CHANNELS: list[float] = [300, 150]
THREE_CHANNELS: list[float] = [300, 150, 100]
IDENTICAL_CHANNELS: list[float] = [300, 300]


def _ues(
    bytes_per_rbg: list[float],
    is_first_intermittent: bool = True,
    arrivals: list[int] | None = None,
) -> tuple[SimulatedUe, ...]:
    # L'UE intermittent est le premier, celui au meilleur canal : sous Max C/I
    # c'est pendant ses pauses que les autres UE apparaissent dans le log.
    return tuple(
        SimulatedUe(
            rnti=0x46 + index,
            bytes_per_rbg=rate,
            arrival_bytes_per_tti=arrivals[index] if arrivals else SATURATING_ARRIVAL_BYTES,
            on_duration_tti=1_000 if (is_first_intermittent and index == 0) else 0,
            off_duration_tti=500 if (is_first_intermittent and index == 0) else 0,
        )
        for index, rate in enumerate(bytes_per_rbg)
    )


def _run(config: SimulationConfig) -> Detection:
    return detect(list(parse_lines(simulate(config))))


def _config(
    policy: SimulatedPolicy, channels: list[float], **overrides: object
) -> SimulationConfig:
    parameters: dict[str, object] = {
        "policy": policy,
        "ues": _ues(channels),
        "nof_tti": NOF_TTI,
        "first_tti": FIRST_TTI,
        "retx_probability": 0.02,
    }
    parameters.update(overrides)
    return SimulationConfig(**parameters)  # type: ignore[arg-type]


# --- Politiques du catalogue ------------------------------------------------


@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS, IDENTICAL_CHANNELS])
def test_detects_round_robin(channels: list[float]) -> None:
    verdict = _run(_config(SimulatedPolicy.ROUND_ROBIN, channels)).verdict
    assert verdict.policy is Policy.ROUND_ROBIN
    assert verdict.confidence is Confidence.HIGH


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS])
def test_detects_pf_and_estimates_alpha(channels: list[float], alpha: float) -> None:
    verdict = _run(
        _config(SimulatedPolicy.PROPORTIONAL_FAIR, channels, fairness_alpha=alpha)
    ).verdict
    assert verdict.policy is Policy.PROPORTIONAL_FAIR
    assert verdict.confidence is Confidence.HIGH
    assert verdict.fairness_alpha == pytest.approx(alpha, rel=0.15)


@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS])
def test_detects_max_ci(channels: list[float]) -> None:
    verdict = _run(_config(SimulatedPolicy.MAX_CI, channels)).verdict
    assert verdict.policy is Policy.MAX_CI
    assert verdict.fairness_alpha == 0.0


@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS, IDENTICAL_CHANNELS])
def test_detects_blind_equal_throughput(channels: list[float]) -> None:
    verdict = _run(_config(SimulatedPolicy.BLIND_EQUAL_THROUGHPUT, channels)).verdict
    assert verdict.policy is Policy.BLIND_EQUAL_THROUGHPUT
    assert math.isinf(verdict.fairness_alpha)


def test_pf_with_identical_channels_is_pf_with_low_confidence() -> None:
    verdict = _run(_config(SimulatedPolicy.PROPORTIONAL_FAIR, IDENTICAL_CHANNELS)).verdict
    assert verdict.policy is Policy.PROPORTIONAL_FAIR
    assert verdict.confidence is Confidence.LOW
    assert math.isnan(verdict.fairness_alpha)


# --- Politiques hors catalogue : « Scheduler personnalisé » ---------------


@pytest.mark.parametrize("channels", [TWO_CHANNELS, THREE_CHANNELS])
@pytest.mark.parametrize(
    "policy",
    [
        SimulatedPolicy.RANDOM,
        SimulatedPolicy.LONGEST_QUEUE_FIRST,
        SimulatedPolicy.FREQUENCY_EQUAL_SHARE,
    ],
)
def test_unknown_policy_is_reported_as_custom(
    policy: SimulatedPolicy, channels: list[float]
) -> None:
    verdict = _run(_config(policy, channels)).verdict
    assert verdict.policy is Policy.CUSTOM
    assert verdict.confidence is Confidence.HIGH


def test_longest_queue_first_with_unequal_loads_is_custom() -> None:
    config = _config(
        SimulatedPolicy.LONGEST_QUEUE_FIRST,
        TWO_CHANNELS,
        ues=_ues(TWO_CHANNELS, is_first_intermittent=False, arrivals=[4_000, 2_500]),
        rlc_queue_bytes=50_000_000,
    )
    assert _run(config).verdict.policy is Policy.CUSTOM


def test_frequency_sharing_is_flagged_in_features() -> None:
    detection = _run(_config(SimulatedPolicy.FREQUENCY_EQUAL_SHARE, TWO_CHANNELS))
    assert detection.features.greedy_violation_rate > 0.9
    assert any("bande est partagée" in reason for reason in detection.verdict.reasons)


def test_custom_verdict_names_the_closest_known_model() -> None:
    verdict = _run(_config(SimulatedPolicy.LONGEST_QUEUE_FIRST, TWO_CHANNELS)).verdict
    assert verdict.closest_model
    assert 0.0 <= verdict.closest_accuracy < 1.0


@pytest.mark.parametrize("policy", [SimulatedPolicy.RANDOM, SimulatedPolicy.MAX_CI])
def test_identical_channels_lower_confidence_of_custom_verdict(policy: SimulatedPolicy) -> None:
    # Avec des canaux identiques, un Max C/I suit des fluctuations invisibles :
    # il ne se distingue pas d'une politique inconnue. Pas de certitude.
    verdict = _run(_config(policy, IDENTICAL_CHANNELS)).verdict
    assert verdict.policy is Policy.CUSTOM
    assert verdict.confidence is Confidence.LOW


# --- Cas limites ------------------------------------------------------------


def test_perfectly_alternating_pf_without_traffic_pause_is_not_high_confidence_rr() -> None:
    # Limite connue : deux UE PF identiques, sans bruit ni pause, alternent
    # exactement comme un RR. Le détecteur ne doit pas l'affirmer avec
    # certitude.
    config = _config(
        SimulatedPolicy.PROPORTIONAL_FAIR,
        IDENTICAL_CHANNELS,
        ues=_ues(IDENTICAL_CHANNELS, is_first_intermittent=False),
        channel_noise=0.0,
    )
    verdict = _run(config).verdict
    assert not (verdict.policy is Policy.ROUND_ROBIN and verdict.confidence is Confidence.HIGH)


def test_unsaturated_cell_is_inconclusive() -> None:
    light_ues = tuple(SimulatedUe(0x46 + i, 300, arrival_bytes_per_tti=500) for i in range(2))
    detection = _run(_config(SimulatedPolicy.ROUND_ROBIN, TWO_CHANNELS, ues=light_ues))
    assert detection.features.nof_contested_tti < MIN_CONTESTED_TTI
    assert detection.verdict.policy is Policy.INCONCLUSIVE


def test_single_ue_is_inconclusive() -> None:
    config = _config(
        SimulatedPolicy.PROPORTIONAL_FAIR, [300], ues=_ues([300], False), nof_tti=2_000
    )
    assert _run(config).verdict.policy is Policy.INCONCLUSIVE


def test_detect_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="Aucune nouvelle transmission"):
        detect([])


def test_pf_model_rejects_negative_alpha() -> None:
    timeline = build_timeline(
        list(parse_lines(simulate(_config(SimulatedPolicy.ROUND_ROBIN, TWO_CHANNELS))))
    )
    with pytest.raises(ValueError, match="alpha"):
        fit_pf_model(timeline, -1.0)


def test_simulator_rejects_invalid_noise() -> None:
    config = _config(SimulatedPolicy.ROUND_ROBIN, [300], ues=_ues([300]), channel_noise=1.5)
    with pytest.raises(ValueError, match="channel_noise"):
        list(simulate(config))
