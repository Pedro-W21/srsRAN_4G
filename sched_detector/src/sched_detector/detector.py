"""Point d'entrée programmatique : allocations → métriques, modèles, verdict.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14 (voie montante : 2026-11-18)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sched_detector.classifier import ModelEvidence, Verdict, best_pf_by_accuracy, classify
from sched_detector.fairness import DEFAULT_JAIN_WINDOW_TTI, FairnessReport, compute_fairness
from sched_detector.features import SchedulerFeatures, build_timeline, extract_features
from sched_detector.log_parser import Allocation, Direction
from sched_detector.models import arbitrate, fit_pf_family, fit_round_robin, naive_accuracy


@dataclass(frozen=True, slots=True)
class Detection:
    """Résultat complet d'une analyse.

    Attributes:
        features: Métriques descriptives.
        evidence: Rejeu des modèles connus.
        verdict: Politique détectée et justification.
        fairness: Indices d'équité de Jain.
    """

    features: SchedulerFeatures
    evidence: ModelEvidence
    verdict: Verdict
    fairness: FairnessReport


def detect(
    allocations: Sequence[Allocation],
    nof_ues: int | None = None,
    jain_window_tti: int = DEFAULT_JAIN_WINDOW_TTI,
) -> Detection:
    """Analyse les allocations d'une porteuse, pour une voie (DL ou UL).

    Args:
        allocations: Allocations d'une seule voie lues dans le log (voir
            :func:`sched_detector.log_parser.read_log`).
        nof_ues: Nombre d'UE connectés, si certains n'apparaissent pas dans
            le log.
        jain_window_tti: Fenêtre de l'indice de Jain à court terme, en TTI.

    Returns:
        Métriques, résultats des modèles, verdict et indices d'équité.

    Raises:
        ValueError: Si le log ne contient aucune nouvelle transmission,
            ou si ``jain_window_tti < 1``.
    """
    timeline = build_timeline(allocations, nof_ues)
    fairness = compute_fairness(timeline, window_tti=jain_window_tti)
    features = extract_features(timeline)
    round_robin = fit_round_robin(timeline)
    pf_family = fit_pf_family(timeline) if timeline.contested else ()
    best_pf = best_pf_by_accuracy(pf_family)
    if best_pf is None:
        disagreements, rr_wins, pf_wins = 0, 0, 0
    else:
        arbitration = arbitrate(round_robin, best_pf, timeline)
        disagreements = arbitration.nof_disagreements
        rr_wins = arbitration.first_model_correct
        pf_wins = arbitration.second_model_correct
    evidence = ModelEvidence(
        round_robin,
        pf_family,
        disagreements,
        rr_wins,
        pf_wins,
        naive_accuracy=naive_accuracy(timeline),
    )
    return Detection(features, evidence, classify(features, evidence), fairness)


@dataclass(frozen=True, slots=True)
class DirectionChoice:
    """Voie retenue par :func:`choose_direction`.

    Attributes:
        direction: Voie à analyser.
        contested_tti: TTI disputés trouvés dans chaque voie présente dans
            le log (0 si la voie a des allocations mais aucune contention).
    """

    direction: Direction
    contested_tti: dict[Direction, int]


def choose_direction(
    allocations: Mapping[Direction, Sequence[Allocation]], nof_ues: int | None = None
) -> DirectionChoice:
    """Choisit la voie où le scheduler a réellement eu à arbitrer.

    La politique ne se révèle que dans les TTI disputés. Selon le sens du
    trafic de test (``iperf3`` lancé côté réseau ou côté UE), la contention
    est en DL ou en UL : on retient la voie qui en a le plus (le DL à
    égalité, pour rester compatible avec la version DL seule).

    Args:
        allocations: Allocations de chaque voie (voir
            :func:`sched_detector.log_parser.read_log_all_directions`).
        nof_ues: Nombre d'UE connectés, comme pour :func:`detect`.

    Returns:
        La voie retenue et le nombre de TTI disputés de chaque voie.

    Raises:
        ValueError: Si aucune voie ne contient de nouvelle transmission.
    """
    contested: dict[Direction, int] = {}
    for direction in Direction:
        items = allocations.get(direction, ())
        if any(not a.is_retransmission and a.nof_resources > 0 for a in items):
            contested[direction] = len(build_timeline(items, nof_ues).contested)
    if not contested:
        raise ValueError(
            "Aucune nouvelle transmission DL ni UL dans le log : vérifier mac_level=info "
            "et la porteuse (--cc)"
        )
    best = max(contested, key=lambda d: (contested[d], d is Direction.DL))
    return DirectionChoice(best, contested)
