"""Point d'entrée programmatique : allocations → métriques, modèles, verdict.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sched_detector.classifier import ModelEvidence, Verdict, best_pf_by_accuracy, classify
from sched_detector.features import SchedulerFeatures, build_timeline, extract_features
from sched_detector.log_parser import DlAllocation
from sched_detector.models import arbitrate, fit_pf_family, fit_round_robin


@dataclass(frozen=True, slots=True)
class Detection:
    """Résultat complet d'une analyse.

    Attributes:
        features: Métriques descriptives.
        evidence: Rejeu des modèles connus.
        verdict: Politique détectée et justification.
    """

    features: SchedulerFeatures
    evidence: ModelEvidence
    verdict: Verdict


def detect(allocations: Sequence[DlAllocation], nof_ues: int | None = None) -> Detection:
    """Analyse les allocations DL d'une porteuse.

    Args:
        allocations: Allocations lues dans le log (voir
            :func:`sched_detector.log_parser.read_log`).
        nof_ues: Nombre d'UE connectés, si certains n'apparaissent pas dans
            le log.

    Returns:
        Métriques, résultats des modèles et verdict.

    Raises:
        ValueError: Si le log ne contient aucune nouvelle transmission DL.
    """
    timeline = build_timeline(allocations, nof_ues)
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
    evidence = ModelEvidence(round_robin, pf_family, disagreements, rr_wins, pf_wins)
    return Detection(features, evidence, classify(features, evidence))
