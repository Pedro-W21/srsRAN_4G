"""Décision en ensemble ouvert : politique connue ou « Scheduler personnalisé ».

Le classifieur ne choisit pas « la case la plus proche ». Une politique du
catalogue n'est retenue que si son modèle, rejoué sur le log, reproduit les
décisions observées. Si aucun modèle n'y parvient, le verdict est
« Scheduler personnalisé », avec le modèle connu le plus proche à titre
indicatif.

Catalogue : time_rr, time_pf (alpha estimé), Max C/I (PF alpha = 0),
Blind Equal Throughput (PF alpha → ∞).

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from sched_detector.features import SchedulerFeatures
from sched_detector.models import ModelFit

#: Nombre minimal de TTI disputés pour conclure (~0,5 s de saturation).
MIN_CONTESTED_TTI: int = 500
#: Au-delà, la bande est partagée entre UE non satisfaits : aucune politique
#: temporelle gloutonne de srsRAN ne fait cela.
MAX_GREEDY_VIOLATION_RATE: float = 0.10
#: Le RR est validé si le numéro de TTI explique le gagnant (gain normalisé)…
RR_MIN_EXCESS: float = 0.90
#: … et si sa précision en validation croisée dépasse ce seuil.
RR_MIN_ACCURACY: float = 0.95
#: Un modèle PF est validé si sa précision sur les TTI décisifs dépasse ce
#: seuil. Sur les logs simulés, le bon modèle obtient 1,00 et les mauvais
#: au plus 0,87.
PF_MIN_DECISIVE_ACCURACY: float = 0.95
#: Nombre minimal de TTI décisifs pour juger un modèle PF.
PF_MIN_DECISIVE_TTI: int = 80
#: Précision globale à partir de laquelle un modèle PF « explique aussi » un
#: log validé par le RR (ambiguïté à arbitrer).
PF_EXPLAINS_ACCURACY: float = 0.95
#: Désaccords RR / PF nécessaires pour trancher avec confiance.
MIN_DISAGREEMENTS_FOR_ARBITRATION: int = 30
#: Avance de précision globale que le meilleur alpha doit avoir sur les
#: autres alpha validés pour être retenu quand la plage validée est large.
ALPHA_PEAK_MARGIN: float = 0.05
#: Écart de canal au-dessous duquel un Max C/I devient imprévisible.
MIN_CHANNEL_SPREAD: float = 1.25


class Policy(enum.Enum):
    """Politiques reconnues."""

    ROUND_ROBIN = "time_rr (Round Robin)"
    PROPORTIONAL_FAIR = "time_pf (Proportional Fair)"
    MAX_CI = "Max C/I"
    BLIND_EQUAL_THROUGHPUT = "Blind Equal Throughput"
    CUSTOM = "Scheduler personnalisé"
    INCONCLUSIVE = "indéterminé"


class Confidence(enum.Enum):
    """Niveau de confiance du verdict."""

    HIGH = "élevée"
    LOW = "faible"
    NONE = "aucune"


@dataclass(frozen=True, slots=True)
class Verdict:
    """Résultat de la classification.

    Attributes:
        policy: Politique détectée.
        confidence: Niveau de confiance.
        fairness_alpha: Coefficient d'équité retenu pour la famille PF
            (0 pour Max C/I, ``inf`` pour BET), ``nan`` sinon ou s'il n'est
            pas identifiable.
        alpha_range: Plus petite et plus grande valeur de alpha validées
            (``nan`` si aucune).
        closest_model: Modèle connu qui reproduit le mieux le log, utile
            pour décrire un scheduler personnalisé.
        closest_accuracy: Précision globale de ce modèle, dans [0, 1].
        reasons: Justifications lisibles, dans l'ordre des règles.
    """

    policy: Policy
    confidence: Confidence
    fairness_alpha: float
    alpha_range: tuple[float, float]
    closest_model: str
    closest_accuracy: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    """Résultats du rejeu de tous les modèles sur un log.

    Attributes:
        round_robin: Rejeu du RR.
        pf_family: Rejeu de la famille PF, un résultat par alpha.
        rr_pf_disagreements: TTI où le RR et le meilleur PF divergent.
        rr_wins_disagreements: Parmi eux, TTI où le RR avait raison.
        pf_wins_disagreements: Parmi eux, TTI où le PF avait raison.
    """

    round_robin: ModelFit
    pf_family: tuple[ModelFit, ...]
    rr_pf_disagreements: int
    rr_wins_disagreements: int
    pf_wins_disagreements: int


def _no_alpha() -> tuple[float, float]:
    return (math.nan, math.nan)


def _closest(evidence: ModelEvidence) -> ModelFit:
    candidates = [evidence.round_robin, *evidence.pf_family]
    scored = [m for m in candidates if not math.isnan(m.accuracy)]
    if not scored:
        return evidence.round_robin
    return max(scored, key=lambda m: m.accuracy)


def best_pf_by_accuracy(pf_family: tuple[ModelFit, ...]) -> ModelFit | None:
    """Modèle PF de meilleure précision globale (``None`` si aucun évalué)."""
    scored = [m for m in pf_family if not math.isnan(m.accuracy)]
    return max(scored, key=lambda m: m.accuracy) if scored else None


def _validated_pf(pf_family: tuple[ModelFit, ...]) -> list[ModelFit]:
    return [
        m
        for m in pf_family
        if m.nof_decisive >= PF_MIN_DECISIVE_TTI and m.decisive_accuracy >= PF_MIN_DECISIVE_ACCURACY
    ]


def classify(features: SchedulerFeatures, evidence: ModelEvidence) -> Verdict:
    """Déduit la politique de scheduling.

    Ordre des règles :

    1. assez d'UE et de TTI disputés, sinon indéterminé ;
    2. bande partagée entre UE non satisfaits → personnalisé ;
    3. RR validé et aucun PF ne l'explique aussi → Round Robin ; si un PF
       l'explique aussi, arbitrage sur les TTI où ils divergent ;
    4. un modèle de la famille PF validé → PF, Max C/I ou BET selon alpha ;
    5. sinon → Scheduler personnalisé.

    Args:
        features: Métriques descriptives du log.
        evidence: Résultats du rejeu des modèles.

    Returns:
        Le verdict et ses justifications.
    """
    closest = _closest(evidence)
    if features.nof_ues < 2:
        return _inconclusive(
            "Un seul UE : toutes les politiques servent le même UE, rien à comparer.", closest
        )
    if features.nof_contested_tti < MIN_CONTESTED_TTI:
        return _inconclusive(
            f"Seulement {features.nof_contested_tti} TTI disputés (minimum "
            f"{MIN_CONTESTED_TTI}) : la cellule n'est pas saturée, le scheduler "
            "n'a presque jamais eu à choisir.",
            closest,
        )
    if features.greedy_violation_rate > MAX_GREEDY_VIOLATION_RATE:
        return _custom_verdict(
            features,
            closest,
            [
                f"Dans {features.greedy_violation_rate:.0%} des TTI disputés, plusieurs UE "
                "sont servis tout en gardant des données : la bande est partagée "
                "(politique fréquentielle ou quota par UE). Les politiques temporelles "
                "gloutonnes de srsRAN donnent tous les RBG restants au premier UE "
                "non satisfait."
            ],
            confidence=Confidence.HIGH,
        )

    rr = evidence.round_robin
    rr_ok = rr.excess_over_baseline >= RR_MIN_EXCESS and rr.accuracy >= RR_MIN_ACCURACY
    validated_pf = _validated_pf(evidence.pf_family)
    best_pf = best_pf_by_accuracy(evidence.pf_family)
    pf_explains = best_pf is not None and best_pf.accuracy >= PF_EXPLAINS_ACCURACY

    if rr_ok and not validated_pf and not pf_explains:
        return Verdict(
            Policy.ROUND_ROBIN,
            Confidence.HIGH,
            math.nan,
            _no_alpha(),
            rr.name,
            rr.accuracy,
            (
                _rr_reason(rr),
                "Aucun modèle de la famille PF ne reproduit ces décisions.",
            ),
        )
    if rr_ok:
        return _arbitrate_rr_and_pf(evidence, validated_pf, features)
    if validated_pf:
        return _pf_family_verdict(validated_pf, features, closest)
    return _custom_verdict(
        features,
        closest,
        [
            f"Le numéro de TTI n'explique pas le gagnant (gain normalisé "
            f"{rr.excess_over_baseline:.2f}, seuil {RR_MIN_EXCESS}) : ce n'est pas un RR.",
            _pf_rejection_reason(evidence.pf_family),
        ],
        confidence=Confidence.HIGH,
    )


def _inconclusive(reason: str, closest: ModelFit) -> Verdict:
    return Verdict(
        Policy.INCONCLUSIVE,
        Confidence.NONE,
        math.nan,
        _no_alpha(),
        closest.name,
        closest.accuracy,
        (reason,),
    )


def _rr_reason(rr: ModelFit) -> str:
    return (
        f"Le gagnant est fixé par (tti_tx_dl mod N, UE en attente) : précision "
        f"{rr.accuracy:.3f} en validation croisée, gain normalisé "
        f"{rr.excess_over_baseline:.2f}. C'est la rotation de sched_time_rr."
    )


def _pf_rejection_reason(pf_family: tuple[ModelFit, ...]) -> str:
    testable = [m for m in pf_family if m.nof_decisive >= PF_MIN_DECISIVE_TTI]
    if not testable:
        return (
            "Trop peu de TTI décisifs pour tester la famille PF (canaux et débits "
            "moyens trop proches)."
        )
    best = max(testable, key=lambda m: m.decisive_accuracy)
    return (
        f"Aucun modèle PF / Max C/I / BET ne reproduit les décisions nettes : au mieux "
        f"{best.decisive_accuracy:.0%} sur {best.nof_decisive} TTI décisifs ({best.name}), "
        f"seuil {PF_MIN_DECISIVE_ACCURACY:.0%}."
    )


def _arbitrate_rr_and_pf(
    evidence: ModelEvidence, validated_pf: list[ModelFit], features: SchedulerFeatures
) -> Verdict:
    rr = evidence.round_robin
    reasons = [_rr_reason(rr), "Un modèle de la famille PF reproduit aussi ces décisions."]
    disagreements = evidence.rr_pf_disagreements
    if disagreements >= MIN_DISAGREEMENTS_FOR_ARBITRATION:
        reasons.append(
            f"Sur les {disagreements} TTI où RR et PF divergent, le RR a raison "
            f"{evidence.rr_wins_disagreements} fois, le PF {evidence.pf_wins_disagreements} fois."
        )
        if evidence.pf_wins_disagreements > evidence.rr_wins_disagreements and validated_pf:
            return _pf_family_verdict(validated_pf, features, rr, tuple(reasons))
        return Verdict(
            Policy.ROUND_ROBIN,
            Confidence.HIGH,
            math.nan,
            _no_alpha(),
            rr.name,
            rr.accuracy,
            tuple(reasons),
        )
    reasons.append(
        f"Seulement {disagreements} TTI où les deux modèles divergent : un PF entre "
        "UE identiques alterne comme un RR. Relancer avec un UE en trafic "
        "intermittent pour trancher."
    )
    return Verdict(
        Policy.ROUND_ROBIN,
        Confidence.LOW,
        math.nan,
        _no_alpha(),
        rr.name,
        rr.accuracy,
        tuple(reasons),
    )


def _pf_family_verdict(
    validated: list[ModelFit],
    features: SchedulerFeatures,
    closest: ModelFit,
    previous_reasons: tuple[str, ...] = (),
) -> Verdict:
    best = max(validated, key=lambda m: (m.decisive_accuracy, m.accuracy))
    alphas = sorted(m.fairness_alpha for m in validated)
    alpha_range = (alphas[0], alphas[-1])
    reasons = list(previous_reasons)
    reasons.append(
        f"{best.name} reproduit {best.decisive_accuracy:.0%} des {best.nof_decisive} "
        f"décisions nettes (précision globale {best.accuracy:.0%} : à l'équilibre, les "
        "priorités sont presque égales et le CQI tranche)."
    )
    reasons.append(f"Valeurs de alpha compatibles : {_format_range(alpha_range)}.")

    is_narrow_range = alpha_range[0] == alpha_range[1] or alpha_range[1] <= 2 * max(
        alpha_range[0], 0.25
    )
    other_accuracies = [m.accuracy for m in validated if m is not best]
    has_clear_peak = (
        not other_accuracies or best.accuracy - max(other_accuracies) >= ALPHA_PEAK_MARGIN
    )
    identifiable = is_narrow_range or has_clear_peak
    if not identifiable:
        reasons.append(
            "Plage de alpha trop large : les canaux des UE sont trop proches pour "
            "situer la politique dans la famille PF. Dégrader le canal d'un UE."
        )
        return Verdict(
            Policy.PROPORTIONAL_FAIR,
            Confidence.LOW,
            math.nan,
            alpha_range,
            best.name,
            best.accuracy,
            tuple(reasons),
        )
    policy = Policy.PROPORTIONAL_FAIR
    if best.fairness_alpha == 0.0:
        policy = Policy.MAX_CI
    elif math.isinf(best.fairness_alpha):
        policy = Policy.BLIND_EQUAL_THROUGHPUT
    return Verdict(
        policy,
        Confidence.HIGH,
        best.fairness_alpha,
        alpha_range,
        best.name,
        best.accuracy,
        tuple(reasons),
    )


def _format_range(alpha_range: tuple[float, float]) -> str:
    low, high = alpha_range
    if low == high:
        return f"{low:g}"
    return f"de {low:g} à {high:g}"


def _custom_verdict(
    features: SchedulerFeatures,
    closest: ModelFit,
    reasons: list[str],
    confidence: Confidence,
) -> Verdict:
    spread = features.channel_spread
    if (
        confidence is Confidence.HIGH
        and features.greedy_violation_rate <= MAX_GREEDY_VIOLATION_RATE
        and (math.isnan(spread) or spread < MIN_CHANNEL_SPREAD)
    ):
        confidence = Confidence.LOW
        reasons.append(
            f"Canaux presque identiques (écart {spread:.2f}) : un Max C/I choisit alors "
            "au gré de fluctuations de CQI invisibles dans le log et ressemble à une "
            "politique inconnue. Dégrader le canal d'un UE pour confirmer."
        )
    return Verdict(
        Policy.CUSTOM,
        confidence,
        math.nan,
        _no_alpha(),
        closest.name,
        closest.accuracy,
        tuple(reasons),
    )
