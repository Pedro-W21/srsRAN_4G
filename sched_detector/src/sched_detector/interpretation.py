"""Traduction des résultats en chiffres lisibles sans piège.

Le chiffre le plus trompeur du rejeu est la précision brute : si un UE gagne
90 % des TTI disputés, tout modèle qui le désigne toujours a « raison 90 fois
sur 100 » sans rien expliquer. Chaque score est donc présenté à côté de sa
référence :

* **a raison** : sur les décisions où le modèle a un avis net, combien de
  fois sur 100 il désigne le même UE que le scheduler ;
* **favori** : combien de fois sur 100 on aurait raison sur ces mêmes
  décisions en désignant toujours l'UE qui gagne le plus souvent (la
  prédiction naïve, :func:`~sched_detector.models.naive_accuracy`) ;
* **gain** : la différence, en points. Un modèle ne ressemble au scheduler
  que s'il a raison au moins 95 fois sur 100 **et** fait nettement mieux que
  le favori.

Pour l'équité, l'indice de Jain J est aussi donné sous la forme ``J x n`` :
le nombre d'UE qui, servis à égalité, donneraient la même équité.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-11-18
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence

from sched_detector.classifier import (
    CLOSEST_MODEL_MIN_EXCESS,
    MIN_CHANNEL_SPREAD,
    PF_MIN_DECISIVE_ACCURACY,
    PF_MIN_DECISIVE_TTI,
    RR_MIN_ACCURACY,
    Confidence,
    Policy,
    closest_model,
    is_pf_testable,
    is_pf_validated,
    is_round_robin_validated,
)
from sched_detector.detector import Detection
from sched_detector.models import ModelFit

#: Part des octets en dessous de laquelle un UE est considéré comme inactif
#: dans la voie analysée (signalisation seulement).
IDLE_UE_MAX_BYTE_SHARE: float = 0.01


class ModelStatus(enum.Enum):
    """Compatibilité d'un modèle connu avec le log."""

    VALIDATED = "validé"
    REJECTED = "rejeté"
    UNTESTABLE = "non testable"


def model_status(model: ModelFit) -> ModelStatus:
    """Statut d'un modèle rejoué, avec les mêmes seuils que le classifieur.

    Args:
        model: Résultat du rejeu (RR ou famille PF).

    Returns:
        Validé, rejeté, ou non testable (trop peu de décisions pour juger).
    """
    if model.nof_evaluated == 0:
        return ModelStatus.UNTESTABLE
    if math.isnan(model.fairness_alpha):
        return ModelStatus.VALIDATED if is_round_robin_validated(model) else ModelStatus.REJECTED
    if not is_pf_testable(model):
        return ModelStatus.UNTESTABLE
    return ModelStatus.VALIDATED if is_pf_validated(model) else ModelStatus.REJECTED


def gain_points(model: ModelFit) -> float:
    """Écart, en points de pourcentage, entre le modèle et le favori.

    Calculé sur les décisions jugées (où le modèle a un avis net) :
    ``100 x (a raison - favori)``. ``nan`` si rien n'a été jugé.
    """
    return 100.0 * (model.decisive_accuracy - model.judged_naive_accuracy)


def format_percent(value: float) -> str:
    """Pourcentage arrondi, ``-`` si inconnu."""
    return "-" if math.isnan(value) else f"{value:.0%}"


def format_points(points: float) -> str:
    """Gain en points avec son signe (``+0 pt``, ``-80 pts``), ``-`` si inconnu."""
    if math.isnan(points):
        return "-"
    rounded = round(points)
    unit = "pt" if abs(rounded) <= 1 else "pts"
    return f"{rounded:+d} {unit}"


def status_reason(model: ModelFit) -> str:
    """Verdict d'un modèle avec sa raison, en quelques mots."""
    status = model_status(model)
    if status is ModelStatus.VALIDATED:
        return "validé"
    if status is ModelStatus.UNTESTABLE:
        return f"non testable : {_decisions(model.nof_decisive)}, il en faut {PF_MIN_DECISIVE_TTI}"
    if gain_points(model) <= 0.5:
        return "rejeté : pas mieux que le favori"
    threshold = RR_MIN_ACCURACY if math.isnan(model.fairness_alpha) else PF_MIN_DECISIVE_ACCURACY
    if model.decisive_accuracy < threshold:
        return f"rejeté : a raison moins de {threshold:.0%} du temps"
    return "rejeté : pas assez mieux que le favori"


def _decisions(count: int) -> str:
    if count == 0:
        return "aucune décision nette"
    return f"{count} décision nette" if count == 1 else f"{count} décisions nettes"


def favourite_rnti(detection: Detection) -> int | None:
    """UE qui gagne le plus de TTI disputés (``None`` sans TTI disputé)."""
    features = detection.features
    if features.nof_contested_tti == 0:
        return None
    return max(features.ues, key=lambda ue: ue.contested_wins).rnti


def describe_closest_model(detection: Detection) -> str:
    """Ligne « Modèle connu le plus proche » d'un verdict personnalisé.

    Le modèle n'est cité que s'il fait réellement mieux que le favori ;
    sinon le rapport le dit, pour qu'une précision élevée due au seul
    déséquilibre entre UE ne passe pas pour une ressemblance.
    """
    closest = closest_model(detection.evidence)
    if not math.isnan(closest.excess_over_baseline) and (
        closest.excess_over_baseline >= CLOSEST_MODEL_MIN_EXCESS
    ):
        return (
            f"Modèle connu le plus proche : {closest.name} (a raison "
            f"{format_percent(closest.decisive_accuracy)}, favori "
            f"{format_percent(closest.judged_naive_accuracy)}, gain "
            f"{format_points(gain_points(closest))})"
        )
    return (
        "Modèle connu le plus proche : aucun (aucun ne fait mieux que désigner toujours le favori)"
    )


def effective_ues(jain: float, nof_ues: int) -> str:
    """Indice de Jain traduit en nombre d'UE servis à égalité (``J x n``)."""
    if math.isnan(jain):
        return ""
    return f"≈ {jain * nof_ues:.1f} UE sur {nof_ues} servis à égalité"


def _headline(detection: Detection) -> str:
    verdict = detection.verdict
    policy = verdict.policy
    if policy is Policy.ROUND_ROBIN:
        return "Rotation sur le numéro de TTI : c'est un Round Robin."
    if policy is Policy.PROPORTIONAL_FAIR:
        alpha = (
            "alpha non identifiable"
            if math.isnan(verdict.fairness_alpha)
            else f"alpha ≈ {verdict.fairness_alpha:g}"
        )
        return f"Priorité r / R^alpha : c'est un Proportional Fair ({alpha})."
    if policy is Policy.MAX_CI:
        return "Le meilleur canal passe toujours d'abord : c'est un Max C/I."
    if policy is Policy.BLIND_EQUAL_THROUGHPUT:
        return "Le moins bien servi passe d'abord : c'est un Blind Equal Throughput."
    if policy is Policy.CUSTOM:
        return "Aucune politique connue ne reproduit ses décisions : politique personnalisée."
    return f"Impossible de conclure. {verdict.reasons[0]}"


def _shares(detection: Detection) -> str | None:
    """Part des TTI disputés gagnés par chaque UE, et canaux."""
    features = detection.features
    if features.nof_contested_tti == 0:
        return None
    ranked = sorted(features.ues, key=lambda ue: ue.contested_wins, reverse=True)
    shares = ", ".join(
        f"0x{ue.rnti:x} {ue.contested_wins / features.nof_contested_tti:.0%}" for ue in ranked
    )
    line = f"Servi en premier quand plusieurs UE attendent : {shares}"
    unit = features.direction.resource_unit
    rates = [ue.bytes_per_resource for ue in features.ues if not math.isnan(ue.bytes_per_resource)]
    spread = features.channel_spread
    if len(rates) >= 2 and not math.isnan(spread):
        kind = "équivalents" if spread < MIN_CHANNEL_SPREAD else "différents"
        line += f" (canaux {kind} : {min(rates):.1f} à {max(rates):.1f} octets/{unit})"
    return line + "."


def _model_line(label: str, models: Sequence[ModelFit], favourite: str) -> str:
    """Une ligne chiffrée par politique connue (le meilleur modèle de la famille)."""
    validated = [m for m in models if model_status(m) is ModelStatus.VALIDATED]
    judged = [m for m in models if model_status(m) is not ModelStatus.UNTESTABLE]
    if validated:
        best = max(validated, key=lambda m: (m.decisive_accuracy, m.accuracy))
        verdict = "oui"
    elif judged:
        best = max(judged, key=lambda m: m.decisive_accuracy)
        verdict = "non"
    else:
        most = max(models, key=lambda m: m.nof_decisive)
        return (
            f"{label} : non testable ({_decisions(most.nof_decisive)}, il en faut "
            f"{PF_MIN_DECISIVE_TTI})."
        )
    name = f" (meilleur modèle : {best.name})" if len(models) > 1 else ""
    return (
        f"{label} : {verdict}{name}. A raison {format_percent(best.decisive_accuracy)} du "
        f"temps, contre {format_percent(best.judged_naive_accuracy)} en désignant toujours "
        f"{favourite} : gain {format_points(gain_points(best))}."
    )


def _confidence_line(detection: Detection) -> str | None:
    verdict = detection.verdict
    if verdict.confidence is not Confidence.LOW:
        return None
    if verdict.policy is Policy.CUSTOM:
        return (
            "Confiance faible : canaux équivalents, donc Max C/I non testable (il départage "
            "les égalités toujours dans le même sens). Pour trancher : dégrader le canal "
            "d'un UE ou faire un essai témoin en Max C/I."
        )
    if verdict.policy is Policy.ROUND_ROBIN:
        return (
            "Confiance faible : un RR et un PF expliquent tous deux le log ; un UE en "
            "trafic intermittent permet de trancher."
        )
    return "Confiance faible : alpha non identifiable, canaux des UE trop proches."


def _idle_ue_hint(detection: Detection, ignored_rntis: Sequence[int]) -> str | None:
    fairness = detection.fairness
    if detection.features.nof_contested_tti == 0 or fairness.nof_contention_tti > 0:
        return None
    total = sum(ue.total_bytes for ue in fairness.ues)
    idle = [
        ue.rnti for ue in fairness.ues if total and ue.total_bytes / total < IDLE_UE_MAX_BYTE_SHARE
    ]
    if not idle:
        return None
    share = sum(ue.total_bytes for ue in fairness.ues if ue.rnti in idle) / total
    options = " ".join(f"--ignore-rnti 0x{rnti:x}" for rnti in [*ignored_rntis, *idle])
    names = ", ".join(f"0x{rnti:x}" for rnti in idle)
    return (
        f"Astuce : {names} = {share:.1%} des octets. Il n'attend jamais en même temps que "
        "les autres : 0 TTI de « pleine contention », d'où les indices de Jain vides. "
        f"Relancer avec {options}."
    )


def summarize(detection: Detection, ignored_rntis: Sequence[int] = ()) -> list[str]:
    """Rédige la section « En bref » du rapport.

    Args:
        detection: Résultat de :func:`~sched_detector.detector.detect`.
        ignored_rntis: UE écartés de l'analyse avec ``--ignore-rnti``.

    Returns:
        Les lignes de la section, sans indentation ni puces.
    """
    lines = [_headline(detection)]
    shares = _shares(detection)
    if shares is not None:
        lines.append(shares)
    favourite = favourite_rnti(detection)
    if favourite is not None:
        label = f"0x{favourite:x}"
        family = detection.evidence.pf_family
        lines.append(_model_line("Round Robin", [detection.evidence.round_robin], label))
        lines.append(
            _model_line(
                "Proportional Fair / BET", [m for m in family if m.fairness_alpha > 0], label
            )
        )
        lines.append(_model_line("Max C/I", [m for m in family if m.fairness_alpha == 0], label))
    for optional in (_confidence_line(detection), _idle_ue_hint(detection, ignored_rntis)):
        if optional is not None:
            lines.append(optional)
    return lines
