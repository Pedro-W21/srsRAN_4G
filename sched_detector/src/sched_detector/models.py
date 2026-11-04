"""Rejeu des politiques connues sur une chronologie observée.

Pour chaque politique du catalogue, on reconstruit à chaque TTI disputé
l'UE qu'elle aurait servi en premier avec l'information disponible dans le
log, puis on compare au gagnant réellement observé. La **précision** d'un
modèle est la part de TTI disputés qu'il prédit correctement. Une politique
inconnue n'est bien prédite par aucun modèle : c'est ce qui permet de la
classer « personnalisée » au lieu de la forcer dans une case.

Modèles :

* **Round Robin** (``sched_time_rr.cc``) : le gagnant est une fonction fixe
  du couple (``tti_tx_dl mod N``, UE en attente). L'ordre de ``ue_db`` n'est
  pas connu, donc la fonction est apprise sur une moitié du log et évaluée
  sur l'autre (validation croisée à deux plis, pour ne pas récompenser le
  sur-apprentissage).
* **Famille PF** (``sched_time_pf.cc``) : priorité ``r / R^alpha``, avec R
  mise à jour comme ``save_dl_alloc`` (moyenne arithmétique sur 100
  échantillons puis exponentielle de pas 0,01). On rejoue la moyenne à partir
  des octets alloués à chaque UE, lisibles dans le log. Deux cas limites de
  la même formule sont des politiques classiques :

  - ``alpha = 0`` : priorité ``r`` seule → **Max C/I** ;
  - ``alpha → ∞`` : seul compte le plus petit R → **Blind Equal Throughput**
    (débit égal pour tous).

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from sched_detector.features import ContestedTti, Timeline

#: Pas de la moyenne exponentielle du PF de srsRAN (codé en dur).
PF_EXP_AVG_ALPHA: float = 0.01
#: TTI ignorés au début du rejeu PF : la moyenne rejouée part de zéro et
#: oublie cette condition initiale en ~5 constantes de temps (5 x 100 TTI).
PF_WARMUP_TTI: int = 500
#: Rapport minimal entre la meilleure et la deuxième priorité pour qu'un TTI
#: compte comme décisif. À l'équilibre d'un PF, les priorités des UE sont
#: presque égales (c'est le but de l'algorithme) et le gagnant se joue sur des
#: fluctuations de CQI invisibles dans le log : ces TTI ne permettent ni de
#: valider ni de rejeter le modèle. 1,2 dépasse l'effet d'un pas de CQI.
DECISIVE_PRIORITY_RATIO: float = 1.2
#: Coefficients d'équité essayés. ``inf`` = Blind Equal Throughput.
DEFAULT_ALPHA_GRID: tuple[float, ...] = (
    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, math.inf,
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class ModelFit:
    """Résultat du rejeu d'un modèle.

    Attributes:
        name: Nom lisible du modèle.
        accuracy: Part des TTI évalués où le gagnant prédit est le bon,
            dans [0, 1] (``nan`` si aucun TTI évaluable).
        nof_evaluated: Nombre de TTI disputés évalués.
        predictions: Gagnant prédit par TTI déplié (TTI évalués seulement).
        fairness_alpha: Coefficient d'équité pour la famille PF, ``nan``
            pour les autres modèles.
        excess_over_baseline: Pour le RR, gain normalisé par rapport à une
            prédiction qui ignore le numéro de TTI (même principe que le
            kappa de Cohen) ; ``nan`` pour les autres.
        decisive_accuracy: Précision sur les seuls TTI « décisifs », où le
            modèle préfère nettement un UE (voir
            :data:`DECISIVE_PRIORITY_RATIO`). Pour le RR, dont les décisions
            sont déterministes, elle vaut ``accuracy``.
        nof_decisive: Nombre de TTI décisifs évalués.
    """

    name: str
    accuracy: float
    nof_evaluated: int
    predictions: dict[int, int] = field(repr=False)
    fairness_alpha: float = math.nan
    excess_over_baseline: float = math.nan
    decisive_accuracy: float = math.nan
    nof_decisive: int = 0


def _modal(counter: Counter[int] | None) -> int | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def fit_round_robin(timeline: Timeline) -> ModelFit:
    """Évalue l'hypothèse « gagnant = f(tti_tx_dl mod N, UE en attente) ».

    Args:
        timeline: Chronologie observée.

    Returns:
        La précision en validation croisée et le gain par rapport à la
        prédiction sans numéro de TTI.
    """
    items = list(timeline.contested)
    half = len(items) // 2
    folds = ((items[:half], items[half:]), (items[half:], items[:half]))
    predictions: dict[int, int] = {}
    correct = 0
    correct_baseline = 0
    evaluated = 0
    for train, test in folds:
        by_key: dict[tuple[int, frozenset[int]], Counter[int]] = defaultdict(Counter)
        by_set: dict[frozenset[int], Counter[int]] = defaultdict(Counter)
        for item in train:
            residue = item.winner.tti_tx_dl % timeline.nof_ues
            by_key[(residue, item.backlogged_rntis)][item.winner.rnti] += 1
            by_set[item.backlogged_rntis][item.winner.rnti] += 1
        for item in test:
            residue = item.winner.tti_tx_dl % timeline.nof_ues
            baseline = _modal(by_set.get(item.backlogged_rntis))
            predicted = _modal(by_key.get((residue, item.backlogged_rntis)))
            if predicted is None:
                predicted = baseline
            if predicted is None:
                continue
            evaluated += 1
            predictions[item.winner.tti_unwrapped] = predicted
            correct += int(predicted == item.winner.rnti)
            correct_baseline += int(baseline == item.winner.rnti)
    if evaluated == 0:
        return ModelFit("Round Robin", math.nan, 0, predictions)
    accuracy = correct / evaluated
    baseline_accuracy = correct_baseline / evaluated
    excess = (
        0.0
        if baseline_accuracy >= 1.0
        else (accuracy - baseline_accuracy) / (1.0 - baseline_accuracy)
    )
    return ModelFit(
        name="Round Robin",
        accuracy=accuracy,
        nof_evaluated=evaluated,
        predictions=predictions,
        excess_over_baseline=excess,
        decisive_accuracy=accuracy,
        nof_decisive=evaluated,
    )


@dataclass(slots=True)
class _PfAverage:
    value: float = 0.0
    nof_samples: int = 0

    def update(self, allocated_bytes: float) -> None:
        # Copie de sched_time_pf::ue_ctxt::save_dl_alloc.
        if self.nof_samples < 1 / PF_EXP_AVG_ALPHA:
            self.value += (allocated_bytes - self.value) / (self.nof_samples + 1)
        else:
            self.value = (1 - PF_EXP_AVG_ALPHA) * self.value + PF_EXP_AVG_ALPHA * allocated_bytes
        self.nof_samples += 1


def _pf_priority(rate: float, average: float, alpha: float) -> float:
    """Priorité positive, comparable par rapport (``inf`` si R = 0)."""
    if alpha == 0.0:
        return rate
    if average == 0.0:
        return math.inf
    if math.isinf(alpha):
        # Limite alpha → ∞ : le classement ne dépend plus que de R (BET).
        return 1.0 / average
    return float(rate / average**alpha)


def _margin(best: float, second: float) -> float:
    if math.isinf(best):
        return 1.0 if math.isinf(second) else math.inf
    return math.inf if second == 0.0 else best / second


def fit_pf_model(timeline: Timeline, alpha: float, warmup_tti: int = PF_WARMUP_TTI) -> ModelFit:
    """Rejoue un PF temporel de coefficient ``alpha`` sur la chronologie.

    Le débit instantané ``r`` d'un UE est le dernier TBS par RBG observé
    pour lui avec des données restantes (MCS non bridé par le buffer) ; un
    facteur commun à tous les UE (nombre de RBG) ne change pas le classement.
    La moyenne R est rejouée pour chaque UE à partir de sa première
    apparition, avec 0 octet les TTI où il n'est pas servi, comme srsRAN le
    fait pour les UE présents dans la file.

    Args:
        timeline: Chronologie observée.
        alpha: Coefficient d'équité ; 0 = Max C/I, ``math.inf`` = BET.
        warmup_tti: TTI ignorés au début pour l'évaluation.

    Returns:
        La précision du modèle.

    Raises:
        ValueError: Si ``alpha`` est négatif.
    """
    if alpha < 0:
        raise ValueError(f"alpha doit être >= 0, reçu {alpha}")
    contested_by_tti: dict[int, ContestedTti] = {
        item.winner.tti_unwrapped: item for item in timeline.contested
    }
    averages: dict[int, _PfAverage] = {}
    last_rate: dict[int, float] = {}
    predictions: dict[int, int] = {}
    correct = 0
    decisive = 0
    decisive_correct = 0
    for tti in range(timeline.first_tti, timeline.last_tti + 1):
        item = contested_by_tti.get(tti)
        if (
            item is not None
            and tti - timeline.first_tti >= warmup_tti
            and all(r in last_rate for r in item.backlogged_rntis)
        ):
            ranked = sorted(
                ((_pf_priority(last_rate[r], averages[r].value, alpha), r)
                 for r in item.backlogged_rntis),
                reverse=True,
            )  # fmt: skip
            predicted = ranked[0][1]
            is_correct = predicted == item.winner.rnti
            predictions[tti] = predicted
            correct += int(is_correct)
            if _margin(ranked[0][0], ranked[1][0]) >= DECISIVE_PRIORITY_RATIO:
                decisive += 1
                decisive_correct += int(is_correct)
        allocated: dict[int, int] = defaultdict(int)
        for allocation in timeline.allocations_by_tti.get(tti, ()):
            allocated[allocation.rnti] += allocation.tbs_bytes
            averages.setdefault(allocation.rnti, _PfAverage())
            if allocation.buffer_after_bytes > 0 or allocation.rnti not in last_rate:
                last_rate[allocation.rnti] = allocation.tbs_bytes / allocation.nof_rbg
        for rnti, average in averages.items():
            average.update(allocated.get(rnti, 0))
    name = _pf_family_name(alpha)
    if not predictions:
        return ModelFit(name, math.nan, 0, predictions, fairness_alpha=alpha)
    return ModelFit(
        name=name,
        accuracy=correct / len(predictions),
        nof_evaluated=len(predictions),
        predictions=predictions,
        fairness_alpha=alpha,
        decisive_accuracy=decisive_correct / decisive if decisive else math.nan,
        nof_decisive=decisive,
    )


def _pf_family_name(alpha: float) -> str:
    if alpha == 0.0:
        return "Max C/I"
    if math.isinf(alpha):
        return "Blind Equal Throughput"
    return f"Proportional Fair (alpha={alpha:g})"


def fit_pf_family(
    timeline: Timeline, alphas: Sequence[float] = DEFAULT_ALPHA_GRID
) -> tuple[ModelFit, ...]:
    """Rejoue la famille PF pour chaque valeur de la grille de ``alpha``.

    Args:
        timeline: Chronologie observée.
        alphas: Coefficients à essayer.

    Returns:
        Un résultat par coefficient, dans l'ordre de ``alphas``.
    """
    return tuple(fit_pf_model(timeline, alpha) for alpha in alphas)


@dataclass(frozen=True, slots=True)
class Arbitration:
    """Comparaison de deux modèles sur les TTI où ils divergent.

    Attributes:
        nof_disagreements: TTI évalués par les deux où les prédictions
            diffèrent.
        first_model_correct: Parmi eux, TTI où le premier modèle avait raison.
        second_model_correct: Parmi eux, TTI où le second avait raison.
    """

    nof_disagreements: int
    first_model_correct: int
    second_model_correct: int


def arbitrate(first: ModelFit, second: ModelFit, timeline: Timeline) -> Arbitration:
    """Départage deux modèles là où seuls leurs choix diffèrent.

    Deux modèles peuvent tous deux bien prédire un log (un PF entre UE
    identiques alterne comme un RR). Seuls les TTI où ils prédisent des UE
    différents permettent de trancher ; ce sont typiquement les TTI qui
    suivent une reprise de trafic.
    """
    winners = {item.winner.tti_unwrapped: item.winner.rnti for item in timeline.contested}
    common = first.predictions.keys() & second.predictions.keys()
    disagreements = [tti for tti in common if first.predictions[tti] != second.predictions[tti]]
    return Arbitration(
        nof_disagreements=len(disagreements),
        first_model_correct=sum(first.predictions[t] == winners[t] for t in disagreements),
        second_model_correct=sum(second.predictions[t] == winners[t] for t in disagreements),
    )
