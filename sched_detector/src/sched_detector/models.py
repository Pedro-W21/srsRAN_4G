"""Rejeu des politiques connues sur une chronologie observée.

Pour chaque politique du catalogue, on reconstruit à chaque TTI disputé
l'UE qu'elle aurait servi en premier avec l'information disponible dans le
log, puis on compare au gagnant réellement observé. La **précision** d'un
modèle est la part de TTI disputés qu'il prédit correctement. Une politique
inconnue n'est bien prédite par aucun modèle : c'est ce qui permet de la
classer « personnalisée » au lieu de la forcer dans une case.

Modèles :

* **Round Robin** (``sched_time_rr.cc``) : le gagnant est une fonction fixe
  du couple (``tti_tx mod N``, UE en attente), avec ``tti_tx_dl`` en DL et
  ``tti_tx_ul`` en UL (même modulo dans ``sched_ul_newtxs``). L'ordre de ``ue_db`` n'est
  pas connu, donc la fonction est apprise sur une moitié du log et évaluée
  sur l'autre (validation croisée à deux plis, pour ne pas récompenser le
  sur-apprentissage).
* **Famille PF** (``sched_time_pf.cc``) : priorité ``r / R^alpha``, avec R
  mise à jour comme ``save_dl_alloc`` / ``save_ul_alloc`` (moyenne arithmétique sur 100
  échantillons puis exponentielle de pas 0,01). On rejoue la moyenne à partir
  des octets alloués à chaque UE, lisibles dans le log. Deux cas limites de
  la même formule sont des politiques classiques :

  - ``alpha = 0`` : priorité ``r`` seule → **Max C/I** ;
  - ``alpha → ∞`` : seul compte le plus petit R → **Blind Equal Throughput**
    (débit égal pour tous).

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14 (voie montante : 2026-11-18)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
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
        excess_over_baseline: « Apport » du modèle : gain normalisé par
            rapport à la prédiction naïve (voir :func:`naive_accuracy`), sur
            les mêmes TTI. 1 = il explique toutes les décisions que la
            prédiction naïve rate, 0 = il ne fait pas mieux qu'elle. Pour le
            RR, la référence est apprise en validation croisée comme le
            modèle lui-même.
        decisive_accuracy: Précision sur les seuls TTI « décisifs », où le
            modèle préfère nettement un UE (voir
            :data:`DECISIVE_PRIORITY_RATIO`). Pour le RR, dont les décisions
            sont déterministes, elle vaut ``accuracy``.
        nof_decisive: Nombre de TTI décisifs évalués.
        judged_naive_accuracy: Précision de la prédiction naïve (désigner
            toujours le favori, voir :func:`naive_accuracy`) sur les mêmes
            TTI que ``decisive_accuracy``. C'est la référence qui rend
            ``decisive_accuracy`` lisible : un modèle qui ne fait pas mieux
            n'explique rien.
    """

    name: str
    accuracy: float
    nof_evaluated: int
    predictions: dict[int, int] = field(repr=False)
    fairness_alpha: float = math.nan
    excess_over_baseline: float = math.nan
    decisive_accuracy: float = math.nan
    nof_decisive: int = 0
    judged_naive_accuracy: float = math.nan


def _modal(counter: Counter[int] | None) -> int | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _excess(accuracy: float, baseline_accuracy: float) -> float:
    """Gain normalisé sur une référence (même construction que le kappa de Cohen).

    Returns:
        1 = le modèle explique toutes les décisions que la référence rate,
        0 = il ne fait pas mieux qu'elle, < 0 = il fait moins bien.
    """
    if baseline_accuracy >= 1.0:
        return 0.0
    return (accuracy - baseline_accuracy) / (1.0 - baseline_accuracy)


def _naive_predictor(timeline: Timeline) -> dict[frozenset[int], int]:
    """Gagnant le plus fréquent pour chaque ensemble d'UE en attente."""
    by_set: dict[frozenset[int], Counter[int]] = defaultdict(Counter)
    for item in timeline.contested:
        by_set[item.backlogged_rntis][item.winner.rnti] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in by_set.items()}


def naive_accuracy(timeline: Timeline, ttis: Collection[int] | None = None) -> float:
    """Précision de la prédiction naïve, référence commune à tous les modèles.

    La prédiction naïve ignore tout ce qui fait une politique (numéro de TTI,
    canal, historique) : pour chaque ensemble d'UE en attente, elle désigne
    toujours l'UE qui y gagne le plus souvent. Quand un UE gagne 90 % des
    TTI, elle a déjà raison 90 % du temps : un modèle n'apporte quelque
    chose que s'il fait mieux qu'elle.

    Args:
        timeline: Chronologie observée.
        ttis: TTI (dépliés) sur lesquels l'évaluer ; par défaut tous les TTI
            disputés.

    Returns:
        La précision dans [0, 1], ``nan`` si aucun TTI n'est évaluable.
    """
    predictor = _naive_predictor(timeline)
    items = [
        item for item in timeline.contested if ttis is None or item.winner.tti_unwrapped in ttis
    ]
    if not items:
        return math.nan
    correct = sum(predictor[item.backlogged_rntis] == item.winner.rnti for item in items)
    return correct / len(items)


def fit_round_robin(timeline: Timeline) -> ModelFit:
    """Évalue l'hypothèse « gagnant = f(tti_tx mod N, UE en attente) ».

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
            residue = item.winner.tti % timeline.nof_ues
            by_key[(residue, item.backlogged_rntis)][item.winner.rnti] += 1
            by_set[item.backlogged_rntis][item.winner.rnti] += 1
        for item in test:
            residue = item.winner.tti % timeline.nof_ues
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
    excess = _excess(accuracy, correct_baseline / evaluated)
    return ModelFit(
        name="Round Robin",
        accuracy=accuracy,
        nof_evaluated=evaluated,
        predictions=predictions,
        excess_over_baseline=excess,
        decisive_accuracy=accuracy,
        nof_decisive=evaluated,
        judged_naive_accuracy=naive_accuracy(timeline, predictions.keys()),
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

    Le débit instantané ``r`` d'un UE est le dernier TBS par ressource (RBG
    en DL, PRB en UL) observé
    pour lui avec des données restantes (MCS non bridé par le buffer) ; un
    facteur commun à tous les UE (taille de la bande) ne change pas le classement.
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
    decisive_ttis: list[int] = []
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
                decisive_ttis.append(tti)
                decisive_correct += int(is_correct)
        allocated: dict[int, int] = defaultdict(int)
        for allocation in timeline.allocations_by_tti.get(tti, ()):
            allocated[allocation.rnti] += allocation.tbs_bytes
            averages.setdefault(allocation.rnti, _PfAverage())
            if allocation.pending_after_bytes > 0 or allocation.rnti not in last_rate:
                last_rate[allocation.rnti] = allocation.tbs_bytes / allocation.nof_resources
        for rnti, average in averages.items():
            average.update(allocated.get(rnti, 0))
    name = _pf_family_name(alpha)
    if not predictions:
        return ModelFit(name, math.nan, 0, predictions, fairness_alpha=alpha)
    accuracy = correct / len(predictions)
    return ModelFit(
        name=name,
        accuracy=accuracy,
        nof_evaluated=len(predictions),
        predictions=predictions,
        fairness_alpha=alpha,
        excess_over_baseline=_excess(accuracy, naive_accuracy(timeline, predictions.keys())),
        decisive_accuracy=decisive_correct / len(decisive_ttis) if decisive_ttis else math.nan,
        nof_decisive=len(decisive_ttis),
        judged_naive_accuracy=naive_accuracy(timeline, frozenset(decisive_ttis)),
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
