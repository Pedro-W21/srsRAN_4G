"""Indice d'équité de Jain calculé sur les allocations DL ou UL du log.

Définition (Jain, Chiu et Hawe, DEC-TR-301, 1984) pour n UE de débits x_i ::

    J(x) = (Σ x_i)² / (n · Σ x_i²)

J vaut 1 quand tous les UE reçoivent la même chose et 1/n quand un seul UE
reçoit tout. L'indice ne dépend pas de l'unité (J(c·x) = J(x)) : on peut
le calculer directement sur des octets ou des ressources cumulées, sans diviser par
la durée.

Quatre indices sont calculés, parce qu'« équitable » n'a pas le même sens
selon la grandeur et la période observées :

* **Débit global** : octets de tout le log. Dépend du trafic offert : un UE
  intermittent reçoit moins parce qu'il demande moins, pas parce que la
  politique le défavorise. C'est l'équité vue par l'utilisateur.
* **Débit en pleine contention** : octets des seuls TTI disputés en régime
  établi où tous les UE avaient des données (même sélection que la colonne
  « part (régime établi) »). À demande égale, c'est l'équité que la
  politique produit elle-même.
* **Ressources en pleine contention** : RBG (DL) ou PRB (UL) au lieu
  d'octets. Sépare la politique du canal : un RR donne J ≈ 1 même avec des canaux
  différents, alors que son J en octets baisse avec l'écart de CQI.
* **Court terme** : moyenne de J (en octets) sur des fenêtres de W TTI de
  pleine contention consécutifs. L'allocation gloutonne de srsRAN sert un
  seul UE par TTI à saturation : J sur une fenêtre de 1 TTI vaut toujours
  1/n. La fenêtre montre à partir de quelle échelle de temps la politique
  devient équitable.

Repères théoriques en pleine contention (PF temporel, part des TTI
``f_i ∝ r_i^(1/alpha - 1)``, voir ``features.py``) :

=====================  ==============  ======================
Politique              J ressources    J sur les octets
=====================  ==============  ======================
time_rr                ≈ 1             < 1 si canaux différents
time_pf, alpha = 1     ≈ 1             < 1 si canaux différents
time_pf, alpha = 2     < 1             plus proche de 1 qu'en RR
Blind Equal Throughput < 1             ≈ 1
Max C/I                → 1/n           → 1/n
=====================  ==============  ======================

Limite : le log donne le TBS **alloué**, pas les ACK. Un bloc perdu (NACK)
compte quand même ; il s'agit d'un débit alloué, pas d'un goodput.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-11-04 (voie montante : 2026-11-18)
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from sched_detector.features import (
    DEFAULT_TRANSIENT_EXCLUSION_TTI,
    ContestedTti,
    Timeline,
    full_contention_ttis,
)

#: Fenêtre par défaut de l'indice à court terme, en TTI : la constante de
#: temps de la moyenne PF de srsRAN (beta = 0,01 dans ``sched_time_pf.cc``).
DEFAULT_JAIN_WINDOW_TTI: int = 100

#: Durée d'un TTI LTE, en secondes (36.211 §4.1).
_TTI_DURATION_S: float = 1e-3


def jain_index(values: Iterable[float]) -> float:
    """Indice d'équité de Jain.

    Args:
        values: Quantités reçues par chaque UE (débits, octets, RBG, PRB...),
            toutes positives ou nulles.

    Returns:
        J dans [1/n, 1], ou ``nan`` s'il y a moins de deux valeurs ou si
        toutes sont nulles (rien n'a été distribué : l'équité n'est pas
        définie).

    Raises:
        ValueError: Si une valeur est négative ou non finie.
    """
    samples = list(values)
    if any(not math.isfinite(x) or x < 0.0 for x in samples):
        raise ValueError(f"Les valeurs doivent être finies et positives : {samples}")
    if len(samples) < 2:
        return math.nan
    sum_of_squares = math.fsum(x * x for x in samples)
    if sum_of_squares == 0.0:
        return math.nan
    total = math.fsum(samples)
    return total * total / (len(samples) * sum_of_squares)


@dataclass(frozen=True, slots=True)
class UeFairness:
    """Ressources reçues par un UE (nouvelles transmissions uniquement).

    Attributes:
        rnti: Identifiant radio.
        total_bytes: Octets alloués sur tout le log.
        mean_throughput_mbps: ``total_bytes`` rapporté à la durée du log,
            en Mbit/s.
        contention_bytes: Octets alloués pendant les TTI de pleine
            contention.
        contention_resources: RBG (DL) ou PRB (UL) alloués pendant les TTI
            de pleine contention.
    """

    rnti: int
    total_bytes: int
    mean_throughput_mbps: float
    contention_bytes: int
    contention_resources: int


@dataclass(frozen=True, slots=True)
class FairnessReport:
    """Indices de Jain d'une chronologie.

    Attributes:
        jain_throughput: J sur les octets de tout le log.
        jain_contention_throughput: J sur les octets des TTI de pleine
            contention.
        jain_contention_resources: J sur les ressources (RBG ou PRB) des TTI
            de pleine contention.
        jain_short_term: Moyenne de J (octets) sur les fenêtres de
            ``window_tti`` TTI de pleine contention consécutifs.
        jain_short_term_min: Plus petite valeur de J parmi ces fenêtres.
        window_tti: Taille de la fenêtre de l'indice à court terme.
        nof_windows: Nombre de fenêtres complètes utilisées.
        nof_contention_tti: TTI de pleine contention retenus.
        ues: Ressources reçues par UE, triées par RNTI.
    """

    jain_throughput: float
    jain_contention_throughput: float
    jain_contention_resources: float
    jain_short_term: float
    jain_short_term_min: float
    window_tti: int
    nof_windows: int
    nof_contention_tti: int
    ues: tuple[UeFairness, ...] = field(default_factory=tuple)


def _contiguous_runs(
    items: Sequence[ContestedTti], allocation_ttis: Sequence[int]
) -> list[list[ContestedTti]]:
    """Découpe une suite chronologique de TTI de pleine contention en blocs sans trou.

    Deux TTI de pleine contention restent dans le même bloc si aucun TTI
    **avec allocation** ne les sépare. Un TTI sans aucune allocation n'est
    pas un trou : personne n'a rien reçu, il ne change pas le partage. C'est
    fréquent en UL, où les grants ne tombent pas à chaque TTI (HARQ
    synchrone, PHICH, PDCCH) ; les exiger strictement consécutifs empêcherait
    toute fenêtre de se former.

    Args:
        items: TTI de pleine contention, dans l'ordre chronologique.
        allocation_ttis: Tous les TTI (dépliés) qui ont au moins une
            nouvelle transmission, triés.
    """
    runs: list[list[ContestedTti]] = []
    for item in items:
        tti = item.winner.tti_unwrapped
        if runs:
            previous = runs[-1][-1].winner.tti_unwrapped
            between = bisect.bisect_left(allocation_ttis, tti) - bisect.bisect_right(
                allocation_ttis, previous
            )
            if between == 0:
                runs[-1].append(item)
                continue
        runs.append([item])
    return runs


def _short_term_indices(
    timeline: Timeline, contention: Sequence[ContestedTti], window_tti: int
) -> list[float]:
    """J (octets) de chaque fenêtre complète de ``window_tti`` TTI consécutifs.

    Une fenêtre ne chevauche jamais un trou (TTI avec allocation mais sans
    pleine contention) : sinon elle mélangerait des TTI où un UE n'avait rien
    à recevoir. ``window_tti`` compte les TTI de pleine contention, pas les
    TTI vides intercalés.
    """
    indices: list[float] = []
    allocation_ttis = sorted(timeline.allocations_by_tti)
    for run in _contiguous_runs(contention, allocation_ttis):
        for start in range(0, len(run) - window_tti + 1, window_tti):
            received = dict.fromkeys(timeline.rntis, 0)
            for item in run[start : start + window_tti]:
                for allocation in timeline.allocations_by_tti[item.winner.tti_unwrapped]:
                    received[allocation.rnti] += allocation.tbs_bytes
            index = jain_index(float(v) for v in received.values())
            if not math.isnan(index):
                indices.append(index)
    return indices


def compute_fairness(
    timeline: Timeline,
    window_tti: int = DEFAULT_JAIN_WINDOW_TTI,
    transient_exclusion_tti: int = DEFAULT_TRANSIENT_EXCLUSION_TTI,
) -> FairnessReport:
    """Calcule les indices de Jain d'une chronologie.

    Les retransmissions HARQ sont exclues (déjà écartées par
    :func:`~sched_detector.features.build_timeline`) : elles renvoient des
    octets déjà comptés à la première transmission.

    Seuls les UE vus dans le log sont comptés. Un UE connecté qui n'a jamais
    reçu de données (``--nof-ues`` supérieur au nombre de RNTI) n'a, dans le
    log, aucune demande visible : l'inclure avec un débit nul ferait baisser
    J sans que la politique y soit pour rien.

    Args:
        timeline: Sortie de :func:`~sched_detector.features.build_timeline`.
        window_tti: Taille des fenêtres de l'indice à court terme, en TTI.
        transient_exclusion_tti: TTI ignorés après chaque reprise de trafic.

    Returns:
        Les indices et le détail par UE.

    Raises:
        ValueError: Si ``window_tti < 1``.
    """
    if window_tti < 1:
        raise ValueError(f"window_tti doit être >= 1, reçu {window_tti}")

    total_bytes: dict[int, int] = defaultdict(int)
    for allocations in timeline.allocations_by_tti.values():
        for allocation in allocations:
            total_bytes[allocation.rnti] += allocation.tbs_bytes

    contention = full_contention_ttis(timeline, transient_exclusion_tti)
    contention_bytes: dict[int, int] = defaultdict(int)
    contention_resources: dict[int, int] = defaultdict(int)
    for item in contention:
        for allocation in timeline.allocations_by_tti[item.winner.tti_unwrapped]:
            contention_bytes[allocation.rnti] += allocation.tbs_bytes
            contention_resources[allocation.rnti] += allocation.nof_resources

    duration_s = (timeline.last_tti - timeline.first_tti + 1) * _TTI_DURATION_S
    ues = tuple(
        UeFairness(
            rnti=rnti,
            total_bytes=total_bytes[rnti],
            mean_throughput_mbps=total_bytes[rnti] * 8 / duration_s / 1e6,
            contention_bytes=contention_bytes[rnti],
            contention_resources=contention_resources[rnti],
        )
        for rnti in timeline.rntis
    )
    short_term = _short_term_indices(timeline, contention, window_tti)
    return FairnessReport(
        jain_throughput=jain_index(float(ue.total_bytes) for ue in ues),
        jain_contention_throughput=jain_index(float(ue.contention_bytes) for ue in ues),
        jain_contention_resources=jain_index(float(ue.contention_resources) for ue in ues),
        jain_short_term=statistics.fmean(short_term) if short_term else math.nan,
        jain_short_term_min=min(short_term) if short_term else math.nan,
        window_tti=window_tti,
        nof_windows=len(short_term),
        nof_contention_tti=len(contention),
        ues=ues,
    )
