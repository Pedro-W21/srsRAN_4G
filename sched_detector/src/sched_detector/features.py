"""Métriques comportementales extraites des allocations DL.

Idée générale : on ne regarde que les TTI « disputés », c'est-à-dire ceux où
au moins deux UE avaient des données en attente. C'est seulement là que la
politique fait un choix. Pour chacun de ces TTI, le « gagnant » est l'UE qui
a reçu le plus de RBG (avec l'allocation gloutonne de srsRAN, le premier UE
servi prend en général toute la bande).

Trois empreintes en découlent :

* **Verrouillage de phase.** ``sched_time_rr`` commence son parcours à
  l'indice ``tti_tx_dl % ue_db.size()`` (``sched_time_rr.cc``) et saute les
  UE sans données. Le gagnant est donc une fonction du seul couple
  (``tti_tx_dl mod N``, ensemble des UE en attente), quel que soit
  l'historique. PF et Max C/I dépendent du passé ou du canal, pas du numéro
  de TTI.
* **Part de TTI en fonction du canal.** À l'équilibre d'un PF temporel avec
  priorité ``r / R^alpha``, toutes les priorités sont égales, donc
  ``R_i ∝ r_i^(1/alpha)``. Comme ``R_i = f_i r_i`` (f_i = part des TTI
  gagnés), on obtient ``f_i ∝ r_i^(1/alpha - 1)``. La pente ``s`` de
  ``log f`` en fonction de ``log r`` donne ``alpha = 1 / (1 + s)``.
  RR donne ``s ≈ 0`` ; Max C/I donne presque tout au meilleur canal.
* **Rattrapage.** Après une pause de trafic, un UE PF a une moyenne basse et
  gagne plusieurs TTI d'affilée ; un UE RR reprend sa place dans la rotation.
  C'est ce qui casse le verrouillage de phase d'un PF qui alternerait par
  hasard au même rythme qu'un RR.

Ces métriques décrivent le log. La décision elle-même repose sur le rejeu
des modèles (``models.py``), qui estime aussi alpha plus finement que la
pente des parts, conservée ici comme vérification croisée.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14
"""

from __future__ import annotations

import bisect
import itertools
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from sched_detector.log_parser import DlAllocation

#: Durée exclue après une reprise de trafic pour estimer les parts en régime
#: établi, en TTI. Trois constantes de temps de la moyenne PF de srsRAN
#: (beta = 0,01 dans ``sched_time_pf.cc``, soit ~100 TTI).
DEFAULT_TRANSIENT_EXCLUSION_TTI: int = 300


@dataclass(frozen=True, slots=True)
class ContestedTti:
    """Un TTI où au moins deux UE avaient des données.

    Attributes:
        winner: Allocation qui a reçu le plus de RBG.
        backlogged_rntis: UE qui avaient des données à ce TTI.
    """

    winner: DlAllocation
    backlogged_rntis: frozenset[int]


@dataclass(frozen=True, slots=True)
class Timeline:
    """Chronologie des décisions reconstituée depuis le log.

    Attributes:
        nof_ues: Nombre d'UE (N du modulo RR).
        rntis: RNTI des UE vus, triés.
        first_tti: Premier TTI (déplié) avec une nouvelle transmission.
        last_tti: Dernier TTI (déplié) avec une nouvelle transmission.
        contested: TTI disputés, dans l'ordre chronologique.
        resume_ttis: TTI des reprises de trafic, triés.
        allocations_by_tti: Nouvelles transmissions groupées par TTI déplié.
    """

    nof_ues: int
    rntis: tuple[int, ...]
    first_tti: int
    last_tti: int
    contested: tuple[ContestedTti, ...]
    resume_ttis: tuple[int, ...]
    allocations_by_tti: dict[int, tuple[DlAllocation, ...]]


@dataclass(frozen=True, slots=True)
class UeStatistics:
    """Statistiques d'un UE.

    Attributes:
        rnti: Identifiant radio.
        contested_wins: TTI disputés gagnés (tout le log).
        steady_share: Part des TTI gagnés quand **tous** les UE avaient des
            données, hors régimes transitoires, dans [0, 1]. ``nan`` si cette
            situation n'a jamais eu lieu.
        bytes_per_rbg: Médiane du TBS par RBG, en octets, sur les allocations
            où l'UE avait encore des données après coup (MCS non bridé par
            un buffer presque vide). ``nan`` si aucune.
    """

    rnti: int
    contested_wins: int
    steady_share: float
    bytes_per_rbg: float


@dataclass(frozen=True, slots=True)
class SchedulerFeatures:
    """Ensemble des métriques utilisées par le classifieur.

    Attributes:
        nof_ues: Nombre d'UE (N du modulo RR).
        nof_tti_with_allocation: TTI ayant au moins une nouvelle transmission.
        nof_contested_tti: TTI où au moins deux UE avaient des données.
        nof_full_contention_tti: TTI disputés en régime établi où tous les
            UE avaient des données (base du calcul des parts).
        phase_lock: Part des TTI disputés dont le gagnant est le gagnant
            majoritaire de son couple (résidu ``tti_tx_dl mod N``, UE en
            attente).
        phase_lock_excess: ``phase_lock`` normalisé par rapport à une
            prédiction qui n'utiliserait que l'ensemble des UE en attente
            (1 = gagnant entièrement fixé par le résidu, 0 = résidu inutile).
            Même construction que le kappa de Cohen, pour qu'un UE qui gagne
            tout ne ressemble pas à du RR.
        same_winner_rate: Part des paires de TTI disputés consécutifs avec le
            même gagnant.
        greedy_violation_rate: Part des TTI disputés où au moins deux UE
            servis gardent des données (bande partagée, impossible avec
            l'allocation gloutonne de RR, PF et Max C/I).
        nof_resume_events: Reprises de trafic (buffer vide, puis de nouveau
            des données).
        channel_spread: Rapport max/min des ``bytes_per_rbg`` des UE.
        share_vs_rate_slope: Pente de ``log(part)`` contre
            ``log(octets par RBG)``, ``nan`` si non calculable.
        ues: Statistiques par UE, triées par RNTI.
    """

    nof_ues: int
    nof_tti_with_allocation: int
    nof_contested_tti: int
    nof_full_contention_tti: int
    phase_lock: float
    phase_lock_excess: float
    same_winner_rate: float
    greedy_violation_rate: float
    nof_resume_events: int
    channel_spread: float
    share_vs_rate_slope: float
    ues: tuple[UeStatistics, ...] = field(default_factory=tuple)

    @property
    def estimated_fairness_alpha(self) -> float:
        """Coefficient d'équité PF déduit de la pente : ``1 / (1 + s)``.

        Returns:
            ``alpha`` estimé, ``nan`` si la pente est inconnue, ``inf`` si
            ``s <= -1`` (la part décroît plus vite que le débit ne croît).
        """
        slope = self.share_vs_rate_slope
        if math.isnan(slope):
            return math.nan
        if slope <= -1.0:
            return math.inf
        return 1.0 / (1.0 + slope)


def _new_transmissions(allocations: Sequence[DlAllocation]) -> list[DlAllocation]:
    # Les retransmissions HARQ passent avant tout le monde dans RR comme dans
    # PF : elles ne disent rien de la politique.
    return [a for a in allocations if not a.is_retransmission and a.nof_rbg > 0]


def _winners_by_tti(allocations: Sequence[DlAllocation]) -> dict[int, DlAllocation]:
    winners: dict[int, DlAllocation] = {}
    for allocation in allocations:
        current = winners.get(allocation.tti_unwrapped)
        if current is None or allocation.nof_rbg > current.nof_rbg:
            winners[allocation.tti_unwrapped] = allocation
    return winners


def _backlog_by_ue(
    allocations: Sequence[DlAllocation], first_tti: int, span: int
) -> tuple[dict[int, bytearray], list[int]]:
    """Reconstitue, pour chaque UE, les TTI où il avait des données en attente.

    Un UE est en attente sur ``]t_k, t_k+1]`` (entre deux de ses allocations
    successives) quand l'allocation ``t_k`` laisse ``buffer_after > 0`` : les
    données restent dans la file RLC jusqu'à la suivante. Après une
    allocation qui vide le buffer, on ne sait pas quand de nouvelles données
    sont arrivées : l'UE n'est pas compté avant sa prochaine allocation,
    qui est enregistrée comme une reprise de trafic.

    Returns:
        Un masque par UE (1 octet par TTI, indice ``tti - first_tti``) et les
        TTI des reprises de trafic.
    """
    per_ue: dict[int, list[DlAllocation]] = defaultdict(list)
    for allocation in allocations:
        per_ue[allocation.rnti].append(allocation)
    backlog: dict[int, bytearray] = {}
    resume_ttis: list[int] = []
    for rnti, ue_allocations in per_ue.items():
        ue_allocations.sort(key=lambda a: a.tti_unwrapped)
        mask = bytearray(span)
        for current, following in itertools.pairwise(ue_allocations):
            if current.buffer_after_bytes > 0:
                start = current.tti_unwrapped - first_tti + 1
                stop = following.tti_unwrapped - first_tti + 1
                mask[start:stop] = b"\x01" * (stop - start)
            else:
                resume_ttis.append(following.tti_unwrapped)
        backlog[rnti] = mask
    return backlog, resume_ttis


def _contested_ttis(
    winners_by_tti: dict[int, DlAllocation], backlog: dict[int, bytearray], first_tti: int
) -> list[ContestedTti]:
    contested: list[ContestedTti] = []
    for tti in sorted(winners_by_tti):
        index = tti - first_tti
        waiting = frozenset(rnti for rnti, mask in backlog.items() if mask[index])
        if len(waiting) >= 2:
            contested.append(ContestedTti(winners_by_tti[tti], waiting))
    return contested


def _phase_lock(contested: Sequence[ContestedTti], nof_ues: int) -> tuple[float, float]:
    by_key: dict[tuple[int, frozenset[int]], Counter[int]] = defaultdict(Counter)
    by_backlog: dict[frozenset[int], Counter[int]] = defaultdict(Counter)
    for item in contested:
        # Résidu calculé sur le TTI brut, comme dans sched_time_rr.cc : le
        # rebouclage à 10240 est donc reproduit à l'identique.
        residue = item.winner.tti_tx_dl % nof_ues
        by_key[(residue, item.backlogged_rntis)][item.winner.rnti] += 1
        by_backlog[item.backlogged_rntis][item.winner.rnti] += 1
    total = len(contested)
    phase_lock = sum(c.most_common(1)[0][1] for c in by_key.values()) / total
    baseline = sum(c.most_common(1)[0][1] for c in by_backlog.values()) / total
    if baseline >= 1.0:
        return phase_lock, 0.0
    return phase_lock, (phase_lock - baseline) / (1.0 - baseline)


def _same_winner_rate(contested: Sequence[ContestedTti]) -> float:
    pairs = 0
    repeats = 0
    for previous, current in itertools.pairwise(contested):
        if current.winner.tti_unwrapped - previous.winner.tti_unwrapped == 1:
            pairs += 1
            repeats += int(current.winner.rnti == previous.winner.rnti)
    return repeats / pairs if pairs else math.nan


def _bytes_per_rbg(allocations: Sequence[DlAllocation]) -> dict[int, float]:
    samples: dict[int, list[float]] = defaultdict(list)
    for allocation in allocations:
        # Avec un buffer presque vide, srsRAN choisit le plus petit MCS qui
        # suffit (compute_min_mcs_and_tbs_from_required_bytes) : ces
        # allocations sous-estimeraient la qualité du canal.
        if allocation.buffer_after_bytes > 0:
            samples[allocation.rnti].append(allocation.tbs_bytes / allocation.nof_rbg)
    return {rnti: statistics.median(values) for rnti, values in samples.items()}


def _exclude_transients(
    contested: Sequence[ContestedTti], resume_ttis: Sequence[int], exclusion_tti: int
) -> list[ContestedTti]:
    if not resume_ttis or exclusion_tti <= 0:
        return list(contested)
    sorted_resumes = sorted(resume_ttis)
    kept: list[ContestedTti] = []
    for item in contested:
        tti = item.winner.tti_unwrapped
        position = bisect.bisect_right(sorted_resumes, tti)
        if position == 0 or tti - sorted_resumes[position - 1] >= exclusion_tti:
            kept.append(item)
    return kept


def _log_log_slope(points: Sequence[tuple[float, float]]) -> float:
    """Pente des moindres carrés de ``log(y)`` contre ``log(x)``."""
    usable = [(math.log(x), math.log(y)) for x, y in points if x > 0 and y > 0]
    if len(usable) < 2:
        return math.nan
    mean_x = statistics.fmean(p[0] for p in usable)
    mean_y = statistics.fmean(p[1] for p in usable)
    variance_x = sum((p[0] - mean_x) ** 2 for p in usable)
    if variance_x == 0.0:
        return math.nan
    covariance = sum((p[0] - mean_x) * (p[1] - mean_y) for p in usable)
    return covariance / variance_x


def build_timeline(allocations: Sequence[DlAllocation], nof_ues: int | None = None) -> Timeline:
    """Reconstitue la chronologie des décisions du scheduler.

    Args:
        allocations: Allocations DL d'une porteuse, ordre quelconque.
        nof_ues: Nombre d'UE connectés (N du modulo RR). Par défaut, le
            nombre de RNTI distincts vus dans le log. À fournir si un UE
            connecté n'a jamais reçu de données.

    Returns:
        La chronologie, base commune des métriques et du rejeu des modèles.

    Raises:
        ValueError: Si ``allocations`` ne contient aucune nouvelle
            transmission ou si ``nof_ues < 1``.
    """
    new_tx = _new_transmissions(allocations)
    if not new_tx:
        raise ValueError("Aucune nouvelle transmission DL dans le log (mac_level=info ?)")
    seen_rntis = tuple(sorted({a.rnti for a in new_tx}))
    nof_ues = len(seen_rntis) if nof_ues is None else nof_ues
    if nof_ues < 1:
        raise ValueError(f"nof_ues doit être >= 1, reçu {nof_ues}")

    first_tti = min(a.tti_unwrapped for a in new_tx)
    last_tti = max(a.tti_unwrapped for a in new_tx)
    winners_by_tti = _winners_by_tti(new_tx)
    backlog, resume_ttis = _backlog_by_ue(new_tx, first_tti, last_tti - first_tti + 1)
    by_tti: dict[int, list[DlAllocation]] = defaultdict(list)
    for allocation in new_tx:
        by_tti[allocation.tti_unwrapped].append(allocation)
    return Timeline(
        nof_ues=nof_ues,
        rntis=seen_rntis,
        first_tti=first_tti,
        last_tti=last_tti,
        contested=tuple(_contested_ttis(winners_by_tti, backlog, first_tti)),
        resume_ttis=tuple(sorted(resume_ttis)),
        allocations_by_tti={tti: tuple(items) for tti, items in by_tti.items()},
    )


def _greedy_violation_rate(timeline: Timeline) -> float:
    """Part des TTI disputés où au moins deux UE restent non satisfaits.

    Avec l'allocation gloutonne des politiques temporelles de srsRAN, le
    premier UE qui ne peut pas vider son buffer prend tous les RBG libres :
    les UE servis après lui n'obtiennent rien. Deux UE servis le même TTI et
    tous deux avec ``buffer_after > 0`` signent donc un partage de la bande
    (politique fréquentielle ou quota par UE).
    """
    violations = 0
    for item in timeline.contested:
        allocations = timeline.allocations_by_tti[item.winner.tti_unwrapped]
        unsatisfied = sum(1 for a in allocations if a.buffer_after_bytes > 0)
        violations += int(unsatisfied >= 2)
    return violations / len(timeline.contested)


def extract_features(
    timeline: Timeline, transient_exclusion_tti: int = DEFAULT_TRANSIENT_EXCLUSION_TTI
) -> SchedulerFeatures:
    """Calcule les métriques descriptives d'une chronologie.

    Args:
        timeline: Sortie de :func:`build_timeline`.
        transient_exclusion_tti: TTI ignorés après chaque reprise de trafic
            pour le calcul des parts en régime établi.

    Returns:
        Les métriques. Si aucun TTI n'est disputé, les métriques de
        comparaison valent ``nan``.
    """
    all_allocations = [a for items in timeline.allocations_by_tti.values() for a in items]
    rates = _bytes_per_rbg(all_allocations)
    contested = timeline.contested
    if not contested:
        return SchedulerFeatures(
            nof_ues=timeline.nof_ues,
            nof_tti_with_allocation=len(timeline.allocations_by_tti),
            nof_contested_tti=0,
            nof_full_contention_tti=0,
            phase_lock=math.nan,
            phase_lock_excess=math.nan,
            same_winner_rate=math.nan,
            greedy_violation_rate=math.nan,
            nof_resume_events=len(timeline.resume_ttis),
            channel_spread=math.nan,
            share_vs_rate_slope=math.nan,
            ues=tuple(UeStatistics(r, 0, math.nan, rates.get(r, math.nan)) for r in timeline.rntis),
        )

    phase_lock, phase_lock_excess = _phase_lock(contested, timeline.nof_ues)
    all_rntis = frozenset(timeline.rntis)
    full_contention = [
        item
        for item in _exclude_transients(contested, timeline.resume_ttis, transient_exclusion_tti)
        if item.backlogged_rntis == all_rntis
    ]
    wins = Counter(item.winner.rnti for item in contested)
    steady_wins = Counter(item.winner.rnti for item in full_contention)
    ues = tuple(
        UeStatistics(
            rnti=rnti,
            contested_wins=wins[rnti],
            steady_share=steady_wins[rnti] / len(full_contention) if full_contention else math.nan,
            bytes_per_rbg=rates.get(rnti, math.nan),
        )
        for rnti in timeline.rntis
    )
    known_rates = [u.bytes_per_rbg for u in ues if not math.isnan(u.bytes_per_rbg)]
    channel_spread = max(known_rates) / min(known_rates) if len(known_rates) >= 2 else math.nan

    return SchedulerFeatures(
        nof_ues=timeline.nof_ues,
        nof_tti_with_allocation=len(timeline.allocations_by_tti),
        nof_contested_tti=len(contested),
        nof_full_contention_tti=len(full_contention),
        phase_lock=phase_lock,
        phase_lock_excess=phase_lock_excess,
        same_winner_rate=_same_winner_rate(contested),
        greedy_violation_rate=_greedy_violation_rate(timeline),
        nof_resume_events=len(timeline.resume_ttis),
        channel_spread=channel_spread,
        share_vs_rate_slope=_log_log_slope([(u.bytes_per_rbg, u.steady_share) for u in ues]),
        ues=ues,
    )
