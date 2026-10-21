"""Générateur de logs ``SCHED: DL`` synthétiques, pour tester le détecteur.

Le modèle reprend la logique des politiques de srsRAN 4G décrite dans
``sched_time_rr.cc`` et ``sched_time_pf.cc`` :

* allocation gloutonne : chaque UE servi prend autant de RBG libres qu'il
  lui en faut pour vider son buffer (``try_dl_newtx_alloc_greedy``) ;
* RR : parcours circulaire à partir de ``tti_tx_dl % N`` ;
* PF : tri par ``r / R^alpha``, moyenne arithmétique sur les 100 premiers
  échantillons puis moyenne exponentielle de pas 0,01 (``save_dl_alloc``) ;
* Max C/I : tri par débit instantané décroissant.

Pour vérifier que le détecteur reconnaît aussi ce qu'il ne connaît pas, il
simule en plus des politiques absentes de srsRAN :

* BET (Blind Equal Throughput) : sert d'abord le plus petit débit moyen ;
* LQF (Longest Queue First) : sert d'abord le plus gros buffer ;
* aléatoire : ordre tiré au sort à chaque TTI ;
* partage fréquentiel : la bande est divisée en parts égales entre les UE
  qui ont des données, au lieu d'être donnée au premier servi.

Ce n'est pas un modèle de canal ni de HARQ : il sert uniquement à produire
des logs dont on connaît la politique, pour valider le classifieur.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14
"""

from __future__ import annotations

import argparse
import enum
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from sched_detector.log_parser import TTI_WRAP_PERIOD

#: 50 PRB (10 MHz), RBG de 3 PRB : 17 RBG (36.213 §7.1.6.1).
DEFAULT_NOF_RBG: int = 17
#: Pas de la moyenne exponentielle du PF dans srsRAN (codé en dur).
PF_EXP_AVG_ALPHA: float = 0.01
#: Taille par défaut de la file RLC simulée, en octets. Une file bornée
#: reproduit le comportement réel sous iperf UDP : au-delà, RLC jette les SDU,
#: et le buffer se vide en quelques dizaines de ms quand le trafic s'arrête.
DEFAULT_RLC_QUEUE_BYTES: int = 64_000


class SimulatedPolicy(enum.Enum):
    """Politiques simulées."""

    ROUND_ROBIN = "rr"
    PROPORTIONAL_FAIR = "pf"
    MAX_CI = "maxci"
    BLIND_EQUAL_THROUGHPUT = "bet"
    LONGEST_QUEUE_FIRST = "lqf"
    RANDOM = "random"
    FREQUENCY_EQUAL_SHARE = "freq_share"


@dataclass(frozen=True, slots=True)
class SimulatedUe:
    """Paramètres d'un UE simulé.

    Attributes:
        rnti: Identifiant radio.
        bytes_per_rbg: Octets transportés par RBG avec le MCS du canal.
        arrival_bytes_per_tti: Octets arrivant dans la file RLC par TTI
            pendant les périodes actives.
        on_duration_tti: Durée d'une période active ; 0 = toujours actif.
        off_duration_tti: Durée d'une période sans trafic.
        phase_tti: Décalage du motif actif/inactif.
    """

    rnti: int
    bytes_per_rbg: float
    arrival_bytes_per_tti: int
    on_duration_tti: int = 0
    off_duration_tti: int = 0
    phase_tti: int = 0

    def is_active(self, tti: int) -> bool:
        """Indique si l'UE reçoit du trafic au TTI ``tti``."""
        if self.on_duration_tti == 0 or self.off_duration_tti == 0:
            return True
        period = self.on_duration_tti + self.off_duration_tti
        return (tti + self.phase_tti) % period < self.on_duration_tti


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Paramètres d'une simulation.

    Attributes:
        policy: Politique simulée.
        ues: UE simulés, dans l'ordre d'itération de ``ue_db``.
        nof_tti: Nombre de TTI simulés.
        first_tti: TTI de départ (non replié), utile pour tester le
            rebouclage à 10240.
        fairness_alpha: Coefficient d'équité du PF.
        channel_noise: Variation relative uniforme du débit par TTI
            (fluctuation du CQI), dans [0, 1).
        retx_probability: Probabilité par TTI d'émettre une retransmission
            HARQ factice de 2 RBG (sert à vérifier qu'elles sont ignorées).
        nof_rbg: Nombre de RBG de la cellule.
        rlc_queue_bytes: Capacité de la file RLC de chaque UE, en octets.
        seed: Graine du générateur aléatoire.
    """

    policy: SimulatedPolicy
    ues: tuple[SimulatedUe, ...]
    nof_tti: int
    first_tti: int = 0
    fairness_alpha: float = 2.0
    channel_noise: float = 0.05
    retx_probability: float = 0.0
    nof_rbg: int = DEFAULT_NOF_RBG
    rlc_queue_bytes: int = DEFAULT_RLC_QUEUE_BYTES
    seed: int = 1


@dataclass(slots=True)
class _UeState:
    params: SimulatedUe
    buffer_bytes: int = 0
    average_rate: float = 0.0
    nof_samples: int = 0

    def save_allocation(self, allocated_bytes: float) -> None:
        # Copie de sched_time_pf::ue_ctxt::save_dl_alloc.
        if self.nof_samples < 1 / PF_EXP_AVG_ALPHA:
            self.average_rate += (allocated_bytes - self.average_rate) / (self.nof_samples + 1)
        else:
            self.average_rate = (
                1 - PF_EXP_AVG_ALPHA
            ) * self.average_rate + PF_EXP_AVG_ALPHA * allocated_bytes
        self.nof_samples += 1


def _service_order(
    config: SimulationConfig,
    states: Sequence[_UeState],
    rates: Sequence[float],
    tti_raw: int,
    generator: random.Random,
) -> list[int]:
    indices = list(range(len(states)))
    policy = config.policy
    if policy in (SimulatedPolicy.ROUND_ROBIN, SimulatedPolicy.FREQUENCY_EQUAL_SHARE):
        start = tti_raw % len(states)
        return indices[start:] + indices[:start]
    if policy is SimulatedPolicy.MAX_CI:
        return sorted(indices, key=lambda i: rates[i], reverse=True)
    if policy is SimulatedPolicy.BLIND_EQUAL_THROUGHPUT:
        return sorted(indices, key=lambda i: states[i].average_rate)
    if policy is SimulatedPolicy.LONGEST_QUEUE_FIRST:
        return sorted(indices, key=lambda i: states[i].buffer_bytes, reverse=True)
    if policy is SimulatedPolicy.RANDOM:
        generator.shuffle(indices)
        return indices

    def priority(i: int) -> float:
        full_band_rate = rates[i] * config.nof_rbg
        average = states[i].average_rate
        return full_band_rate / average**config.fairness_alpha if average else math.inf

    return sorted(indices, key=priority, reverse=True)


_POLICIES_WITH_AVERAGE = frozenset(
    {SimulatedPolicy.PROPORTIONAL_FAIR, SimulatedPolicy.BLIND_EQUAL_THROUGHPUT}
)


def _rbg_quota(config: SimulationConfig, states: Sequence[_UeState], nof_free: int) -> int:
    """Nombre maximal de RBG par UE ce TTI (sans limite hors partage)."""
    if config.policy is not SimulatedPolicy.FREQUENCY_EQUAL_SHARE:
        return nof_free
    nof_waiting = sum(1 for s in states if s.buffer_bytes > 0)
    return max(1, nof_free // nof_waiting) if nof_waiting else nof_free


def _format_line(
    kind: str, rnti: int, mask: int, tbs: int, before: int, after: int, tti_raw: int
) -> str:
    return (
        f"2026-10-14T16:00:00.000000 [MAC    ] [I] [{tti_raw:5d}] SCHED: DL {kind} "
        f"rnti=0x{rnti:x}, cc=0, pid=0, mask=0x{mask:x}, dci=(2,0), n_rtx=0, cfi=1, "
        f"tbs={tbs}, buffer={before}/{after}, tti_tx_dl={tti_raw}"
    )


def _take_rbgs(free_mask: int, count: int) -> int:
    mask = 0
    bit = 0
    while count > 0:
        if free_mask >> bit & 1:
            mask |= 1 << bit
            count -= 1
        bit += 1
    return mask


def simulate(config: SimulationConfig) -> Iterator[str]:
    """Produit les lignes de log d'une simulation.

    Args:
        config: Paramètres de la simulation.

    Yields:
        Une ligne ``SCHED: DL tx`` (ou ``retx``) par allocation.

    Raises:
        ValueError: Si aucun UE n'est fourni ou si ``channel_noise`` sort
            de [0, 1).
    """
    if not config.ues:
        raise ValueError("Au moins un UE est nécessaire")
    if not 0.0 <= config.channel_noise < 1.0:
        raise ValueError(f"channel_noise doit être dans [0, 1), reçu {config.channel_noise}")
    generator = random.Random(config.seed)
    states = [_UeState(params=ue) for ue in config.ues]
    full_mask = (1 << config.nof_rbg) - 1

    for tti in range(config.first_tti, config.first_tti + config.nof_tti):
        tti_raw = tti % TTI_WRAP_PERIOD
        for state in states:
            if state.params.is_active(tti):
                state.buffer_bytes = min(
                    config.rlc_queue_bytes, state.buffer_bytes + state.params.arrival_bytes_per_tti
                )
        rates = [
            s.params.bytes_per_rbg
            * (1 + generator.uniform(-config.channel_noise, config.channel_noise))
            for s in states
        ]
        free_mask = full_mask
        if config.retx_probability and generator.random() < config.retx_probability:
            retx_ue = generator.choice(states)
            retx_mask = _take_rbgs(free_mask, 2)
            free_mask &= ~retx_mask
            tbs = int(2 * retx_ue.params.bytes_per_rbg)
            yield _format_line(
                "retx",
                retx_ue.params.rnti,
                retx_mask,
                tbs,
                retx_ue.buffer_bytes,
                retx_ue.buffer_bytes,
                tti_raw,
            )
        allocated = [0.0] * len(states)
        quota = _rbg_quota(config, states, free_mask.bit_count())
        for index in _service_order(config, states, rates, tti_raw, generator):
            state = states[index]
            nof_free = free_mask.bit_count()
            if state.buffer_bytes == 0 or nof_free == 0:
                continue
            needed = math.ceil(state.buffer_bytes / rates[index])
            nof_rbg = min(nof_free, needed, quota)
            mask = _take_rbgs(free_mask, nof_rbg)
            free_mask &= ~mask
            tbs = int(nof_rbg * rates[index])
            before = state.buffer_bytes
            state.buffer_bytes = max(0, before - tbs)
            allocated[index] = tbs
            yield _format_line(
                "tx", state.params.rnti, mask, tbs, before, state.buffer_bytes, tti_raw
            )
        if config.policy in _POLICIES_WITH_AVERAGE:
            for state, allocated_bytes in zip(states, allocated, strict=True):
                state.save_allocation(allocated_bytes)


def main(argv: list[str] | None = None) -> int:
    """Écrit un log synthétique, pour essayer ``sched-detect`` sans radio.

    Exemple : ``sched-simulate --policy pf --alpha 2 --out demo_pf.log``.
    Deux UE : le premier (bon canal) en trafic intermittent, le second
    (canal deux fois moins bon) saturé en permanence.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--policy", choices=[p.value for p in SimulatedPolicy], required=True)
    parser.add_argument("--alpha", type=float, default=2.0, help="Coefficient d'équité du PF.")
    parser.add_argument("--nof-tti", type=int, default=20_000, help="Durée simulée, en TTI.")
    parser.add_argument("--out", type=Path, required=True, help="Fichier de log à écrire.")
    args = parser.parse_args(argv)
    ues = (
        SimulatedUe(0x46, 300, 7_000, on_duration_tti=1_000, off_duration_tti=500),
        SimulatedUe(0x47, 150, 7_000),
    )
    config = SimulationConfig(
        SimulatedPolicy(args.policy), ues, args.nof_tti, fairness_alpha=args.alpha
    )
    with args.out.open("w", encoding="utf-8") as log_file:
        for line in simulate(config):
            log_file.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
