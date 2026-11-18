"""Générateur de logs ``SCHED: DL`` ou ``SCHED: UL`` synthétiques, pour tester le détecteur.

Le modèle reprend la logique des politiques de srsRAN 4G décrite dans
``sched_time_rr.cc`` et ``sched_time_pf.cc`` :

* allocation gloutonne : chaque UE servi prend autant de ressources libres
  qu'il lui en faut pour vider son buffer (``try_dl_newtx_alloc_greedy`` en
  DL ; en UL, un bloc de PRB contigus ramené à une taille valide pour la DFT,
  comme ``find_contiguous_ul_prbs``) ;
* RR : parcours circulaire à partir de ``tti_tx % N`` ;
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

En UL, la bande utile exclut les PRB du PUCCH aux deux bords et, un TTI sur
dix, les 6 PRB du PRACH, comme dans ``sf_grid_t::new_tti`` ; le « buffer »
est le BSR de l'UE.

Ce n'est pas un modèle de canal ni de HARQ : il sert uniquement à produire
des logs dont on connaît la politique, pour valider le classifieur.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14 (voie montante : 2026-11-18)
"""

from __future__ import annotations

import argparse
import enum
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from sched_detector.log_parser import TTI_WRAP_PERIOD, Direction

#: 50 PRB (10 MHz), RBG de 3 PRB : 17 RBG (36.213 §7.1.6.1).
DEFAULT_NOF_RBG: int = 17
#: PRB de la porteuse UL (10 MHz).
DEFAULT_NOF_UL_PRB: int = 50
#: PRB réservés au PUCCH à chaque bord de la bande UL (valeur observée dans
#: les logs srsENB par défaut : grants dans ``[2, 48)``).
UL_PUCCH_EDGE_PRB: int = 2
#: PRB du PRACH (6 PRB, 36.211 §5.7) et TTI où il est réservé.
UL_PRACH_PRB: range = range(4, 10)
UL_PRACH_PERIOD_TTI: int = 10
UL_PRACH_TTI_OFFSET: int = 1
#: Écart entre TTI de réception et TTI d'émission UL dans le log (FDD).
UL_TX_DELAY_TTI: int = 8
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
        bytes_per_resource: Octets transportés par RBG (DL) ou par PRB (UL)
            avec le MCS du canal.
        arrival_bytes_per_tti: Octets arrivant dans la file RLC par TTI
            pendant les périodes actives.
        on_duration_tti: Durée d'une période active ; 0 = toujours actif.
        off_duration_tti: Durée d'une période sans trafic.
        phase_tti: Décalage du motif actif/inactif.
    """

    rnti: int
    bytes_per_resource: float
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
            HARQ factice de 2 RBG ou 4 PRB (sert à vérifier qu'elles sont
            ignorées).
        nof_rbg: Nombre de RBG de la cellule (DL).
        rlc_queue_bytes: Capacité de la file RLC de chaque UE, en octets.
        seed: Graine du générateur aléatoire.
        direction: Voie simulée.
        nof_ul_prb: Nombre de PRB de la porteuse UL.
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
    direction: Direction = Direction.DL
    nof_ul_prb: int = DEFAULT_NOF_UL_PRB

    @property
    def nof_resources(self) -> int:
        """Ressources de la bande : RBG en DL, PRB utiles (hors PUCCH) en UL."""
        if self.direction is Direction.DL:
            return self.nof_rbg
        return self.nof_ul_prb - 2 * UL_PUCCH_EDGE_PRB


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
        full_band_rate = rates[i] * config.nof_resources
        average = states[i].average_rate
        return full_band_rate / average**config.fairness_alpha if average else math.inf

    return sorted(indices, key=priority, reverse=True)


_POLICIES_WITH_AVERAGE = frozenset(
    {SimulatedPolicy.PROPORTIONAL_FAIR, SimulatedPolicy.BLIND_EQUAL_THROUGHPUT}
)


#: Taille des retransmissions factices : 2 RBG en DL, 4 PRB en UL.
_RETX_RESOURCES: dict[Direction, int] = {Direction.DL: 2, Direction.UL: 4}


def _quota(config: SimulationConfig, states: Sequence[_UeState], nof_free: int) -> int:
    """Nombre maximal de ressources par UE ce TTI (sans limite hors partage)."""
    if config.policy is not SimulatedPolicy.FREQUENCY_EQUAL_SHARE:
        return nof_free
    nof_waiting = sum(1 for s in states if s.buffer_bytes > 0)
    return max(1, nof_free // nof_waiting) if nof_waiting else nof_free


def is_valid_dft_size(nof_prb: int) -> bool:
    """Taille de bloc PUSCH autorisée : 2^a·3^b·5^c (``srsran_dft_precoding_valid_prb``)."""
    if nof_prb < 1:
        return False
    for factor in (2, 3, 5):
        while nof_prb % factor == 0:
            nof_prb //= factor
    return nof_prb == 1


def _valid_dft_size_at_most(nof_prb: int) -> int:
    while nof_prb > 0 and not is_valid_dft_size(nof_prb):
        nof_prb -= 1
    return nof_prb


def _valid_dft_size_at_least(nof_prb: int) -> int:
    nof_prb = max(1, nof_prb)
    while not is_valid_dft_size(nof_prb):
        nof_prb += 1
    return nof_prb


def _take_rbgs(free_mask: int, count: int) -> int:
    mask = 0
    bit = 0
    while count > 0:
        if free_mask >> bit & 1:
            mask |= 1 << bit
            count -= 1
        bit += 1
    return mask


def _take_contiguous_prbs(free_mask: int, count: int, nof_prb: int) -> int:
    """Copie de ``find_contiguous_ul_prbs`` (``sched_phy_resource.cc``).

    Prend le premier bloc libre d'au moins ``count`` PRB, sinon le plus long
    bloc libre, puis le raccourcit jusqu'à une taille valide pour la DFT.
    """
    best_start, best_length = 0, 0
    start, length = 0, 0
    for prb in range(nof_prb):
        if free_mask >> prb & 1:
            if length == 0:
                start = prb
            length += 1
            if length >= count:
                best_start, best_length = start, length
                break
        else:
            if length > best_length:
                best_start, best_length = start, length
            length = 0
    else:
        if length > best_length:
            best_start, best_length = start, length
    size = _valid_dft_size_at_most(min(best_length, count))
    return ((1 << size) - 1) << best_start if size else 0


def _initial_free_mask(config: SimulationConfig, tti_raw: int) -> int:
    if config.direction is Direction.DL:
        return (1 << config.nof_rbg) - 1
    usable = ((1 << config.nof_resources) - 1) << UL_PUCCH_EDGE_PRB
    if tti_raw % UL_PRACH_PERIOD_TTI == UL_PRACH_TTI_OFFSET:
        for prb in UL_PRACH_PRB:
            usable &= ~(1 << prb)
    return usable


def _take(config: SimulationConfig, free_mask: int, count: int) -> int:
    """Ressources prises pour un UE : RBG quelconques (DL) ou PRB contigus (UL)."""
    if config.direction is Direction.DL:
        return _take_rbgs(free_mask, count)
    return _take_contiguous_prbs(free_mask, count, config.nof_ul_prb)


def _format_line(
    config: SimulationConfig,
    kind: str,
    rnti: int,
    mask: int,
    tbs: int,
    before: int,
    after: int,
    tti_raw: int,
) -> str:
    if config.direction is Direction.DL:
        return (
            f"2026-10-14T16:00:00.000000 [MAC    ] [I] [{tti_raw:5d}] SCHED: DL {kind} "
            f"rnti=0x{rnti:x}, cc=0, pid=0, mask=0x{mask:x}, dci=(2,0), n_rtx=0, cfi=1, "
            f"tbs={tbs}, buffer={before}/{after}, tti_tx_dl={tti_raw}"
        )
    start = (mask & -mask).bit_length() - 1
    stop = mask.bit_length()
    tti_rx = (tti_raw - UL_TX_DELAY_TTI) % TTI_WRAP_PERIOD
    return (
        f"2026-11-18T14:00:00.000000 [MAC    ] [I] [{tti_rx:5d}] SCHED: UL {kind} "
        f"rnti=0x{rnti:x}, cc=0, pid={tti_raw % 8}, dci=(1,0), prb=[{start}, {stop}), "
        f"n_rtx=0, cfi=1, tbs={tbs}, bsr={after} ({before}-{before - after}), "
        f"tti_tx_ul={tti_raw}"
    )


def simulate(config: SimulationConfig) -> Iterator[str]:
    """Produit les lignes de log d'une simulation.

    Args:
        config: Paramètres de la simulation.

    Yields:
        Une ligne ``SCHED: DL|UL tx`` (ou ``retx``) par allocation.

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
    is_uplink = config.direction is Direction.UL

    for tti in range(config.first_tti, config.first_tti + config.nof_tti):
        tti_raw = tti % TTI_WRAP_PERIOD
        for state in states:
            if state.params.is_active(tti):
                state.buffer_bytes = min(
                    config.rlc_queue_bytes, state.buffer_bytes + state.params.arrival_bytes_per_tti
                )
        rates = [
            s.params.bytes_per_resource
            * (1 + generator.uniform(-config.channel_noise, config.channel_noise))
            for s in states
        ]
        free_mask = _initial_free_mask(config, tti_raw)
        if config.retx_probability and generator.random() < config.retx_probability:
            retx_ue = generator.choice(states)
            retx_mask = _take(config, free_mask, _RETX_RESOURCES[config.direction])
            free_mask &= ~retx_mask
            tbs = int(retx_mask.bit_count() * retx_ue.params.bytes_per_resource)
            yield _format_line(
                config,
                "retx",
                retx_ue.params.rnti,
                retx_mask,
                tbs,
                retx_ue.buffer_bytes,
                retx_ue.buffer_bytes,
                tti_raw,
            )
        allocated = [0.0] * len(states)
        quota = _quota(config, states, free_mask.bit_count())
        for index in _service_order(config, states, rates, tti_raw, generator):
            state = states[index]
            nof_free = free_mask.bit_count()
            if state.buffer_bytes == 0 or nof_free == 0:
                continue
            needed = math.ceil(state.buffer_bytes / rates[index])
            if is_uplink:
                # get_required_prb_ul : plus petite taille valide qui suffit.
                needed = _valid_dft_size_at_least(needed)
            mask = _take(config, free_mask, min(nof_free, needed, quota))
            if mask == 0:
                continue
            free_mask &= ~mask
            tbs = int(mask.bit_count() * rates[index])
            before = state.buffer_bytes
            state.buffer_bytes = max(0, before - tbs)
            allocated[index] = tbs
            yield _format_line(
                config, "tx", state.params.rnti, mask, tbs, before, state.buffer_bytes, tti_raw
            )
        if config.policy in _POLICIES_WITH_AVERAGE:
            for state, allocated_bytes in zip(states, allocated, strict=True):
                state.save_allocation(allocated_bytes)


#: UE de la démonstration, par voie : (octets par ressource du bon canal,
#: du canal dégradé). En UL, 100 et 50 octets par PRB donnent 4600 et
#: 2300 octets pleine bande, du même ordre que les logs réels (2865 octets
#: sur 45 PRB).
_DEMO_RATES: dict[Direction, tuple[float, float]] = {
    Direction.DL: (300.0, 150.0),
    Direction.UL: (100.0, 50.0),
}


def main(argv: list[str] | None = None) -> int:
    """Écrit un log synthétique, pour essayer ``sched-detect`` sans radio.

    Exemples : ``sched-simulate --policy pf --alpha 2 --out demo_pf.log`` ou
    ``sched-simulate --direction ul --policy rr --out demo_ul_rr.log``.
    Deux UE : le premier (bon canal) en trafic intermittent, le second
    (canal deux fois moins bon) saturé en permanence.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--policy", choices=[p.value for p in SimulatedPolicy], required=True)
    parser.add_argument(
        "--direction",
        choices=[d.value for d in Direction],
        default=Direction.DL.value,
        help="Voie simulée (défaut dl).",
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Coefficient d'équité du PF.")
    parser.add_argument("--nof-tti", type=int, default=20_000, help="Durée simulée, en TTI.")
    parser.add_argument("--out", type=Path, required=True, help="Fichier de log à écrire.")
    args = parser.parse_args(argv)
    direction = Direction(args.direction)
    good_rate, poor_rate = _DEMO_RATES[direction]
    ues = (
        SimulatedUe(0x46, good_rate, 7_000, on_duration_tti=1_000, off_duration_tti=500),
        SimulatedUe(0x47, poor_rate, 7_000),
    )
    config = SimulationConfig(
        SimulatedPolicy(args.policy),
        ues,
        args.nof_tti,
        fairness_alpha=args.alpha,
        direction=direction,
    )
    with args.out.open("w", encoding="utf-8") as log_file:
        for line in simulate(config):
            log_file.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
