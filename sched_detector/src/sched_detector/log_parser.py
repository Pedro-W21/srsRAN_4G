"""Lecture des lignes ``SCHED: DL`` et ``SCHED: UL`` d'un log srsENB (srsRAN 4G).

Le scheduler écrit une ligne par allocation au niveau ``info`` du logger MAC
(``srsenb/src/stack/mac/sched_grid.cc``). En voie descendante
(``set_dl_data_sched_result``) ::

    SCHED: DL tx rnti=0x46, cc=0, pid=2, mask=0x1ffff, dci=(2,8), n_rtx=0,
    cfi=1, tbs=4584, buffer=12000/7400, tti_tx_dl=5123

En voie montante (``set_ul_sched_result``) ::

    SCHED: UL tx rnti=0x4c, cc=0, pid=0, dci=(1,0), prb=[2, 47), n_rtx=0,
    cfi=1, tbs=2865, bsr=138208 (152053-13845), tti_tx_ul=2624

Les deux voies sont ramenées à une même représentation, :class:`Allocation` :

=====================  ===========================  ==============================
Champ                  DL                           UL
=====================  ===========================  ==============================
ressources             RBG du champ ``mask``        PRB de l'intervalle ``prb=[a, b)``
données avant          ``buffer=avant/…``           ``bsr=… (b-…)`` : dernier BSR reçu
données après          ``buffer=…/après``           ``bsr=a`` : BSR moins les octets
                                                    déjà accordés et pas encore reçus
TTI                    ``tti_tx_dl``                ``tti_tx_ul``
=====================  ===========================  ==============================

« Données après > 0 » veut dire que l'UE a encore quelque chose à recevoir
(DL) ou à émettre (UL) : c'est ce qui permet de savoir quels UE étaient en
attente, dans les deux voies.

Les champs sont lus un par un (et non avec une seule expression régulière
figée) pour tolérer un ordre ou un champ différent selon la version.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14 (voie montante : 2026-11-18)
"""

from __future__ import annotations

import enum
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

#: Période du compteur de TTI LTE : 1024 trames x 10 sous-trames (36.211 §4.1).
TTI_WRAP_PERIOD: int = 10_240


class Direction(enum.Enum):
    """Voie analysée."""

    DL = "dl"
    UL = "ul"

    @property
    def label(self) -> str:
        """Nom lisible de la voie."""
        return "descendante (DL)" if self is Direction.DL else "montante (UL)"

    @property
    def resource_unit(self) -> str:
        """Unité d'allocation : RBG en DL (type 0), PRB en UL (SC-FDMA contigu)."""
        return "RBG" if self is Direction.DL else "PRB"

    @property
    def tti_field(self) -> str:
        """Nom du champ TTI dans le log (utilisé par le modulo du RR)."""
        return "tti_tx_dl" if self is Direction.DL else "tti_tx_ul"


_HEADER_RE: dict[Direction, re.Pattern[str]] = {
    Direction.DL: re.compile(r"SCHED:\s+DL\s+(?P<kind>tx|retx)\b"),
    Direction.UL: re.compile(r"SCHED:\s+UL\s+(?P<kind>tx|retx)\b"),
}
_RNTI_RE = re.compile(r"\brnti=0x(?P<value>[0-9a-fA-F]+)")
_CC_RE = re.compile(r"\bcc=(?P<value>\d+)")
_TBS_RE = re.compile(r"\btbs=(?P<value>\d+)")
_MASK_RE = re.compile(r"\bmask=0x(?P<value>[0-9a-fA-F]+)")
_BUFFER_RE = re.compile(r"\bbuffer=(?P<before>\d+)/(?P<after>\d+)")
_TTI_DL_RE = re.compile(r"\btti_tx_dl=(?P<value>\d+)")
_PRB_RE = re.compile(r"\bprb=\[(?P<start>\d+),\s*(?P<stop>\d+)\)")
_BSR_RE = re.compile(r"\bbsr=(?P<after>-?\d+)(?:\s*\((?P<reported>-?\d+)-(?P<inflight>-?\d+)\))?")
_TTI_UL_RE = re.compile(r"\btti_tx_ul=(?P<value>\d+)")
_POLICY_RE = re.compile(r"Using (?P<desc>.+?) scheduling policy for cc=(?P<cc>\d+)")


@dataclass(frozen=True, slots=True)
class Allocation:
    """Une allocation PDSCH (DL) ou PUSCH (UL) lue dans le log.

    Attributes:
        direction: Voie de l'allocation.
        rnti: Identifiant radio de l'UE.
        carrier_index: Indice de porteuse (champ ``cc``).
        is_retransmission: Vrai pour une ligne ``DL retx`` / ``UL retx``.
        resource_mask: Ressources allouées, bit ``k`` = RBG ``k`` (DL) ou
            PRB ``k`` (UL).
        tbs_bytes: Taille du bloc de transport, en octets.
        pending_before_bytes: Données en attente avant l'allocation (DL :
            buffer RLC ; UL : dernier BSR reçu).
        pending_after_bytes: Données encore en attente après l'allocation
            (DL : buffer RLC restant ; UL : BSR moins les octets accordés).
        tti: TTI d'émission tel qu'écrit dans le log (``tti_tx_dl`` ou
            ``tti_tx_ul``), dans [0, 10240).
        tti_unwrapped: TTI rendu monotone (rebouclage à 10240 supprimé).
    """

    direction: Direction
    rnti: int
    carrier_index: int
    is_retransmission: bool
    resource_mask: int
    tbs_bytes: int
    pending_before_bytes: int
    pending_after_bytes: int
    tti: int
    tti_unwrapped: int

    @property
    def nof_resources(self) -> int:
        """Nombre de RBG (DL) ou de PRB (UL) alloués."""
        return self.resource_mask.bit_count()


#: Ancien nom, conservé pour les scripts écrits avec la version DL seule.
DlAllocation = Allocation


def _prb_mask(start: int, stop: int) -> int:
    """Masque des PRB ``[start, stop)``."""
    return ((1 << (stop - start)) - 1) << start if stop > start else 0


def _parse_dl(line: str, is_retx: bool) -> Allocation | None:
    rnti_m, cc_m, mask_m, tbs_m, buffer_m, tti_m = (
        _RNTI_RE.search(line),
        _CC_RE.search(line),
        _MASK_RE.search(line),
        _TBS_RE.search(line),
        _BUFFER_RE.search(line),
        _TTI_DL_RE.search(line),
    )
    if (
        rnti_m is None
        or cc_m is None
        or mask_m is None
        or tbs_m is None
        or buffer_m is None
        or tti_m is None
    ):
        return None
    tti = int(tti_m["value"])
    return Allocation(
        direction=Direction.DL,
        rnti=int(rnti_m["value"], 16),
        carrier_index=int(cc_m["value"]),
        is_retransmission=is_retx,
        resource_mask=int(mask_m["value"], 16),
        tbs_bytes=int(tbs_m["value"]),
        pending_before_bytes=int(buffer_m["before"]),
        pending_after_bytes=int(buffer_m["after"]),
        tti=tti,
        tti_unwrapped=tti,
    )


def _parse_ul(line: str, is_retx: bool) -> Allocation | None:
    rnti_m, cc_m, prb_m, tbs_m, bsr_m, tti_m = (
        _RNTI_RE.search(line),
        _CC_RE.search(line),
        _PRB_RE.search(line),
        _TBS_RE.search(line),
        _BSR_RE.search(line),
        _TTI_UL_RE.search(line),
    )
    if (
        rnti_m is None
        or cc_m is None
        or prb_m is None
        or tbs_m is None
        or bsr_m is None
        or tti_m is None
    ):
        return None
    tti = int(tti_m["value"])
    pending_after = max(0, int(bsr_m["after"]))
    reported = bsr_m["reported"]
    return Allocation(
        direction=Direction.UL,
        rnti=int(rnti_m["value"], 16),
        carrier_index=int(cc_m["value"]),
        is_retransmission=is_retx,
        resource_mask=_prb_mask(int(prb_m["start"]), int(prb_m["stop"])),
        tbs_bytes=int(tbs_m["value"]),
        pending_before_bytes=max(0, int(reported)) if reported is not None else pending_after,
        pending_after_bytes=pending_after,
        tti=tti,
        tti_unwrapped=tti,
    )


def parse_line(line: str, direction: Direction = Direction.DL) -> Allocation | None:
    """Analyse une ligne du log.

    Args:
        line: Une ligne brute du fichier de log.
        direction: Voie recherchée.

    Returns:
        L'allocation lue, avec ``tti_unwrapped`` égal à ``tti`` (le dépliage
        est fait par :func:`parse_lines`), ou ``None`` si la ligne n'est pas
        une ligne ``SCHED: DL|UL tx/retx`` complète de cette voie. Les lignes
        ``Msg3`` (accès aléatoire) ne sont pas des allocations de la
        politique et sont ignorées.
    """
    header = _HEADER_RE[direction].search(line)
    if header is None:
        return None
    is_retx = header["kind"] == "retx"
    allocation = _parse_dl(line, is_retx) if direction is Direction.DL else _parse_ul(line, is_retx)
    if allocation is None:
        logger.debug("Ligne SCHED incomplète ignorée : %s", line.rstrip())
    return allocation


class _Unwrapper:
    """Rend le TTI monotone : +10240 à chaque recul de plus d'une demi-période.

    Un petit recul (lignes d'un même TTI écrites dans le désordre) n'est pas
    pris pour un rebouclage.
    """

    def __init__(self) -> None:
        self._offset = 0
        self._previous: int | None = None

    def __call__(self, allocation: Allocation) -> Allocation:
        tti = allocation.tti
        if self._previous is not None and self._previous - tti > TTI_WRAP_PERIOD // 2:
            self._offset += TTI_WRAP_PERIOD
        self._previous = tti
        return replace(allocation, tti_unwrapped=tti + self._offset)


def parse_lines(
    lines: Iterable[str], carrier_index: int = 0, direction: Direction = Direction.DL
) -> Iterator[Allocation]:
    """Extrait les allocations d'une voie et d'une porteuse, avec un TTI déplié.

    Args:
        lines: Lignes du log, dans l'ordre du fichier.
        carrier_index: Porteuse à garder (champ ``cc``).
        direction: Voie à extraire.

    Yields:
        Les allocations, dans l'ordre du fichier.
    """
    unwrap = _Unwrapper()
    for line in lines:
        allocation = parse_line(line, direction)
        if allocation is not None and allocation.carrier_index == carrier_index:
            yield unwrap(allocation)


def parse_all_directions(
    lines: Iterable[str], carrier_index: int = 0
) -> dict[Direction, list[Allocation]]:
    """Extrait en une seule passe les allocations DL et UL d'une porteuse.

    Args:
        lines: Lignes du log, dans l'ordre du fichier.
        carrier_index: Porteuse à garder.

    Returns:
        Les allocations de chaque voie (liste vide si la voie est absente).
    """
    unwrappers = {direction: _Unwrapper() for direction in Direction}
    result: dict[Direction, list[Allocation]] = {direction: [] for direction in Direction}
    for line in lines:
        if "SCHED:" not in line:
            continue
        for direction in Direction:
            allocation = parse_line(line, direction)
            if allocation is not None:
                if allocation.carrier_index == carrier_index:
                    result[direction].append(unwrappers[direction](allocation))
                break
    return result


def read_log(
    path: Path, carrier_index: int = 0, direction: Direction = Direction.DL
) -> list[Allocation]:
    """Lit les allocations d'une voie dans un fichier de log srsENB.

    Args:
        path: Chemin du log (``/tmp/enb.log`` par défaut dans srsENB).
        carrier_index: Porteuse à analyser.
        direction: Voie à extraire.

    Returns:
        Les allocations, dans l'ordre du fichier.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    return read_log_all_directions(path, carrier_index)[direction]


def read_log_all_directions(
    path: Path, carrier_index: int = 0
) -> dict[Direction, list[Allocation]]:
    """Lit en une passe les allocations DL et UL d'un fichier de log.

    Args:
        path: Chemin du log.
        carrier_index: Porteuse à analyser.

    Returns:
        Les allocations de chaque voie.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Log introuvable : {path}")
    with path.open(encoding="utf-8", errors="replace") as log_file:
        allocations = parse_all_directions(log_file, carrier_index)
    for direction, items in allocations.items():
        logger.info(
            "%d lignes SCHED: %s lues dans %s (cc=%d)",
            len(items),
            direction.name,
            path,
            carrier_index,
        )
    return allocations


def find_declared_policy(lines: Iterable[str]) -> str | None:
    """Cherche la ligne de démarrage où srsENB annonce sa politique.

    ``sched_carrier.cc`` écrit ``Using time-domain RR scheduling policy for
    cc=0`` (ou ``PF``) au niveau info au moment de la configuration.

    Args:
        lines: Lignes du log.

    Returns:
        Le texte entre « Using » et « scheduling policy », ou ``None``.
    """
    for line in lines:
        match = _POLICY_RE.search(line)
        if match is not None:
            return match["desc"]
    return None
