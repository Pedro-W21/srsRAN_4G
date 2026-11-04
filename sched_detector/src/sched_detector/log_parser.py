"""Lecture des lignes ``SCHED: DL tx`` d'un log srsENB (srsRAN 4G).

Le scheduler écrit une ligne par allocation PDSCH au niveau ``info`` du logger
MAC (``srsenb/src/stack/mac/sched_grid.cc``, fonction
``set_dl_data_sched_result``). Exemple ::

    SCHED: DL tx rnti=0x46, cc=0, pid=2, mask=0x1ffff, dci=(2,8), n_rtx=0,
    cfi=1, tbs=4584, buffer=12000/7400, tti_tx_dl=5123

Les champs sont lus un par un (et non avec une seule expression régulière
figée) pour tolérer un ordre ou un champ différent selon la version.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Période du compteur de TTI LTE : 1024 trames x 10 sous-trames (36.211 §4.1).
TTI_WRAP_PERIOD: int = 10_240

_HEADER_RE = re.compile(r"SCHED:\s+DL\s+(?P<kind>tx|retx)\b")
_RNTI_RE = re.compile(r"\brnti=0x(?P<value>[0-9a-fA-F]+)")
_CC_RE = re.compile(r"\bcc=(?P<value>\d+)")
_MASK_RE = re.compile(r"\bmask=0x(?P<value>[0-9a-fA-F]+)")
_TBS_RE = re.compile(r"\btbs=(?P<value>\d+)")
_BUFFER_RE = re.compile(r"\bbuffer=(?P<before>\d+)/(?P<after>\d+)")
_TTI_RE = re.compile(r"\btti_tx_dl=(?P<value>\d+)")


@dataclass(frozen=True, slots=True)
class DlAllocation:
    """Une allocation PDSCH lue dans le log.

    Attributes:
        rnti: Identifiant radio de l'UE.
        carrier_index: Indice de porteuse (champ ``cc``).
        is_retransmission: Vrai pour une ligne ``DL retx``.
        rbg_mask: Masque des RBG alloués (bit 0 = RBG 0).
        tbs_bytes: Taille du bloc de transport, en octets.
        buffer_before_bytes: Octets en attente avant l'allocation.
        buffer_after_bytes: Octets en attente après l'allocation.
        tti_tx_dl: TTI d'émission tel qu'écrit dans le log, dans [0, 10240).
        tti_unwrapped: TTI rendu monotone (rebouclage à 10240 supprimé).
    """

    rnti: int
    carrier_index: int
    is_retransmission: bool
    rbg_mask: int
    tbs_bytes: int
    buffer_before_bytes: int
    buffer_after_bytes: int
    tti_tx_dl: int
    tti_unwrapped: int

    @property
    def nof_rbg(self) -> int:
        """Nombre de RBG alloués (poids de Hamming du masque)."""
        return self.rbg_mask.bit_count()


def parse_line(line: str) -> DlAllocation | None:
    """Analyse une ligne du log.

    Args:
        line: Une ligne brute du fichier de log.

    Returns:
        L'allocation lue, avec ``tti_unwrapped`` égal à ``tti_tx_dl``
        (le dépliage est fait par :func:`parse_lines`), ou ``None`` si la
        ligne n'est pas une ligne ``SCHED: DL tx/retx`` complète.
    """
    header = _HEADER_RE.search(line)
    if header is None:
        return None
    matches = (
        _RNTI_RE.search(line),
        _CC_RE.search(line),
        _MASK_RE.search(line),
        _TBS_RE.search(line),
        _BUFFER_RE.search(line),
        _TTI_RE.search(line),
    )
    rnti_m, cc_m, mask_m, tbs_m, buffer_m, tti_m = matches
    if (
        rnti_m is None
        or cc_m is None
        or mask_m is None
        or tbs_m is None
        or buffer_m is None
        or tti_m is None
    ):
        logger.debug("Ligne SCHED incomplète ignorée : %s", line.rstrip())
        return None
    tti = int(tti_m["value"])
    return DlAllocation(
        rnti=int(rnti_m["value"], 16),
        carrier_index=int(cc_m["value"]),
        is_retransmission=header["kind"] == "retx",
        rbg_mask=int(mask_m["value"], 16),
        tbs_bytes=int(tbs_m["value"]),
        buffer_before_bytes=int(buffer_m["before"]),
        buffer_after_bytes=int(buffer_m["after"]),
        tti_tx_dl=tti,
        tti_unwrapped=tti,
    )


def parse_lines(lines: Iterable[str], carrier_index: int = 0) -> Iterator[DlAllocation]:
    """Extrait les allocations DL d'une porteuse, avec un TTI déplié.

    Le TTI du log reboucle à 10240. On ajoute 10240 à chaque fois que la
    valeur recule de plus d'une demi-période ; un petit recul (lignes d'un
    même TTI écrites dans le désordre) n'est pas pris pour un rebouclage.

    Args:
        lines: Lignes du log, dans l'ordre du fichier.
        carrier_index: Porteuse à garder (champ ``cc``).

    Yields:
        Les allocations de cette porteuse, dans l'ordre du fichier.
    """
    wrap_offset = 0
    previous_tti: int | None = None
    for line in lines:
        allocation = parse_line(line)
        if allocation is None or allocation.carrier_index != carrier_index:
            continue
        tti = allocation.tti_tx_dl
        if previous_tti is not None and previous_tti - tti > TTI_WRAP_PERIOD // 2:
            wrap_offset += TTI_WRAP_PERIOD
        previous_tti = tti
        yield DlAllocation(
            rnti=allocation.rnti,
            carrier_index=allocation.carrier_index,
            is_retransmission=allocation.is_retransmission,
            rbg_mask=allocation.rbg_mask,
            tbs_bytes=allocation.tbs_bytes,
            buffer_before_bytes=allocation.buffer_before_bytes,
            buffer_after_bytes=allocation.buffer_after_bytes,
            tti_tx_dl=tti,
            tti_unwrapped=tti + wrap_offset,
        )


def read_log(path: Path, carrier_index: int = 0) -> list[DlAllocation]:
    """Lit un fichier de log srsENB.

    Args:
        path: Chemin du log (``/tmp/enb.log`` par défaut dans srsENB).
        carrier_index: Porteuse à analyser.

    Returns:
        Les allocations DL de la porteuse, dans l'ordre du fichier.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Log introuvable : {path}")
    with path.open(encoding="utf-8", errors="replace") as log_file:
        allocations = list(parse_lines(log_file, carrier_index))
    logger.info("%d lignes SCHED: DL lues dans %s (cc=%d)", len(allocations), path, carrier_index)
    return allocations


_POLICY_RE = re.compile(r"Using (?P<desc>.+?) scheduling policy for cc=(?P<cc>\d+)")


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
