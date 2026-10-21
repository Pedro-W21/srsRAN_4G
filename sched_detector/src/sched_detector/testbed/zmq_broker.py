"""Broker ZMQ multi-UE pour srsRAN 4G : un srsENB, plusieurs srsUE, sans GNU Radio.

La radio virtuelle ZMQ de srsRAN relie un émetteur et un récepteur, pas plus
(``lib/src/phy/rf/rf_zmq_imp_*.c``). Cette broker se place au milieu :

* voie descendante : elle tire les échantillons de l'eNB et les distribue à
  chaque UE ;
* voie montante : elle tire les échantillons de chaque UE, les additionne à
  l'instant voulu et les sert à l'eNB.

Protocole ZMQ de srsRAN (vérifié dans le code) : chaque émetteur ouvre un
socket REP (``bind`` sur ``tx_port``) et attend une requête d'un octet avant
d'envoyer un bloc d'échantillons ``complex64`` (fc32). Chaque récepteur ouvre
un socket REQ (``connect`` sur ``rx_port``). La broker joue donc le rôle de
récepteur (REQ) face aux émetteurs, et d'émetteur (REP) face aux récepteurs.

Alignement temporel. L'émetteur srsRAN ne compte un échantillon qu'une fois
envoyé, et ``rf_zmq_tx_align`` comble l'émission avec des zéros pour qu'elle
suive la réception. L'échantillon montant k d'un UE correspond donc à son
échantillon descendant k, c'est-à-dire à l'instant ``offset_UE + k`` côté eNB,
``offset_UE`` étant l'indice du premier échantillon descendant qu'il a reçu.
La broker additionne chaque UE à ce décalage. Deux règles gardent le système
au même pas, comme une liaison ZMQ directe :

* la voie montante servie à l'eNB ne dépasse jamais ce que l'eNB a déjà émis ;
* dès qu'un UE est connecté, la broker l'attend au lieu de le remplacer par des
  zéros, y compris pendant l'initialisation de sa PHY. C'est indispensable :
  chaque nœud srsRAN dort la durée de ce qu'il lit (``usleep`` dans
  ``rf_zmq_recv_with_time_multi``) et ne tourne donc jamais plus vite que le
  temps réel ; un UE laissé en arrière ne rattraperait jamais son retard.
  L'eNB est simplement suspendu, comme avec une liaison ZMQ directe.

Un UE qui a émis puis se tait plus de ``ue_stall_timeout_s`` est écarté ; un
UE connecté qui n'émet jamais est écarté après ``ue_start_timeout_s``.

Ports par défaut, conformes à la doc srsRAN pour une broker :

* eNB : ``tx_port=tcp://*:2101``, ``rx_port=tcp://localhost:2100`` ;
* UE n (n = 1, 2, …) : ``tx_port=tcp://*:2001+10(n-1)``,
  ``rx_port=tcp://localhost:2000+10(n-1)``.

Usage:
    srsran-zmq-broker --ues 2
    srsran-zmq-broker --ues 3 --base-srate 11.52e6 --verbose

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-21
"""

from __future__ import annotations

import argparse
import enum
import functools
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import zmq

logger = logging.getLogger(__name__)

SampleArray = npt.NDArray[np.complex64]

#: Valeur par défaut de ``base_srate`` dans le pilote ZMQ de srsRAN, en Hz.
DEFAULT_BASE_SRATE_HZ: float = 23.04e6
DEFAULT_ENB_TX: str = "tcp://localhost:2101"
DEFAULT_ENB_RX_BIND: str = "tcp://*:2100"
DEFAULT_UE_BASE_PORT: int = 2000
DEFAULT_UE_PORT_STEP: int = 10
#: Période de scrutation des sockets, en ms : borne le temps d'arrêt.
POLL_PERIOD_MS: int = 100
#: Silence montant au-delà duquel un UE est écarté, en s. Inférieur au
#: ``trx_timeout_ms`` de srsRAN (2000 ms) pour que l'eNB ne reste jamais
#: bloqué assez longtemps pour signaler une déconnexion.
DEFAULT_UE_STALL_TIMEOUT_S: float = 1.5
#: Délai accordé à un UE connecté pour commencer à émettre, en s (temps
#: d'initialisation de sa PHY, plusieurs secondes sur une machine chargée).
DEFAULT_UE_START_TIMEOUT_S: float = 60.0
#: Retard descendant maximal accumulé pour un UE, en ms, avant de l'écarter.
DEFAULT_MAX_DL_BACKLOG_MS: float = 2_000.0
#: Durée d'un bloc montant servi à l'eNB, en ms (une sous-trame LTE).
DEFAULT_UL_CHUNK_MS: float = 1.0
#: Intervalle de reconnexion ZMQ, en ms (100 ms par défaut dans libzmq).
RECONNECT_INTERVAL_MS: int = 10
#: Pas d'attente d'un UE en retard sur la voie montante, en s.
UL_WAIT_STEP_S: float = 0.005
#: Période d'affichage des statistiques, en s (0 = jamais).
DEFAULT_STATS_PERIOD_S: float = 5.0
#: Taille maximale d'un message lu par srsRAN : ``ZMQ_MAX_BUFFER_SIZE`` de
#: ``rf_zmq_imp_trx.h`` (3 072 000 échantillons).
SRSRAN_MAX_MESSAGE_SAMPLES: int = 3_072_000


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    """Paramètres de la broker.

    Attributes:
        nof_ues: Nombre d'UE à relier (au moins 1).
        enb_tx: Adresse du ``tx_port`` de l'eNB, à laquelle se connecter.
        enb_rx_bind: Adresse à ouvrir pour le ``rx_port`` de l'eNB.
        ue_base_port: Port de réception du premier UE ; son port d'émission
            est ``ue_base_port + 1``.
        ue_port_step: Écart de ports entre deux UE consécutifs.
        ue_host: Hôte des ``tx_port`` des UE.
        base_srate_hz: ``base_srate`` passé à srsRAN, en Hz.
        ue_stall_timeout_s: Silence montant toléré pour un UE qui a déjà
            émis, en s.
        ue_start_timeout_s: Délai pour qu'un UE connecté commence à émettre,
            en s.
        max_dl_backlog_ms: Retard descendant toléré pour un UE, en ms.
        ul_chunk_ms: Taille des blocs montants servis à l'eNB, en ms.
        stats_period_s: Période des statistiques, en s (0 = désactivé).
    """

    nof_ues: int
    enb_tx: str = DEFAULT_ENB_TX
    enb_rx_bind: str = DEFAULT_ENB_RX_BIND
    ue_base_port: int = DEFAULT_UE_BASE_PORT
    ue_port_step: int = DEFAULT_UE_PORT_STEP
    ue_host: str = "localhost"
    base_srate_hz: float = DEFAULT_BASE_SRATE_HZ
    ue_stall_timeout_s: float = DEFAULT_UE_STALL_TIMEOUT_S
    ue_start_timeout_s: float = DEFAULT_UE_START_TIMEOUT_S
    max_dl_backlog_ms: float = DEFAULT_MAX_DL_BACKLOG_MS
    ul_chunk_ms: float = DEFAULT_UL_CHUNK_MS
    stats_period_s: float = DEFAULT_STATS_PERIOD_S

    def __post_init__(self) -> None:
        if self.nof_ues < 1:
            raise ValueError(f"nof_ues doit être >= 1, reçu {self.nof_ues}")
        if self.base_srate_hz <= 0:
            raise ValueError(f"base_srate_hz doit être > 0, reçu {self.base_srate_hz}")
        if not 0 < self.ul_chunk_samples <= SRSRAN_MAX_MESSAGE_SAMPLES:
            raise ValueError(
                f"ul_chunk_ms donne {self.ul_chunk_samples} échantillons, hors de "
                f"]0, {SRSRAN_MAX_MESSAGE_SAMPLES}]"
            )

    @property
    def ul_chunk_samples(self) -> int:
        """Taille d'un bloc montant, en échantillons."""
        return round(self.base_srate_hz * self.ul_chunk_ms / 1_000)

    @property
    def max_dl_backlog_samples(self) -> int:
        """Retard descendant toléré, en échantillons."""
        return round(self.base_srate_hz * self.max_dl_backlog_ms / 1_000)

    def ue_rx_bind(self, ue_index: int) -> str:
        """Adresse à ouvrir pour le ``rx_port`` de l'UE ``ue_index`` (0-based)."""
        return f"tcp://*:{self.ue_base_port + ue_index * self.ue_port_step}"

    def ue_tx_connect(self, ue_index: int) -> str:
        """Adresse du ``tx_port`` de l'UE ``ue_index`` (0-based)."""
        port = self.ue_base_port + 1 + ue_index * self.ue_port_step
        return f"tcp://{self.ue_host}:{port}"


class SampleFifo:
    """File d'échantillons repérés par un indice absolu.

    Les blocs sont gardés tels quels (pas de copie à l'ajout) ; ``read``
    assemble une tranche ``[start, stop)`` quel que soit le découpage.
    """

    def __init__(self) -> None:
        self._chunks: deque[SampleArray] = deque()
        self._first_index = 0
        self._end_index = 0

    @property
    def first_index(self) -> int:
        """Indice absolu du plus ancien échantillon conservé."""
        return self._first_index

    @property
    def end_index(self) -> int:
        """Indice absolu qui suit le dernier échantillon reçu."""
        return self._end_index

    def append(self, chunk: SampleArray) -> None:
        """Ajoute un bloc à la fin du flux."""
        if chunk.size:
            self._chunks.append(chunk)
            self._end_index += chunk.size

    def read(self, start: int, stop: int) -> SampleArray:
        """Copie la tranche ``[start, stop)``.

        Raises:
            ValueError: Si la tranche sort de ``[first_index, end_index]``.
        """
        if not self._first_index <= start <= stop <= self._end_index:
            raise ValueError(
                f"Tranche [{start}, {stop}) hors de [{self._first_index}, {self._end_index}]"
            )
        out = np.empty(stop - start, dtype=np.complex64)
        position = self._first_index
        written = 0
        for chunk in self._chunks:
            chunk_end = position + chunk.size
            if chunk_end > start and position < stop:
                lo = max(start, position) - position
                hi = min(stop, chunk_end) - position
                out[written : written + hi - lo] = chunk[lo:hi]
                written += hi - lo
            if chunk_end >= stop:
                break
            position = chunk_end
        return out

    def discard_before(self, index: int) -> None:
        """Oublie les échantillons d'indice inférieur à ``index``."""
        index = min(index, self._end_index)
        while self._chunks and self._first_index + self._chunks[0].size <= index:
            self._first_index += self._chunks.popleft().size
        if self._chunks and index > self._first_index:
            cut = index - self._first_index
            self._chunks[0] = self._chunks[0][cut:]
            self._first_index = index
        elif not self._chunks:
            self._first_index = self._end_index


class UeState(enum.Enum):
    """Cycle de vie d'un UE vu par la broker."""

    WAITING = "en attente"
    JOINING = "connexion"
    ACTIVE = "actif"


@dataclass(slots=True)
class UeLink:
    """État partagé d'un UE (protégé par la condition de la broker).

    Attributes:
        index: Indice 0-based de l'UE.
        state: Étape du cycle de vie.
        generation: Incrémenté à chaque (re)connexion : signale au fil
            montant qu'il doit recréer son socket.
        dl_offset: Indice eNB du premier échantillon descendant servi.
        dl_queue: Blocs descendants pas encore servis à l'UE.
        dl_backlog: Nombre d'échantillons dans ``dl_queue``.
        ul: Flux montant, indicé dans le temps de l'UE.
        last_ul_time: Instant (``time.monotonic``) du dernier bloc montant,
            ou de la connexion s'il n'a encore rien émis.
        dl_delivered: Échantillons descendants servis depuis la connexion :
            c'est l'indice de réception que l'UE atteindra au prochain bloc.
        needs_dl_resync: Vrai quand la file descendante a été vidée ; le
            prochain bloc redéfinit ``dl_offset``.
        dl_samples: Total d'échantillons descendants servis (statistiques).
        ul_samples: Total d'échantillons montants reçus (statistiques).
    """

    index: int
    state: UeState = UeState.WAITING
    generation: int = 0
    dl_offset: int = 0
    dl_queue: deque[SampleArray] = field(default_factory=deque)
    dl_backlog: int = 0
    ul: SampleFifo = field(default_factory=SampleFifo)
    last_ul_time: float = 0.0
    dl_delivered: int = 0
    needs_dl_resync: bool = False
    dl_samples: int = 0
    ul_samples: int = 0

    @property
    def label(self) -> str:
        """Nom affiché (UE1, UE2…)."""
        return f"UE{self.index + 1}"

    def ul_end_enb_time(self) -> int:
        """Instant eNB qui suit le dernier échantillon montant reçu."""
        return self.dl_offset + self.ul.end_index

    def reset(self) -> None:
        """Écarte l'UE : il devra se reconnecter, avec un flux montant neuf.

        Le changement de ``generation`` fait recréer le socket montant : les
        échantillons suivants viennent du nouveau processus UE, indice 0.
        """
        self.state = UeState.WAITING
        self.generation += 1
        self.dl_queue.clear()
        self.dl_backlog = 0
        self.dl_delivered = 0
        self.needs_dl_resync = False
        self.ul = SampleFifo()


def _request_reply(socket: zmq.Socket[bytes], stop: threading.Event) -> bytes | None:
    """Envoie une requête d'un octet et attend la réponse (``None`` si arrêt)."""
    socket.send(b"\xff")
    while not stop.is_set():
        if socket.poll(POLL_PERIOD_MS, zmq.POLLIN):
            return socket.recv()
    return None


def _wait_request(socket: zmq.Socket[bytes], stop: threading.Event) -> bool:
    """Attend une requête sur un socket REP (``False`` si arrêt)."""
    while not stop.is_set():
        if socket.poll(POLL_PERIOD_MS, zmq.POLLIN):
            socket.recv()
            return True
    return False


class Broker:
    """Relie un eNB et plusieurs UE srsRAN par ZMQ.

    Six familles de fils, chacun propriétaire de ses sockets (les sockets
    ZMQ ne sont pas partagés entre fils) :

    * ``enb-dl`` : tire la voie descendante de l'eNB ;
    * ``enb-ul`` : sert la voie montante additionnée à l'eNB ;
    * ``ueN-dl`` : sert la voie descendante à l'UE N ;
    * ``ueN-ul`` : tire la voie montante de l'UE N ;
    * ``stats`` : journal périodique des débits.
    """

    def __init__(self, config: BrokerConfig, context: zmq.Context[zmq.Socket[bytes]] | None = None):
        self._config = config
        self._context = context or zmq.Context.instance()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._ues = [UeLink(index=i) for i in range(config.nof_ues)]
        self._dl_total = 0
        self._ul_total = 0
        self._threads: list[threading.Thread] = []

    @property
    def dl_total(self) -> int:
        """Échantillons descendants reçus de l'eNB depuis le démarrage."""
        with self._condition:
            return self._dl_total

    @property
    def ul_total(self) -> int:
        """Échantillons montants servis à l'eNB depuis le démarrage."""
        with self._condition:
            return self._ul_total

    def ue_states(self) -> list[UeState]:
        """État courant de chaque UE."""
        with self._condition:
            return [ue.state for ue in self._ues]

    def wait_stopped(self, timeout_s: float) -> bool:
        """Attend l'arrêt de la broker ; ``True`` si elle est arrêtée."""
        return self._stop.wait(timeout_s)

    def start(self) -> None:
        """Démarre tous les fils (non bloquant)."""
        targets: list[tuple[str, Callable[[], None]]] = [
            ("enb-dl", self._run_enb_downlink),
            ("enb-ul", self._run_enb_uplink),
        ]
        for ue in self._ues:
            targets.append((f"ue{ue.index + 1}-dl", functools.partial(self._run_ue_downlink, ue)))
            targets.append((f"ue{ue.index + 1}-ul", functools.partial(self._run_ue_uplink, ue)))
        if self._config.stats_period_s > 0:
            targets.append(("stats", self._run_stats))
        for name, target in targets:
            thread = threading.Thread(target=self._guard(target, name), name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info(
            "Broker prête : eNB tx=%s rx=%s, %d UE", self._config.enb_tx,
            self._config.enb_rx_bind, self._config.nof_ues,
        )  # fmt: skip
        for ue in self._ues:
            logger.info(
                "  %s : rx_port=tcp://localhost:%s  tx_port=tcp://*:%s",
                ue.label,
                self._config.ue_rx_bind(ue.index).rsplit(":", 1)[1],
                self._config.ue_tx_connect(ue.index).rsplit(":", 1)[1],
            )

    def stop(self) -> None:
        """Arrête les fils et attend leur fin."""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=2 * POLL_PERIOD_MS / 1_000 + 1)

    def _guard(self, target: Callable[[], None], name: str) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
            except zmq.ZMQError as error:
                if not self._stop.is_set():
                    logger.exception("Fil %s arrêté par une erreur ZMQ : %s", name, error)
                    self._stop.set()

        return run

    def _socket(self, kind: int) -> zmq.Socket[bytes]:
        socket = self._context.socket(kind)
        socket.setsockopt(zmq.LINGER, 0)
        # Reconnexion rapide quand un nœud srsRAN démarre après la broker.
        socket.setsockopt(zmq.RECONNECT_IVL, RECONNECT_INTERVAL_MS)
        return socket

    # --- Voie descendante ----------------------------------------------------

    def _run_enb_downlink(self) -> None:
        socket = self._socket(zmq.REQ)
        socket.connect(self._config.enb_tx)
        try:
            while not self._stop.is_set():
                message = _request_reply(socket, self._stop)
                if message is None:
                    break
                self._distribute_downlink(np.frombuffer(message, dtype=np.complex64))

        finally:
            socket.close()

    def _distribute_downlink(self, chunk: SampleArray) -> None:
        with self._condition:
            start = self._dl_total
            self._dl_total += chunk.size
            for ue in self._ues:
                if ue.state is UeState.JOINING:
                    ue.dl_offset = start
                    ue.dl_delivered = 0
                    ue.state = UeState.ACTIVE
                    ue.last_ul_time = time.monotonic()
                    logger.info("%s connecté (instant eNB %d)", ue.label, start)
                if ue.state is not UeState.ACTIVE:
                    continue
                if ue.needs_dl_resync:
                    # L'échantillon reçu suivant aura l'indice dl_delivered
                    # chez l'UE et l'indice start chez l'eNB.
                    ue.dl_offset = start - ue.dl_delivered
                    ue.needs_dl_resync = False
                ue.dl_queue.append(chunk)
                ue.dl_backlog += chunk.size
                if ue.dl_backlog > self._config.max_dl_backlog_samples:
                    self._handle_dl_overflow(ue)
            self._condition.notify_all()

    def _handle_dl_overflow(self, ue: UeLink) -> None:
        backlog_ms = 1_000 * ue.dl_backlog / self._config.base_srate_hz
        if ue.ul.end_index == 0:
            # L'UE ne lit plus (initialisation de sa PHY) mais n'a encore rien
            # émis : on peut sauter le retard sans fausser l'alignement.
            logger.info(
                "%s s'initialise (%.0f ms non lus) : voie descendante resynchronisée",
                ue.label,
                backlog_ms,
            )
            ue.dl_queue.clear()
            ue.dl_backlog = 0
            ue.needs_dl_resync = True
            return
        logger.warning(
            "%s ne lit plus la voie descendante (%.0f ms de retard) : écarté. Relancer cet UE.",
            ue.label,
            backlog_ms,
        )
        ue.reset()

    def _run_ue_downlink(self, ue: UeLink) -> None:
        socket = self._socket(zmq.REP)
        socket.bind(self._config.ue_rx_bind(ue.index))
        try:
            while _wait_request(socket, self._stop):
                chunk = self._next_downlink_chunk(ue)
                if chunk is None:
                    break
                socket.send(memoryview(chunk))
        finally:
            socket.close()

    def _next_downlink_chunk(self, ue: UeLink) -> SampleArray | None:
        with self._condition:
            if ue.state is UeState.WAITING:
                # Première requête de l'UE : il démarre au prochain bloc de
                # l'eNB. Son socket montant n'est pas recréé, pour garder la
                # correspondance entre son indice d'émission et le nôtre.
                ue.state = UeState.JOINING
                self._condition.notify_all()
            while not self._stop.is_set():
                if ue.dl_queue:
                    chunk = ue.dl_queue.popleft()
                    ue.dl_backlog -= chunk.size
                    ue.dl_delivered += chunk.size
                    ue.dl_samples += chunk.size
                    return chunk
                if ue.state is UeState.WAITING:
                    # Écarté pendant l'attente : rejoindre comme un nouvel UE.
                    ue.state = UeState.JOINING
                self._condition.wait(POLL_PERIOD_MS / 1_000)
        return None

    # --- Voie montante -------------------------------------------------------

    def _run_ue_uplink(self, ue: UeLink) -> None:
        socket: zmq.Socket[bytes] | None = None
        generation = -1
        pending = False
        try:
            while not self._stop.is_set():
                with self._condition:
                    current_generation = ue.generation
                if socket is None or current_generation != generation:
                    # Nouveau processus UE : socket neuf, pour que le premier
                    # échantillon reçu soit bien son échantillon d'indice 0.
                    if socket is not None:
                        socket.close()
                    socket = self._socket(zmq.REQ)
                    socket.connect(self._config.ue_tx_connect(ue.index))
                    generation = current_generation
                    pending = False
                if not pending:
                    socket.send(b"\xff")
                    pending = True
                if not socket.poll(POLL_PERIOD_MS, zmq.POLLIN):
                    continue
                message = socket.recv()
                pending = False
                chunk = np.frombuffer(message, dtype=np.complex64)
                with self._condition:
                    if ue.generation != generation:
                        continue
                    ue.ul.append(chunk)
                    ue.ul_samples += chunk.size
                    ue.last_ul_time = time.monotonic()
                    self._condition.notify_all()
        finally:
            if socket is not None:
                socket.close()

    def _run_enb_uplink(self) -> None:
        socket = self._socket(zmq.REP)
        socket.bind(self._config.enb_rx_bind)
        try:
            while _wait_request(socket, self._stop):
                block = self._next_uplink_block()
                if block is None:
                    break
                socket.send(memoryview(block))
        finally:
            socket.close()

    def _next_uplink_block(self) -> SampleArray | None:
        """Construit le prochain bloc montant ``[m, end)`` pour l'eNB."""
        with self._condition:
            start = self._ul_total
            while not self._stop.is_set() and self._dl_total <= start:
                self._condition.wait(POLL_PERIOD_MS / 1_000)
            if self._stop.is_set():
                return None
            wanted_end = min(start + self._config.ul_chunk_samples, self._dl_total)
            end = self._wait_for_ues(start, wanted_end)
            if end is None:
                return None
            block = np.zeros(end - start, dtype=np.complex64)
            for ue in self._ues:
                if ue.state is UeState.ACTIVE:
                    self._mix_ue(ue, start, end, block)
            self._ul_total = end
            return block

    def _wait_for_ues(self, start: int, end: int) -> int | None:
        """Attend les UE actifs ; renvoie la fin réelle du bloc.

        Le silence d'un UE se mesure depuis le début de cette attente : quand
        un UE bloque la broker, les autres UE ne reçoivent plus de voie
        descendante et n'émettent plus non plus, sans être en faute.
        """
        wait_start = time.monotonic()
        while not self._stop.is_set():
            lagging = [
                ue for ue in self._ues if ue.state is UeState.ACTIVE and ue.ul_end_enb_time() < end
            ]
            if not lagging:
                return end
            covered = min(ue.ul_end_enb_time() for ue in lagging)
            if covered > start:
                # Servir la partie déjà disponible plutôt que bloquer l'eNB.
                return covered
            now = time.monotonic()
            for ue in lagging:
                silence_s = now - max(ue.last_ul_time, wait_start)
                has_emitted = ue.ul.end_index > 0
                limit_s = (
                    self._config.ue_stall_timeout_s
                    if has_emitted
                    else self._config.ue_start_timeout_s
                )
                if silence_s <= limit_s:
                    continue
                if has_emitted:
                    logger.warning(
                        "%s n'émet plus depuis %.1f s : écarté. Relancer cet UE.",
                        ue.label,
                        silence_s,
                    )
                else:
                    logger.warning(
                        "%s connecté mais muet depuis %.0f s : écarté. Relancer cet UE.",
                        ue.label,
                        silence_s,
                    )
                ue.reset()
            self._condition.wait(UL_WAIT_STEP_S)
        return None

    @staticmethod
    def _mix_ue(ue: UeLink, start: int, end: int, block: SampleArray) -> None:
        local_start = start - ue.dl_offset
        local_end = end - ue.dl_offset
        lo = max(local_start, ue.ul.first_index)
        hi = min(local_end, ue.ul.end_index)
        if hi > lo:
            block[lo - local_start : hi - local_start] += ue.ul.read(lo, hi)
        ue.ul.discard_before(local_end)

    # --- Statistiques --------------------------------------------------------

    def _run_stats(self) -> None:
        previous_dl = previous_ul = 0
        previous_time = time.monotonic()
        while not self._stop.wait(self._config.stats_period_s):
            now = time.monotonic()
            with self._condition:
                dl, ul = self._dl_total, self._ul_total
                states = ", ".join(f"{ue.label} {ue.state.value}" for ue in self._ues)
            elapsed = now - previous_time
            logger.info(
                "DL %.2f Méch/s | UL %.2f Méch/s | %s",
                (dl - previous_dl) / elapsed / 1e6,
                (ul - previous_ul) / elapsed / 1e6,
                states,
            )
            previous_dl, previous_ul, previous_time = dl, ul, now


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ues", type=int, default=2, help="Nombre d'UE (défaut 2).")
    parser.add_argument(
        "--base-srate",
        type=float,
        default=DEFAULT_BASE_SRATE_HZ,
        help="base_srate passé à srsRAN, en Hz (défaut 23.04e6).",
    )
    parser.add_argument("--enb-tx", default=DEFAULT_ENB_TX, help="tx_port de l'eNB.")
    parser.add_argument("--enb-rx-bind", default=DEFAULT_ENB_RX_BIND, help="rx_port de l'eNB.")
    parser.add_argument(
        "--ue-base-port", type=int, default=DEFAULT_UE_BASE_PORT, help="rx_port du 1er UE."
    )
    parser.add_argument(
        "--ue-port-step", type=int, default=DEFAULT_UE_PORT_STEP, help="Écart entre UE."
    )
    parser.add_argument(
        "--stats", type=float, default=DEFAULT_STATS_PERIOD_S, help="Période des stats (s)."
    )
    parser.add_argument("--verbose", action="store_true", help="Messages de debug.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée de ``srsran-zmq-broker``."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = BrokerConfig(
            nof_ues=args.ues,
            enb_tx=args.enb_tx,
            enb_rx_bind=args.enb_rx_bind,
            ue_base_port=args.ue_base_port,
            ue_port_step=args.ue_port_step,
            base_srate_hz=args.base_srate,
            stats_period_s=args.stats,
        )
    except ValueError as error:
        logger.error("%s", error)
        return 1
    broker = Broker(config)
    try:
        broker.start()
        while not broker.wait_stopped(0.5):
            pass
    except KeyboardInterrupt:
        logger.info("Arrêt demandé")
    except zmq.ZMQError as error:
        logger.error("Erreur ZMQ : %s (port déjà utilisé ?)", error)
        return 1
    finally:
        broker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
