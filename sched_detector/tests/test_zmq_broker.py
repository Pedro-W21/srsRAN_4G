"""Tests de la broker ZMQ multi-UE avec de faux eNB et UE au protocole srsRAN.

Le faux eNB émet une rampe (échantillon m = m) ; chaque faux UE renvoie en
voie montante exactement ce qu'il reçoit. Si la broker aligne correctement
les UE, l'eNB reçoit à l'instant m la valeur ``k * m``, k étant le nombre
d'UE synchronisés : le rapport reçu / m doit être un entier. Un décalage d'un
seul échantillon pour un UE donnerait un rapport non entier.
"""

from __future__ import annotations

import random
import socket as stdlib_socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import pytest

pytest.importorskip("numpy")
pytest.importorskip("zmq")

import numpy as np
import zmq

from sched_detector.testbed.zmq_broker import (
    Broker,
    BrokerConfig,
    SampleFifo,
    UeState,
)

ENB_CHUNK = 1_536
ENB_ITERATIONS = 240
POLL_MS = 50
#: Délai maximal d'une réponse attendue par les faux nœuds, en ms.
NODE_TIMEOUT_MS = 5_000


# --- SampleFifo ---------------------------------------------------------------


def _ramp(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.float32).astype(np.complex64)


def test_fifo_reads_across_chunk_boundaries() -> None:
    fifo = SampleFifo()
    for start, stop in [(0, 3), (3, 4), (4, 10)]:
        fifo.append(_ramp(start, stop))
    assert np.array_equal(fifo.read(2, 7), _ramp(2, 7))
    assert fifo.end_index == 10


def test_fifo_discard_keeps_absolute_indices() -> None:
    fifo = SampleFifo()
    fifo.append(_ramp(0, 5))
    fifo.append(_ramp(5, 9))
    fifo.discard_before(6)
    assert fifo.first_index == 6
    assert np.array_equal(fifo.read(6, 9), _ramp(6, 9))


def test_fifo_discard_beyond_end_empties_it() -> None:
    fifo = SampleFifo()
    fifo.append(_ramp(0, 4))
    fifo.discard_before(100)
    assert fifo.first_index == fifo.end_index == 4
    fifo.append(_ramp(4, 6))
    assert np.array_equal(fifo.read(4, 6), _ramp(4, 6))


def test_fifo_rejects_discarded_range() -> None:
    fifo = SampleFifo()
    fifo.append(_ramp(0, 8))
    fifo.discard_before(4)
    with pytest.raises(ValueError, match="hors de"):
        fifo.read(2, 6)


# --- BrokerConfig -------------------------------------------------------------


def test_config_computes_srsran_ports() -> None:
    config = BrokerConfig(nof_ues=2)
    assert config.ue_rx_bind(0) == "tcp://*:2000"
    assert config.ue_tx_connect(0) == "tcp://localhost:2001"
    assert config.ue_rx_bind(1) == "tcp://*:2010"
    assert config.ue_tx_connect(1) == "tcp://localhost:2011"
    assert config.ul_chunk_samples == 23_040


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"nof_ues": 0}, "nof_ues"), ({"nof_ues": 1, "base_srate_hz": -1.0}, "base_srate")],
)
def test_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BrokerConfig(**kwargs)  # type: ignore[arg-type]


# --- Faux nœuds srsRAN ----------------------------------------------------------


def _free_port_block() -> int:
    """Base de ports avec base, base+1, base+10, base+11, base+100, base+101 libres.

    Tirée sous la plage éphémère de Linux (32768+) pour ne pas croiser les
    ports clients ; ``SO_REUSEADDR`` comme libzmq, pour ignorer TIME_WAIT.
    """
    offsets = (0, 1, 10, 11, 100, 101)
    for _ in range(100):
        base = random.randrange(20_000, 32_000, 200)
        try:
            for offset in offsets:
                with stdlib_socket.socket() as probe:
                    probe.setsockopt(stdlib_socket.SOL_SOCKET, stdlib_socket.SO_REUSEADDR, 1)
                    probe.bind(("127.0.0.1", base + offset))
        except OSError:
            continue
        return base
    raise RuntimeError("Aucun bloc de ports libre")


@dataclass
class FakeEnb:
    """eNB minimal : émet une rampe par blocs, lit la voie montante."""

    tx_bind: str
    rx_connect: str
    iterations: int = ENB_ITERATIONS
    #: Pause par bloc, en s : un vrai srsRAN dort la durée du bloc lu
    #: (1 ms ici : 1536 échantillons à 1,536 Méch/s).
    pacing_s: float = 0.001
    received: list[np.ndarray] = field(default_factory=list)
    error: str | None = None

    def run(self, context: zmq.Context[zmq.Socket[bytes]]) -> None:
        tx = context.socket(zmq.REP)
        rx = context.socket(zmq.REQ)
        tx.setsockopt(zmq.LINGER, 0)
        rx.setsockopt(zmq.LINGER, 0)
        tx.bind(self.tx_bind)
        rx.connect(self.rx_connect)
        try:
            for iteration in range(self.iterations):
                start = iteration * ENB_CHUNK
                if not tx.poll(NODE_TIMEOUT_MS):
                    self.error = f"pas de requête DL à l'itération {iteration}"
                    return
                tx.recv()
                tx.send(_ramp(start, start + ENB_CHUNK).tobytes())
                needed = ENB_CHUNK
                while needed > 0:
                    rx.send(b"\xff")
                    if not rx.poll(NODE_TIMEOUT_MS):
                        self.error = f"pas de voie montante à l'itération {iteration}"
                        return
                    block = np.frombuffer(rx.recv(), dtype=np.complex64)
                    self.received.append(block)
                    needed -= block.size
                time.sleep(self.pacing_s)
        finally:
            tx.close()
            rx.close()

    def uplink(self) -> np.ndarray:
        return np.concatenate(self.received) if self.received else np.zeros(0, np.complex64)


@dataclass
class FakeUe:
    """UE minimal : renvoie en voie montante ce qu'il reçoit, en deux moitiés."""

    rx_connect: str
    tx_bind: str
    start_delay_s: float = 0.0
    max_chunks: int | None = None
    #: Pause après le premier bloc reçu, avant toute émission (initialisation
    #: lente de la PHY d'un vrai srsUE).
    init_pause_s: float = 0.0
    stop: threading.Event = field(default_factory=threading.Event)

    def run(self, context: zmq.Context[zmq.Socket[bytes]]) -> None:
        time.sleep(self.start_delay_s)
        rx = context.socket(zmq.REQ)
        tx = context.socket(zmq.REP)
        rx.setsockopt(zmq.LINGER, 0)
        tx.setsockopt(zmq.LINGER, 0)
        tx.bind(self.tx_bind)
        rx.connect(self.rx_connect)
        chunks = 0
        try:
            while not self.stop.is_set():
                if self.max_chunks is not None and chunks >= self.max_chunks:
                    return
                rx.send(b"\xff")
                while not rx.poll(POLL_MS):
                    if self.stop.is_set():
                        return
                samples = np.frombuffer(rx.recv(), dtype=np.complex64)
                is_first_chunk_paused = chunks == 0 and self.init_pause_s > 0
                if is_first_chunk_paused and self.stop.wait(self.init_pause_s):
                    return
                half = samples.size // 2
                for piece in (samples[:half], samples[half:]):
                    while not tx.poll(POLL_MS):
                        if self.stop.is_set():
                            return
                    tx.recv()
                    tx.send(piece.tobytes())
                chunks += 1
        finally:
            rx.close()
            tx.close()


@dataclass
class Scenario:
    broker: Broker
    enb: FakeEnb
    ues: list[FakeUe]


@pytest.fixture
def make_scenario() -> Iterator[object]:
    context = zmq.Context()
    threads: list[threading.Thread] = []
    brokers: list[Broker] = []
    fake_ues: list[FakeUe] = []

    def build(
        ue_specs: list[dict[str, object]],
        max_dl_backlog_ms: float = 2_000.0,
        ue_start_timeout_s: float = 10.0,
    ) -> Scenario:
        base = _free_port_block()
        config = BrokerConfig(
            nof_ues=len(ue_specs),
            enb_tx=f"tcp://127.0.0.1:{base + 101}",
            enb_rx_bind=f"tcp://127.0.0.1:{base + 100}",
            ue_base_port=base,
            ue_host="127.0.0.1",
            base_srate_hz=1.536e6,
            ue_stall_timeout_s=0.3,
            max_dl_backlog_ms=max_dl_backlog_ms,
            ue_start_timeout_s=ue_start_timeout_s,
            stats_period_s=0,
        )
        broker = Broker(config, context)
        brokers.append(broker)
        broker.start()
        ues = [
            FakeUe(
                rx_connect=f"tcp://127.0.0.1:{base + 10 * i}",
                tx_bind=f"tcp://127.0.0.1:{base + 10 * i + 1}",
                **spec,  # type: ignore[arg-type]
            )
            for i, spec in enumerate(ue_specs)
        ]
        fake_ues.extend(ues)
        enb = FakeEnb(f"tcp://127.0.0.1:{base + 101}", f"tcp://127.0.0.1:{base + 100}")
        runners: list[Callable[[zmq.Context[zmq.Socket[bytes]]], None]] = [
            *(ue.run for ue in ues),
            enb.run,
        ]
        for runner in runners:
            thread = threading.Thread(target=runner, args=(context,), daemon=True)
            thread.start()
            threads.append(thread)
        return Scenario(broker, enb, ues)

    yield build
    for ue in fake_ues:
        ue.stop.set()
    for broker in brokers:
        broker.stop()
    for thread in threads:
        thread.join(timeout=10)
    context.destroy(linger=0)


def _run_to_end(scenario: Scenario) -> np.ndarray:
    deadline = time.monotonic() + 60
    while len(scenario.enb.received) == 0 or (
        sum(b.size for b in scenario.enb.received) < ENB_ITERATIONS * ENB_CHUNK
        and scenario.enb.error is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
        if time.monotonic() > deadline:
            break
    assert scenario.enb.error is None, scenario.enb.error
    uplink = scenario.enb.uplink()
    assert uplink.size == ENB_ITERATIONS * ENB_CHUNK
    return uplink


def _active_ue_counts(uplink: np.ndarray) -> np.ndarray:
    """Nombre d'UE alignés à chaque instant (rapport reçu / émis)."""
    indices = np.arange(1, uplink.size, dtype=np.float64)
    ratio = uplink.real[1:].astype(np.float64) / indices
    counts = np.rint(ratio)
    misaligned = np.abs(ratio - counts) > 1e-6
    assert not misaligned.any(), (
        f"{int(misaligned.sum())} échantillons mal alignés, premier à l'instant "
        f"{int(np.argmax(misaligned)) + 1}"
    )
    assert np.all(np.abs(uplink.imag) == 0)
    return counts.astype(int)


def test_two_ues_started_first_are_both_aligned(make_scenario) -> None:  # type: ignore[no-untyped-def]
    scenario = make_scenario([{}, {}])
    counts = _active_ue_counts(_run_to_end(scenario))
    assert set(np.unique(counts)) <= {0, 1, 2}
    assert np.all(counts[len(counts) // 4 :] == 2)
    assert scenario.broker.ue_states() == [UeState.ACTIVE, UeState.ACTIVE]


def test_late_ue_joins_without_breaking_alignment(make_scenario) -> None:  # type: ignore[no-untyped-def]
    # L'eNB dure au moins 240 ms (240 blocs de 1 ms) : l'UE2 arrive en route.
    scenario = make_scenario([{}, {"start_delay_s": 0.12}])
    counts = _active_ue_counts(_run_to_end(scenario))
    assert counts[-1] == 2
    assert (counts == 1).any(), "l'UE tardif aurait dû manquer au début"
    assert np.all(np.diff(counts[counts > 0]) >= 0), "un UE a disparu puis réapparu"


def test_dead_ue_is_dropped_and_others_continue(make_scenario) -> None:  # type: ignore[no-untyped-def]
    scenario = make_scenario([{}, {"max_chunks": 60}])
    counts = _active_ue_counts(_run_to_end(scenario))
    assert (counts == 2).any()
    assert counts[-1] == 1
    assert scenario.broker.ue_states()[0] is UeState.ACTIVE
    assert scenario.broker.ue_states()[1] is not UeState.ACTIVE


def test_slow_starting_ue_holds_enb_until_it_emits(make_scenario) -> None:  # type: ignore[no-untyped-def]
    # Un vrai srsUE lit un premier bloc puis met plusieurs secondes à
    # initialiser sa PHY sans rien émettre. Les nœuds srsRAN ne tournent pas
    # plus vite que le temps réel : la broker doit suspendre l'eNB jusqu'à
    # ce que l'UE émette, sinon il ne rattraperait jamais son retard.
    scenario = make_scenario([{}, {"init_pause_s": 0.6}])
    counts = _active_ue_counts(_run_to_end(scenario))
    assert np.all(counts[len(counts) // 4 :] == 2)
    assert scenario.broker.ue_states() == [UeState.ACTIVE, UeState.ACTIVE]


def test_ue_that_never_emits_is_dropped_after_start_timeout(make_scenario) -> None:  # type: ignore[no-untyped-def]
    scenario = make_scenario([{}, {"init_pause_s": 30.0}], ue_start_timeout_s=0.5)
    counts = _active_ue_counts(_run_to_end(scenario))
    assert counts[-1] == 1
    assert scenario.broker.ue_states()[1] is not UeState.ACTIVE
