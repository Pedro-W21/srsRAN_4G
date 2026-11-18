"""Détecte la politique de scheduling de srsENB à partir de son log (DL ou UL).

Usage:
    sched-detect /tmp/enb.log                    # voie choisie automatiquement
    sched-detect /tmp/enb.log --direction ul     # voie montante (iperf3 depuis les UE)
    sched-detect /tmp/enb.log --direction dl     # voie descendante (iperf3 vers les UE)
    sched-detect /tmp/enb.log --expect rr        # code de sortie 0 si RR détecté
    sched-detect /tmp/enb.log --expect custom    # ma politique ne doit pas passer
                                                 # pour une politique connue
    sched-detect /tmp/enb.log --cc 0 --nof-ues 3 --verbose
    sched-detect /tmp/enb.log --jain-window 10   # équité à 10 ms
    sched-detect /tmp/enb.log --ignore-rnti 0x4b # UE connecté sans trafic

Le log doit contenir les lignes ``SCHED: DL tx`` et/ou ``SCHED: UL tx``
(``mac_level = info`` dans la section ``[log]`` de ``enb.conf``). Par défaut
(``--direction auto``), la voie analysée est celle qui a le plus de TTI
disputés : c'est là que la politique a dû choisir.

Codes de sortie : 0 = verdict obtenu (et conforme à ``--expect`` s'il est
donné), 1 = erreur d'entrée, 2 = verdict différent de ``--expect``,
3 = verdict indéterminé.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14 (voie montante : 2026-11-18)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from sched_detector.classifier import RR_MIN_ACCURACY, Policy
from sched_detector.detector import Detection, DirectionChoice, choose_direction, detect
from sched_detector.fairness import DEFAULT_JAIN_WINDOW_TTI
from sched_detector.interpretation import (
    describe_closest_model,
    effective_ues,
    favourite_rnti,
    format_percent,
    format_points,
    gain_points,
    status_reason,
    summarize,
)
from sched_detector.log_parser import (
    Allocation,
    Direction,
    find_declared_policy,
    read_log_all_directions,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_UNEXPECTED_POLICY = 2
EXIT_INCONCLUSIVE = 3

_EXPECT_CHOICES: dict[str, Policy] = {
    "rr": Policy.ROUND_ROBIN,
    "pf": Policy.PROPORTIONAL_FAIR,
    "maxci": Policy.MAX_CI,
    "bet": Policy.BLIND_EQUAL_THROUGHPUT,
    "custom": Policy.CUSTOM,
}

_DIRECTION_AUTO = "auto"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path, help="Fichier de log srsENB (ex. /tmp/enb.log).")
    parser.add_argument("--cc", type=int, default=0, help="Porteuse analysée (défaut 0).")
    parser.add_argument(
        "--direction",
        choices=[_DIRECTION_AUTO, *(d.value for d in Direction)],
        default=_DIRECTION_AUTO,
        help="Voie analysée : dl, ul, ou auto (défaut) = celle qui a le plus de TTI disputés.",
    )
    parser.add_argument(
        "--nof-ues",
        type=int,
        default=None,
        help="Nombre d'UE connectés, si certains n'apparaissent pas dans le log.",
    )
    parser.add_argument(
        "--expect",
        choices=sorted(_EXPECT_CHOICES),
        default=None,
        help="Politique attendue : le code de sortie vaut 2 si elle n'est pas détectée.",
    )
    parser.add_argument(
        "--jain-window",
        type=int,
        default=DEFAULT_JAIN_WINDOW_TTI,
        metavar="TTI",
        help=(
            "Fenêtre de l'indice de Jain à court terme, en TTI "
            f"(défaut {DEFAULT_JAIN_WINDOW_TTI}, constante de temps du PF de srsRAN)."
        ),
    )
    parser.add_argument(
        "--ignore-rnti",
        type=_parse_rnti,
        action="append",
        default=[],
        metavar="RNTI",
        help=(
            "UE à écarter de l'analyse (ex. 0x4b), typiquement un UE connecté sans trafic "
            "qui empêche toute « pleine contention ». Il reste compté dans le N du modulo "
            "RR. Répétable."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Affiche les messages de debug.")
    args = parser.parse_args(argv)
    if args.jain_window < 1:
        parser.error(f"--jain-window doit être >= 1, reçu {args.jain_window}")
    return args


def _parse_rnti(text: str) -> int:
    """Lit un RNTI en hexadécimal (``0x4b``) ou en décimal (``75``)."""
    try:
        return int(text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"RNTI invalide : {text!r} (ex. 0x4b)") from error


#: Largeur des paragraphes de la section « En bref ».
_REPORT_WIDTH: int = 100


def _format_ratio(value: float) -> str:
    return "   -" if math.isnan(value) else f"{value:4.0%}"


def _format_index(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.3f}"


def _format_fairness(detection: Detection) -> list[str]:
    """Section « Équité (indice de Jain) » du rapport.

    Chaque indice J est aussi traduit en ``J x n`` UE servis à égalité, plus
    parlant que J seul : avec 3 UE, J = 0,374 veut dire « autant d'équité
    que si 1,1 UE sur 3 était servi ».
    """
    fairness = detection.fairness
    unit = detection.features.direction.resource_unit
    nof_ues = len(fairness.ues)

    def row(label: str, jain: float, note: str) -> str:
        value = f"{_format_index(jain):>5}"
        equivalent = effective_ues(jain, nof_ues)
        return f"  {label:<38} : {value}   {equivalent or '-':<32} {note}"

    lines = [
        f"Équité (indice de Jain : 1 = tous les UE reçoivent autant, 1/{nof_ues} = un seul "
        "UE reçoit tout) :",
        row("débit sur tout le log", fairness.jain_throughput, "(dépend du trafic offert)"),
        row(
            "débit en pleine contention",
            fairness.jain_contention_throughput,
            f"({fairness.nof_contention_tti} TTI où tous les UE attendaient)",
        ),
        row(
            f"{unit} en pleine contention",
            fairness.jain_contention_resources,
            "(partage des ressources, hors canal)",
        ),
        row(
            f"court terme, fenêtres de {fairness.window_tti} TTI",
            fairness.jain_short_term,
            f"(min {_format_index(fairness.jain_short_term_min)}, {fairness.nof_windows} fenêtres)",
        ),
        f"{'RNTI':>8} {'débit moyen (Mbit/s)':>21} {'octets (contention)':>20} "
        f"{unit + ' (contention)':>17}",
    ]
    lines.extend(
        f"{f'0x{ue.rnti:x}':>8} {ue.mean_throughput_mbps:>21.2f} "
        f"{ue.contention_bytes:>20} {ue.contention_resources:>17}"
        for ue in fairness.ues
    )
    return lines


def _format_direction(direction: Direction, choice: DirectionChoice | None) -> str:
    line = f"Voie analysée : {direction.label}"
    if choice is None:
        return line
    counts = ", ".join(
        f"{choice.contested_tti[d]} en {d.name}" for d in Direction if d in choice.contested_tti
    )
    return f"{line}, choisie automatiquement (TTI disputés : {counts})"


def _bullet(text: str) -> list[str]:
    return textwrap.wrap(text, width=_REPORT_WIDTH, initial_indent="  - ", subsequent_indent="    ")


def _format_models(detection: Detection) -> list[str]:
    """Tableau des politiques connues, chaque score à côté de sa référence."""
    evidence = detection.evidence
    favourite = favourite_rnti(detection)
    favourite_label = "le favori" if favourite is None else f"0x{favourite:x}"
    lines = [
        "Comparaison avec les politiques connues (sur les décisions où chaque modèle a un avis",
        "net) :",
        "  a raison : part des décisions du scheduler que le modèle reproduit ;",
        f"  favori   : part qu'on reproduirait en désignant toujours l'UE qui gagne le plus "
        f"({favourite_label}) ;",
        "  gain     : différence entre les deux. Un modèle ne ressemble au scheduler que s'il a",
        f"             raison au moins {RR_MIN_ACCURACY:.0%} du temps avec un gain nettement "
        "positif.",
        f"{'Politique connue':<32} {'décisions':>9} {'a raison':>9} {'favori':>7} "
        f"{'gain':>8}  verdict",
    ]
    lines.extend(
        f"{model.name:<32} {model.nof_decisive:>9} "
        f"{format_percent(model.decisive_accuracy):>9} "
        f"{format_percent(model.judged_naive_accuracy):>7} "
        f"{format_points(gain_points(model)):>8}  {status_reason(model)}"
        for model in (evidence.round_robin, *evidence.pf_family)
    )
    return lines


def format_report(
    detection: Detection,
    declared_policy: str | None,
    direction_choice: DirectionChoice | None = None,
    ignored_rntis: Sequence[int] = (),
) -> str:
    """Met en forme le rapport texte affiché sur la sortie standard.

    Args:
        detection: Résultat de :func:`~sched_detector.detector.detect`.
        declared_policy: Politique annoncée par srsENB, ou ``None``.
        direction_choice: Choix automatique de la voie, pour l'expliquer
            dans le rapport ; ``None`` si la voie a été imposée.
        ignored_rntis: UE écartés avec ``--ignore-rnti``.

    Returns:
        Le rapport, sans saut de ligne final.
    """
    features, verdict = detection.features, detection.verdict
    unit = features.direction.resource_unit
    lines = [
        f"Politique détectée : {verdict.policy.value}  (confiance {verdict.confidence.value})",
        _format_direction(features.direction, direction_choice),
    ]
    if ignored_rntis:
        lines.append(
            f"UE écartés de l'analyse : {', '.join(f'0x{r:x}' for r in ignored_rntis)} "
            f"(N = {features.nof_ues} pour le modulo du RR)"
        )
    if not math.isnan(verdict.fairness_alpha):
        lines.append(f"Coefficient d'équité estimé : alpha ≈ {verdict.fairness_alpha:g}")
    if verdict.policy is Policy.CUSTOM:
        lines.append(describe_closest_model(detection))
    lines.append("")
    lines.append("En bref :")
    for sentence in summarize(detection, ignored_rntis):
        lines.extend(_bullet(sentence))
    lines.append("")
    lines.append("Justification détaillée :")
    for reason in verdict.reasons:
        lines.extend(_bullet(reason))
    lines.append("")
    lines.append(
        f"UE : {features.nof_ues} | TTI avec allocation : {features.nof_tti_with_allocation} | "
        f"TTI disputés : {features.nof_contested_tti} | reprises de trafic : "
        f"{features.nof_resume_events} | bande partagée : "
        f"{_format_ratio(features.greedy_violation_rate).strip()}"
    )
    lines.append(
        f"{'RNTI':>8} {'TTI gagnés':>11} {'part (régime établi)':>21} {'octets/' + unit:>11}"
    )
    for ue in features.ues:
        lines.append(
            f"{f'0x{ue.rnti:x}':>8} {ue.contested_wins:>11} "
            f"{ue.steady_share:>21.3f} {ue.bytes_per_resource:>11.1f}"
        )
    lines.append("")
    lines.extend(_format_fairness(detection))
    lines.append("")
    lines.extend(_format_models(detection))
    lines.append("")
    if declared_policy is None:
        lines.append("Ligne « Using ... scheduling policy » absente du log (non vérifiable).")
    else:
        lines.append(f"Politique annoncée par srsENB au démarrage : {declared_policy}")
    return "\n".join(lines)


def _select_direction(
    requested: str, allocations: dict[Direction, list[Allocation]], nof_ues: int | None
) -> tuple[Direction, DirectionChoice | None]:
    """Voie à analyser et, en mode automatique, la justification du choix.

    Raises:
        ValueError: Si la voie demandée n'a aucune allocation, ou si aucune
            voie n'en a.
    """
    if requested != _DIRECTION_AUTO:
        direction = Direction(requested)
        if not allocations[direction]:
            other = Direction.UL if direction is Direction.DL else Direction.DL
            raise ValueError(
                f"aucune ligne SCHED: {direction.name} pour cette porteuse "
                f"({len(allocations[other])} lignes SCHED: {other.name} : essayer "
                f"--direction {other.value} ou --direction auto)"
            )
        return direction, None
    choice = choose_direction(allocations, nof_ues)
    return choice.direction, choice


def _drop_ignored(
    allocations: dict[Direction, list[Allocation]], ignored: Sequence[int], nof_ues: int | None
) -> tuple[dict[Direction, list[Allocation]], dict[Direction, int | None]]:
    """Retire les UE ignorés, en gardant le N du modulo RR de chaque voie.

    ``sched_time_rr`` calcule son indice de départ sur ``ue_db.size()``,
    qui compte tous les UE connectés, y compris ceux qu'on écarte de
    l'analyse : N est donc mesuré avant le filtrage.
    """
    ignored_set = frozenset(ignored)
    kept: dict[Direction, list[Allocation]] = {}
    sizes: dict[Direction, int | None] = {}
    for direction, items in allocations.items():
        seen = {a.rnti for a in items if not a.is_retransmission and a.nof_resources > 0}
        sizes[direction] = nof_ues if nof_ues is not None or not ignored_set else len(seen)
        kept[direction] = [a for a in items if a.rnti not in ignored_set]
    return kept, sizes


def run(args: argparse.Namespace) -> int:
    """Exécute la détection et renvoie le code de sortie."""
    allocations, nof_ues_by_direction = _drop_ignored(
        read_log_all_directions(args.log, carrier_index=args.cc), args.ignore_rnti, args.nof_ues
    )
    direction, choice = _select_direction(args.direction, allocations, args.nof_ues)
    detection = detect(
        allocations[direction],
        nof_ues=nof_ues_by_direction[direction],
        jain_window_tti=args.jain_window,
    )
    with args.log.open(encoding="utf-8", errors="replace") as log_file:
        declared_policy = find_declared_policy(log_file)
    sys.stdout.write(format_report(detection, declared_policy, choice, args.ignore_rnti) + "\n")

    policy = detection.verdict.policy
    if policy is Policy.INCONCLUSIVE:
        return EXIT_INCONCLUSIVE
    if args.expect is not None and policy is not _EXPECT_CHOICES[args.expect]:
        logger.error("Attendu %s, détecté %s", _EXPECT_CHOICES[args.expect].value, policy.value)
        return EXIT_UNEXPECTED_POLICY
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée de ``sched-detect``."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s | %(name)s | %(message)s",
    )
    try:
        return run(args)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return EXIT_INPUT_ERROR
    except ValueError as error:
        logger.error("Log inexploitable : %s", error)
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
