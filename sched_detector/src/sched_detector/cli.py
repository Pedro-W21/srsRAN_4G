"""Détecte la politique de scheduling DL de srsENB à partir de son log.

Usage:
    sched-detect /tmp/enb.log
    sched-detect /tmp/enb.log --expect rr        # code de sortie 0 si RR détecté
    sched-detect /tmp/enb.log --expect custom    # ma politique ne doit pas passer
                                                 # pour une politique connue
    sched-detect /tmp/enb.log --cc 0 --nof-ues 3 --verbose

Le log doit contenir les lignes ``SCHED: DL tx`` (``mac_level = info`` dans
la section ``[log]`` de ``enb.conf``).

Codes de sortie : 0 = verdict obtenu (et conforme à ``--expect`` s'il est
donné), 1 = erreur d'entrée, 2 = verdict différent de ``--expect``,
3 = verdict indéterminé.

Author: Jérémie Caradec (projet RCA, INSA Lyon)
Date: 2026-10-14
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from sched_detector.classifier import Policy
from sched_detector.detector import Detection, detect
from sched_detector.log_parser import find_declared_policy, read_log

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path, help="Fichier de log srsENB (ex. /tmp/enb.log).")
    parser.add_argument("--cc", type=int, default=0, help="Porteuse analysée (défaut 0).")
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
    parser.add_argument("--verbose", action="store_true", help="Affiche les messages de debug.")
    return parser.parse_args(argv)


def _format_ratio(value: float) -> str:
    return "   -" if math.isnan(value) else f"{value:4.0%}"


def format_report(detection: Detection, declared_policy: str | None) -> str:
    """Met en forme le rapport texte affiché sur la sortie standard."""
    features, evidence, verdict = detection.features, detection.evidence, detection.verdict
    lines = [
        f"Politique détectée : {verdict.policy.value}  (confiance {verdict.confidence.value})",
    ]
    if not math.isnan(verdict.fairness_alpha):
        lines.append(f"Coefficient d'équité estimé : alpha ≈ {verdict.fairness_alpha:g}")
    if verdict.policy is Policy.CUSTOM:
        lines.append(
            f"Modèle connu le plus proche : {verdict.closest_model} "
            f"({verdict.closest_accuracy:.0%} des gagnants prédits)"
        )
    lines.append("")
    lines.append("Justification :")
    lines.extend(f"  - {reason}" for reason in verdict.reasons)
    lines.append("")
    lines.append(
        f"UE : {features.nof_ues} | TTI avec allocation : {features.nof_tti_with_allocation} | "
        f"TTI disputés : {features.nof_contested_tti} | reprises de trafic : "
        f"{features.nof_resume_events} | bande partagée : "
        f"{_format_ratio(features.greedy_violation_rate).strip()}"
    )
    lines.append(f"{'RNTI':>8} {'TTI gagnés':>11} {'part (régime établi)':>21} {'octets/RBG':>11}")
    for ue in features.ues:
        lines.append(
            f"{f'0x{ue.rnti:x}':>8} {ue.contested_wins:>11} "
            f"{ue.steady_share:>21.3f} {ue.bytes_per_rbg:>11.1f}"
        )
    lines.append("")
    lines.append(f"{'Modèle rejoué':<36} {'précision':>9} {'décisifs':>9} {'TTI décisifs':>13}")
    for model in (evidence.round_robin, *evidence.pf_family):
        lines.append(
            f"{model.name:<36} {_format_ratio(model.accuracy):>9} "
            f"{_format_ratio(model.decisive_accuracy):>9} {model.nof_decisive:>13}"
        )
    lines.append("")
    if declared_policy is None:
        lines.append("Ligne « Using ... scheduling policy » absente du log (non vérifiable).")
    else:
        lines.append(f"Politique annoncée par srsENB au démarrage : {declared_policy}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    """Exécute la détection et renvoie le code de sortie."""
    allocations = read_log(args.log, carrier_index=args.cc)
    detection = detect(allocations, nof_ues=args.nof_ues)
    with args.log.open(encoding="utf-8", errors="replace") as log_file:
        declared_policy = find_declared_policy(log_file)
    sys.stdout.write(format_report(detection, declared_policy) + "\n")

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
