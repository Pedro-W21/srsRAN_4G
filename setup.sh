#!/bin/bash
# setup.sh : prepare un Debian 12 pour compiler et tester ce depot srsRAN 4G (ZMQ + GNU Radio)
# A placer a la racine du depot (ou dans un sous-dossier comme scripts/).
# Usage : ./setup.sh            (ne pas lancer avec sudo, le script l'appelle quand il faut)
#
# Le script compile le depot dans lequel il se trouve, dans <depot>/build.
# Il ne fait ni git clone ni git pull : fais ton git pull / checkout toi-meme avant.

set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  echo "Ne lance pas ce script en root : utilise un utilisateur normal avec sudo." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Dependances systeme -------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
# Repond "non" a la question "demarrer iperf3 en demon ?"
echo "iperf3 iperf3/start_daemon boolean false" | sudo debconf-set-selections

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake git ccache pkg-config ca-certificates \
  libfftw3-dev libmbedtls-dev libboost-program-options-dev \
  libboost-system-dev libboost-test-dev libboost-thread-dev \
  libconfig++-dev libsctp-dev libzmq3-dev libczmq-dev \
  iproute2 iptables iperf3 net-tools nano \
  gnuradio

# --- 2. Localisation de la racine du depot ----------------------------------
# (apres l'installation de git, qui sert a la trouver)
REPO_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_DIR" ] || [ ! -f "$REPO_DIR/CMakeLists.txt" ]; then
  echo "Erreur : ce script doit etre dans le depot srsRAN (CMakeLists.txt introuvable)." >&2
  exit 1
fi
echo "Depot : $REPO_DIR (branche : $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD))"

# --- 3. Compilation ----------------------------------------------------------
mkdir -p "$REPO_DIR/build"
cd "$REPO_DIR/build"
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache .. | tee cmake.log

if ! grep -qi "Found libZEROMQ" cmake.log; then
  echo "ATTENTION : ZeroMQ n'a pas ete detecte par cmake. Verifie libzmq3-dev." >&2
fi

make -j"$(nproc)"

# --- 4. Installation et fichiers de configuration ---------------------------
sudo make install
sudo ldconfig

# N'ecrase pas des configs existantes (ajouter --force pour les regenerer)
srsran_install_configs.sh user

# --- 5. Verifications --------------------------------------------------------
echo
echo "=== Verifications ==="
gnuradio-config-info --version || echo "GNU Radio : verifier l'installation"
command -v srsenb  >/dev/null && echo "srsenb  : OK" || echo "srsenb  : introuvable"
command -v srsue   >/dev/null && echo "srsue   : OK" || echo "srsue   : introuvable"
command -v srsepc  >/dev/null && echo "srsepc  : OK" || echo "srsepc  : introuvable"
echo "Configs : $HOME/.config/srsran/"
echo
echo "Setup termine. Rappels :"
echo " - sudo ip netns add ue1   (un namespace par UE)"
echo " - ordre : srsepc, srsenb, srsue, puis le flowgraph GNU Radio"
echo " - arret : UE d'abord, puis eNB ; redemarrer eNB et broker apres un detachement"
echo " - apres modification du scheduler : cd $REPO_DIR/build && make -j\$(nproc) srsenb"