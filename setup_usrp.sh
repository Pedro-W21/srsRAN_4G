#!/bin/bash
# setup.sh : prepare un Debian 12 pour compiler et tester ce depot srsRAN 4G (USRP / UHD)
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
# Suppression de libzmq3-dev, libczmq-dev et gnuradio
# Ajout de libuhd-dev et uhd-host
sudo apt-get install -y --no-install-recommends \
  build-essential cmake git ccache pkg-config ca-certificates \
  libfftw3-dev libmbedtls-dev libboost-program-options-dev \
  libboost-system-dev libboost-test-dev libboost-thread-dev \
  libconfig++-dev libsctp-dev \
  libuhd-dev uhd-host \
  iproute2 iptables iperf3 net-tools nano

# --- 2. Telechargement des images firmware USRP -----------------------------
echo "Telechargement des images FPGA/Firmware UHD..."
sudo uhd_images_downloader

# --- 3. Localisation de la racine du depot ----------------------------------
REPO_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$REPO_DIR" ] || [ ! -f "$REPO_DIR/CMakeLists.txt" ]; then
  echo "Erreur : ce script doit etre dans le depot srsRAN (CMakeLists.txt introuvable)." >&2
  exit 1
fi
echo "Depot : $REPO_DIR (branche : $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD))"

# --- 4. Compilation ----------------------------------------------------------
mkdir -p "$REPO_DIR/build"
cd "$REPO_DIR/build"
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache .. | tee cmake.log

if ! grep -qi "Found UHD" cmake.log; then
  echo "ATTENTION : Le driver UHD n'a pas ete detecte par cmake. Verifie libuhd-dev." >&2
fi

make -j"$(nproc)"

# --- 5. Installation et fichiers de configuration ---------------------------
sudo make install
sudo ldconfig

# N'ecrase pas des configs existantes (ajouter --force pour les regenerer)
srsran_install_configs.sh user

# --- 6. Verifications --------------------------------------------------------
echo
echo "=== Verifications ==="
command -v uhd_find_devices >/dev/null && echo "UHD     : OK" || echo "UHD     : introuvable"
command -v srsenb  >/dev/null && echo "srsenb  : OK" || echo "srsenb  : introuvable"
command -v srsue   >/dev/null && echo "srsue   : OK" || echo "srsue   : introuvable"
command -v srsepc  >/dev/null && echo "srsepc  : OK" || echo "srsepc  : introuvable"
echo "Configs : $HOME/.config/srsran/"
echo
echo "Setup termine. Rappels pour USRP :"
echo " - Branche ton USRP (USB ou Ethernet) et lance : uhd_find_devices"
echo " - Modifie les fichiers eNB.conf et ue.conf pour t'assurer que device_name = uhd"
echo " - Ordre de lancement : srsepc, puis srsenb (et srsue sur une autre machine/radio)"