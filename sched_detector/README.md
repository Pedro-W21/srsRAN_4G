# sched-detector : reconnaître la politique de scheduling de srsENB depuis ses logs

Projet RCA, INSA Lyon. Outil de test pour srsRAN 4G (srsENB).

Le programme lit le log de l'eNB et dit quelle politique a distribué les ressources radio. Il reconnaît `time_rr`, `time_pf` (avec son coefficient d'équité α), Max C/I et Blind Equal Throughput. Toute autre politique est classée **« Scheduler personnalisé »**, avec le modèle connu qui s'en approche le plus. L'outil ne lit pas `enb.conf` : il regarde *comment* les RBG ont été attribués et vérifie si un algorithme connu, rejoué sur le même log, aurait pris les mêmes décisions.

---

## 1. Deux façons de savoir quelle politique tourne

**La façon directe.** Au démarrage, `sched_carrier.cc` écrit dans le log MAC `Using time-domain RR scheduling policy for cc=0` (ou `PF`). Le détecteur affiche cette ligne en fin de rapport. Elle dit ce qui a été *configuré*, pas ce que le scheduler *fait* : une faute de frappe dans `enb.conf` donne silencieusement du PF, et une politique maison mal branchée peut annoncer un nom et se comporter autrement.

**La façon comportementale (ce que fait cet outil).** On déduit la politique des allocations. Le test devient : « srsENB annonce X, et les décisions observées sont bien celles de X ». Pour ta propre politique : « elle ne se confond avec aucune politique connue ».

---

## 2. Principe : rejouer chaque politique connue sur le log

### 2.1 Ce qu'on lit dans le log

Chaque ligne `SCHED: DL tx` donne l'UE servi (`rnti`), les RBG reçus (`mask`), le TBS, le buffer avant/après et le TTI d'émission. On reconstruit pour chaque TTI :

- les UE qui avaient des données en attente (un UE est « en attente » entre deux de ses allocations quand la première a laissé `buffer_after > 0`) ;
- le **gagnant**, l'UE qui a reçu le plus de RBG. Avec l'allocation gloutonne de srsRAN (`try_dl_newtx_alloc_greedy`), le premier UE servi prend tous les RBG dont il a besoin : le gagnant est le premier de la file de priorité.

On ne garde que les **TTI disputés**, où au moins deux UE attendaient, car c'est seulement là que la politique choisit. Les retransmissions HARQ sont ignorées : RR et PF les servent toutes en premier.

### 2.2 Classification en ensemble ouvert

Un classifieur ordinaire range tout log dans la case la plus proche. Une politique inventée serait alors étiquetée PF, par défaut. Ici, chaque politique du catalogue est un **modèle qu'on rejoue** : à chaque TTI disputé, il prédit le gagnant à partir de ce que contient le log, et on compte ses réussites. Un modèle n'est retenu que s'il reproduit les décisions. Si aucun n'y parvient, la politique est déclarée personnalisée.

| Modèle | Ce qu'il prédit | Test d'acceptation |
|---|---|---|
| Round Robin (`sched_time_rr.cc`) | gagnant = fonction fixe de (`tti_tx_dl mod N`, UE en attente) | précision ≥ 95 % en validation croisée, et le numéro de TTI apporte ≥ 90 % de l'information |
| Famille PF (`sched_time_pf.cc`) | gagnant = UE de plus grande priorité $r_i / \bar R_i^{\alpha}$ | précision ≥ 95 % sur au moins 80 TTI décisifs |

**Le modèle RR.** `sched_time_rr` commence son parcours à `tti_tx_dl % ue_db.size()` et saute les UE sans données : le gagnant ne dépend que du résidu et de l'ensemble des UE en attente, jamais du passé. L'ordre interne de `ue_db` n'est pas connu, donc la correspondance est apprise sur une moitié du log et évaluée sur l'autre (validation croisée à deux plis). On compare ensuite à une prédiction qui n'utiliserait pas le numéro de TTI : ce gain normalisé (même construction que le kappa de Cohen) empêche qu'un UE qui gagne tout, comme sous Max C/I, passe pour un RR.

**La famille PF.** On rejoue la moyenne $\bar R_i$ exactement comme `save_dl_alloc` : moyenne arithmétique sur 100 échantillons, puis exponentielle de pas $\beta = 0{,}01$, alimentée par les octets que le log attribue à chaque UE (0 quand il n'est pas servi). Le débit instantané $r_i$ est le dernier TBS par RBG observé pour l'UE. Le rejeu est fait pour α ∈ {0 ; 0,25 ; … ; 6 ; ∞}. Deux valeurs extrêmes correspondent à des politiques classiques :

- α = 0 : seule compte la qualité du canal → **Max C/I** ;
- α → ∞ : seul compte le plus petit débit moyen → **Blind Equal Throughput** (débit égal pour tous).

### 2.3 Pourquoi les « TTI décisifs »

À l'équilibre, un PF égalise les priorités des UE : c'est son but. Le gagnant se joue alors sur de petites fluctuations de CQI que le log ne montre pas, et même le bon modèle ne prédit qu'environ 65 % des gagnants. Ces TTI ne prouvent rien, ni pour ni contre. Le modèle est donc jugé sur les TTI où il préfère nettement un UE (meilleure priorité au moins 1,2 fois la deuxième). Ce sont surtout les TTI qui suivent une reprise de trafic. Sur les logs simulés, le bon modèle y obtient 100 %, et un mauvais α ou une politique étrangère au plus 87 %.

α est la valeur dont le rejeu réussit le mieux. Si plusieurs valeurs passent le test, on garde celle dont la précision globale est la plus haute, à condition qu'elle devance les autres d'au moins 5 points. Sinon α est déclaré non identifiable (cas des canaux identiques).

### 2.4 Partage de la bande

RR, PF, Max C/I et BET de srsRAN sont gloutons : le premier UE qui ne peut pas vider son buffer prend tous les RBG libres. Voir **deux UE servis dans le même TTI et qui gardent tous deux des données** est donc impossible avec ces politiques. Au-delà de 10 % des TTI disputés, le log est classé personnalisé (politique fréquentielle, quotas par UE…) sans même rejouer les modèles.

### 2.5 Le cas RR contre PF

Entre deux UE identiques, un PF alterne A, B, A, B… exactement comme un RR, et les deux modèles reproduisent le log. On les départage sur les seuls TTI où leurs prédictions diffèrent. Il s'agit typiquement des TTI qui suivent une pause de trafic : sous PF, l'UE qui revient a une moyenne basse et gagne plusieurs TTI d'affilée, alors que sous RR il reprend sa place dans la rotation. Sans au moins 30 TTI de désaccord, le verdict RR sort avec une confiance « faible ».

### 2.6 Arbre de décision

```
moins de 2 UE, ou moins de 500 TTI disputés           → indéterminé
bande partagée dans plus de 10 % des TTI disputés     → Scheduler personnalisé
RR validé et aucun PF ne reproduit le log             → time_rr
RR validé et un PF aussi                              → arbitrage sur les désaccords
un modèle de la famille PF validé :
    α = 0                                             → Max C/I
    α = ∞                                             → Blind Equal Throughput
    sinon                                             → time_pf (α estimé)
aucun modèle validé                                   → Scheduler personnalisé
```

Le verdict « personnalisé » passe en confiance faible quand les canaux des UE sont presque identiques (écart < 25 %). Dans ce cas, un Max C/I choisit au gré de fluctuations de CQI invisibles et ressemble à une politique inconnue.

Tous les seuils sont des constantes nommées en tête de `classifier.py` et `models.py`.

---

## 3. Installation

```bash
cd sched_detector
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Le détecteur n'a aucune dépendance à l'exécution (bibliothèque standard uniquement). La broker multi-UE (section 4.2) a besoin de `pyzmq` et `numpy`, installés par l'extra `dev` ou seuls avec `pip install -e ".[testbed]"`. Python 3.10 ou plus.

Vérifier que tout fonctionne :

```bash
pytest            # 64 tests, ~45 s
ruff check . && mypy src
```

Essai sans radio, sur des logs synthétiques :

```bash
sched-simulate --policy rr --out demo_rr.log
sched-simulate --policy pf --alpha 2 --out demo_pf.log
sched-simulate --policy lqf --out demo_lqf.log      # politique hors catalogue
sched-detect demo_lqf.log
```

Politiques simulables : `rr`, `pf`, `maxci`, `bet`, et trois politiques absentes de srsRAN qui servent à vérifier le verdict « personnalisé » : `lqf` (plus gros buffer d'abord), `random` (ordre tiré au sort), `freq_share` (bande partagée en parts égales).

---

## 4. Protocole de mesure sur srsRAN

### 4.1 Configuration de l'eNB

Dans `enb.conf` :

```ini
[log]
filename  = /tmp/enb.log
all_level = warning
mac_level = info          ; indispensable : les lignes SCHED: DL tx sont au niveau info

[scheduler]
policy      = time_rr     ; puis time_pf, puis ta politique
policy_args = 2
```

Ne pas fixer `pdsch_mcs` : les modèles PF et Max C/I ont besoin que le MCS suive le canal de chaque UE.

### 4.2 Au moins deux UE : la broker ZMQ

Avec un seul UE, toutes les politiques donnent le même résultat. Or la radio virtuelle ZMQ de srsRAN relie un émetteur à un seul récepteur. La commande `srsran-zmq-broker` (module `sched_detector.testbed.zmq_broker`) se place entre l'eNB et les UE : elle distribue la voie descendante à chaque UE et additionne leurs voies montantes, sans GNU Radio.

**Fonctionnement.** Chaque émetteur srsRAN ouvre un socket REP et attend une requête d'un octet ; chaque récepteur interroge avec un socket REQ (`lib/src/phy/rf/rf_zmq_imp_tx.c` et `_rx.c`). Un émetteur srsRAN ne compte un échantillon qu'une fois envoyé, et `rf_zmq_tx_align` comble l'émission avec des zéros pour qu'elle suive la réception. L'échantillon montant k d'un UE correspond donc à l'instant eNB `offset_UE + k`, où `offset_UE` est le premier échantillon descendant qu'il a reçu. La broker additionne chaque UE à ce décalage. Deux règles gardent tout le monde au même pas :

- la voie montante servie à l'eNB ne dépasse jamais ce que l'eNB a déjà émis ;
- un UE connecté est toujours attendu, même pendant l'initialisation de sa PHY. Chaque nœud srsRAN dort la durée de ce qu'il lit (`usleep` dans `rf_zmq_recv_with_time_multi`) et ne va jamais plus vite que le temps réel : un UE laissé en arrière ne rattraperait jamais son retard. L'eNB est simplement suspendu, comme avec une liaison ZMQ directe.

Un UE qui a émis puis se tait plus de 1,5 s est écarté, et un UE connecté qui n'émet pas au bout de 60 s aussi. Il faut alors le relancer.

**Validation.** Les tests (`tests/test_zmq_broker.py`) utilisent de faux eNB et UE au protocole srsRAN : l'eNB émet une rampe et chaque UE renvoie ce qu'il reçoit. L'eNB doit recevoir exactement k × m à l'instant m (k = nombre d'UE), sans un seul échantillon décalé, y compris quand un UE arrive en retard, s'initialise lentement ou meurt. Avec les vrais srsENB et srsUE (commit `bef8680`), deux UE terminent leur accès aléatoire en même temps à travers la broker, ce qui suppose que l'eNB a décodé le Msg3 montant de chacun.

**Ports** (convention de la doc srsRAN pour une broker) :

| Nœud | `tx_port` | `rx_port` |
|---|---|---|
| eNB | `tcp://*:2101` | `tcp://localhost:2100` |
| UE1 | `tcp://*:2001` | `tcp://localhost:2000` |
| UE2 | `tcp://*:2011` | `tcp://localhost:2010` |
| UE n | `tcp://*:2001+10(n-1)` | `tcp://localhost:2000+10(n-1)` |

**Lancement**, chaque commande dans son terminal, depuis `srsRAN_4G/build`, dans cet ordre :

```bash
# 1. EPC
sudo ip netns add ue1; sudo ip netns add ue2
sudo ./srsepc/src/srsepc $HOME/.config/srsran/epc.conf \
  --hss.db_file=$HOME/.config/srsran/user_db.csv

# 2. Broker (dans le venv du détecteur)
srsran-zmq-broker --ues 2

# 3. eNB : pas de fail_on_disconnect, l'eNB doit pouvoir attendre un UE
./srsenb/src/srsenb --rf.device_name=zmq \
  --rf.device_args="tx_port=tcp://*:2101,rx_port=tcp://localhost:2100,id=enb,base_srate=23.04e6" \
  --scheduler.policy=time_rr --log.mac_level=info --log.filename=/tmp/enb_rr.log

# 4. UE1 : abonné milenage par défaut de ue.conf (IMSI 001010123456780), canal parfait
sudo ./srsue/src/srsue $HOME/.config/srsran/ue.conf --rf.device_name=zmq \
  --rf.device_args="tx_port=tcp://*:2001,rx_port=tcp://localhost:2000,id=ue1,base_srate=23.04e6" \
  --gw.netns=ue1 --log.filename=/tmp/ue1.log

# 5. UE2 : abonné xor de user_db.csv (IMSI 001010123456789), canal dégradé
sudo ./srsue/src/srsue $HOME/.config/srsran/ue.conf --rf.device_name=zmq \
  --rf.device_args="tx_port=tcp://*:2011,rx_port=tcp://localhost:2010,id=ue2,base_srate=23.04e6" \
  --gw.netns=ue2 --log.filename=/tmp/ue2.log \
  --usim.algo=xor --usim.imsi=001010123456789 --usim.imei=353490069873318 \
  --channel.dl.enable=true --channel.dl.awgn.enable=true --channel.dl.awgn.snr=12
```

Attendre `Network attach successful` sur les deux UE avant d'envoyer du trafic. Le premier UE attaché reçoit en général `172.16.0.2` et le second `172.16.0.3` ; vérifier avec `sudo ip netns exec ue2 ip -4 addr show tun_srsue`. Dans la suite, `IP_UE1` et `IP_UE2` désignent ces adresses.

La broker affiche toutes les 5 s son débit (`DL 23.04 Méch/s` = temps réel à `base_srate=23.04e6`) et l'état de chaque UE. Un débit plus faible veut dire que la machine ne suit pas le temps réel : la cellule ralentit d'autant, mais reste cohérente. Sur une machine peu puissante, `base_srate=11.52e6` (à mettre partout, broker comprise avec `--base-srate 11.52e6`) divise la charge par deux. Cette valeur convient à la cellule de 50 PRB par défaut et a été validée avec les vrais srsENB et srsUE.

Arrêt : les UE, puis l'eNB, puis la broker, puis l'EPC. Relancer la broker à chaque nouvel essai : l'eNB doit lui-même être relancé après le détachement d'un UE.

### 4.3 Des canaux différents

En ZMQ le canal est parfait : les deux UE remontent CQI 15. Il faut alors dégrader `ue2` pour séparer PF, Max C/I et BET, estimer α, et donner une confiance élevée au verdict « personnalisé » :

- l'émulateur de canal de srsUE, options `--channel.dl.enable=true --channel.dl.awgn.enable=true --channel.dl.awgn.snr=12` pour `ue2` seulement (déjà dans la commande de la section 4.2). Si `ue2` n'arrive pas à s'attacher, remonter le SNR à 18 ou 20 dB.

La colonne `octets/RBG` du rapport doit montrer un écart d'au moins 25 % entre les UE.

### 4.4 Le trafic : un UE saturé en continu, l'autre intermittent

L'UE au **meilleur** canal (`ue1`) doit être l'UE intermittent. Sous Max C/I, l'autre UE n'est servi que pendant les pauses de `ue1`, et sans ces pauses il n'apparaîtrait jamais dans le log.

```bash
sudo ip netns exec ue1 iperf3 -s -p 5201 &
sudo ip netns exec ue2 iperf3 -s -p 5202 &

# ue2 : saturé en continu pendant 60 s
iperf3 -c IP_UE2 -p 5202 -u -b 60M -t 60 &

# ue1 : 1 s de trafic, 0,5 s de pause, répété
for i in $(seq 40); do iperf3 -c IP_UE1 -p 5201 -u -b 60M -t 1; sleep 0.5; done
```

60 Mbit/s dépasse la capacité de la cellule (environ 35 Mbit/s à 50 PRB en SISO), ce qui garde les files RLC pleines. Chaque pause de `ue1` produit une reprise de trafic : ce sont elles qui fournissent les TTI décisifs.

### 4.5 Analyse

```bash
sched-detect /tmp/enb.log --expect rr       # essai avec policy = time_rr
sched-detect /tmp/enb.log --expect pf       # essai avec policy = time_pf
sched-detect /tmp/enb.log --expect custom   # essai avec ta politique
```

`--expect` accepte `rr`, `pf`, `maxci`, `bet`, `custom`. Codes de sortie :

| Code | Sens |
|---|---|
| 0 | verdict obtenu et conforme à `--expect` |
| 1 | fichier absent ou sans ligne `SCHED: DL` (vérifier `mac_level = info`) |
| 2 | politique détectée différente de `--expect` |
| 3 | indéterminé (cellule pas assez chargée, un seul UE) |

Options : `--cc 1` pour une autre porteuse, `--nof-ues 3` si un UE connecté n'a jamais reçu de données (le modulo du RR porte sur **tous** les UE connectés, `ue_db.size()`), `--verbose` pour le debug.

Exemple de sortie sur une politique hors catalogue (plus gros buffer d'abord) :

```
Politique détectée : Scheduler personnalisé  (confiance élevée)
Modèle connu le plus proche : Round Robin (67% des gagnants prédits)

Justification :
  - Le numéro de TTI n'explique pas le gagnant (gain normalisé 0.00, seuil 0.9) : ce n'est pas un RR.
  - Aucun modèle PF / Max C/I / BET ne reproduit les décisions nettes : au mieux 66% sur 19000 TTI décisifs (Max C/I), seuil 95%.

Modèle rejoué                        précision  décisifs  TTI décisifs
Round Robin                                67%       67%         19498
Max C/I                                    66%       66%         19000
Proportional Fair (alpha=0.25)             15%       11%         17236
...
Blind Equal Throughput                     10%        9%         18536
```

Le tableau des modèles rejoués est la partie la plus utile pour ton rapport : il montre en chiffres que ta politique ne se réduit à aucune politique connue.

---

## 5. Ajouter une politique au catalogue

Si ta politique devient une référence que le détecteur doit nommer, et non plus seulement classer « personnalisée » :

1. **Écrire son modèle** dans `models.py` : une fonction `fit_<nom>(timeline) -> ModelFit` qui, pour chaque TTI disputé, prédit le gagnant à partir de l'état visible dans le log (buffers, débits, historique des allocations) ;
2. **L'ajouter au simulateur** (`SimulatedPolicy` et `_service_order` dans `simulator.py`) pour générer des logs dont on connaît la politique ;
3. **Ajouter un membre à `Policy`** et une règle dans `classify()`, puis un test qui vérifie qu'elle est reconnue **et** que les politiques existantes ne sont pas confondues avec elle.

Si ta politique utilise des grandeurs absentes du log (CQI de chaque UE, priorités internes), le plus fiable est de les écrire toi-même dans le log depuis ta classe (`logger.info`) et de les lire dans `log_parser.py`.

---

## 6. Organisation du code

```
src/sched_detector/
  log_parser.py   lecture des lignes SCHED: DL, dépliage du TTI (rebouclage à 10240)
  features.py     chronologie, TTI disputés, partage de bande, parts, débits
  models.py       rejeu du RR (validation croisée) et de la famille PF (grille de α)
  classifier.py   décision en ensemble ouvert, seuils
  detector.py     enchaînement : allocations → chronologie → modèles → verdict
  cli.py          commande sched-detect
  simulator.py    logs synthétiques, 7 politiques (commande sched-simulate)
  testbed/
    zmq_broker.py   broker ZMQ multi-UE pour srsRAN (commande srsran-zmq-broker)
tests/
  test_log_parser.py   format des lignes, filtrage, rebouclage
  test_classifier.py   bout en bout : catalogue reconnu, hors catalogue → personnalisé
  test_cli.py          rapport et codes de sortie
  test_zmq_broker.py   alignement de la broker avec de faux nœuds srsRAN
```

Le simulateur reproduit la logique de `sched_time_rr.cc` et `sched_time_pf.cc` (ordre de service, allocation gloutonne, moyenne de `save_dl_alloc`). Il ne modélise ni le canal radio ni le HARQ. Il sert à fabriquer des logs dont on connaît la politique, pour vérifier que le détecteur ne se trompe pas.

---

## 7. Limites

- **« Toutes les politiques » veut dire : celles du catalogue sont nommées, les autres sont détectées comme différentes.** Le détecteur ne peut pas deviner la règle d'une politique inconnue. Il dit qu'elle n'est aucune des politiques connues et laquelle s'en approche le plus.
- **Une politique inconnue peut coïncider avec une connue dans une situation donnée.** Par exemple, « plus gros buffer d'abord » avec des files RLC saturées à la même taille revient à une priorité fixe, et si l'UE prioritaire a aussi le meilleur canal, le log est identique à celui d'un Max C/I. Le détecteur juge le comportement observé, pas le code. Varier les conditions (pauses, canaux, charges différentes) réduit ce risque.
- **Cellule saturée obligatoire.** Sans contention, aucune politique n'a de choix à faire : en dessous de 500 TTI disputés, le verdict est « indéterminé ».
- **Canaux identiques.** PF, BET et un α donné ne se distinguent plus (confiance faible, α non identifiable), et un Max C/I y ressemble à une politique inconnue.
- **PF parfaitement régulier sans pause de trafic.** Il alterne comme un RR : verdict RR en confiance faible, jamais élevée (cas couvert par un test).
- **Le rejeu PF suppose que le log reflète les allocations.** Avec beaucoup d'UE, la saturation du PDCCH (`Couldn't find space in PDCCH`) ou de nombreuses retransmissions écartent srsRAN du modèle, et la précision baisse. Analyser une fenêtre où les UE restent connectés : une connexion change `ue_db.size()` et l'ordre du RR.
- **Voie descendante seulement.** La voie montante suit la même logique (même modulo dans `sched_ul_newtxs`), mais ses lignes `SCHED: UL tx` donnent un intervalle de PRB et non un masque. L'extension est possible sur le même modèle.
- **Validé sur des logs simulés.** Les seuils (95 %, 80 TTI décisifs, ratio 1,2) sont calibrés sur le simulateur. Il faut les contrôler sur tes premiers logs réels, en lançant d'abord `time_rr` et `time_pf`, dont tu connais la réponse.
