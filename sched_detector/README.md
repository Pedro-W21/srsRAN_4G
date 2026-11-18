# sched-detector : reconnaître la politique de scheduling de srsENB depuis ses logs

Projet RCA, INSA Lyon. Outil de test pour srsRAN 4G (srsENB).

Le programme lit le log de l'eNB et dit quelle politique a distribué les ressources radio. Il reconnaît `time_rr`, `time_pf` (avec son coefficient d'équité α), Max C/I et Blind Equal Throughput. Toute autre politique est classée **« Scheduler personnalisé »**, avec le modèle connu qui s'en approche le plus. L'outil ne lit pas `enb.conf` : il regarde *comment* les ressources ont été attribuées et vérifie si un algorithme connu, rejoué sur le même log, aurait pris les mêmes décisions. Il donne aussi quatre indices d'équité de Jain (section 2.7).

Il fonctionne **en voie descendante (DL) comme en voie montante (UL)**. La politique ne se révèle que là où les UE se disputent les ressources, et c'est le sens du trafic de test qui le décide : `iperf3` lancé depuis l'EPC vers les UE charge le DL, lancé depuis les UE il charge l'UL. Par défaut, le détecteur analyse la voie qui a le plus de TTI disputés (`--direction auto`).

---

## 1. Deux façons de savoir quelle politique tourne

**La façon directe.** Au démarrage, `sched_carrier.cc` écrit dans le log MAC `Using time-domain RR scheduling policy for cc=0` (ou `PF`). Le détecteur affiche cette ligne en fin de rapport. Elle dit ce qui a été *configuré*, pas ce que le scheduler *fait* : une faute de frappe dans `enb.conf` donne silencieusement du PF, et une politique maison mal branchée peut annoncer un nom et se comporter autrement.

**La façon comportementale (ce que fait cet outil).** On déduit la politique des allocations. Le test devient : « srsENB annonce X, et les décisions observées sont bien celles de X ». Pour ta propre politique : « elle ne se confond avec aucune politique connue ».

---

## 2. Principe : rejouer chaque politique connue sur le log

### 2.1 Ce qu'on lit dans le log

Chaque ligne d'allocation donne l'UE servi (`rnti`), les ressources reçues, le TBS, les données en attente et le TTI d'émission :

| | DL : `SCHED: DL tx` | UL : `SCHED: UL tx` |
|---|---|---|
| ressources | RBG du champ `mask=0x…` | PRB de l'intervalle `prb=[a, b)` (fin exclue) |
| données en attente après l'allocation | `buffer=avant/après` (file RLC) | `bsr=a (b-c)` : dernier BSR `b` moins les octets déjà accordés `c` |
| TTI | `tti_tx_dl` | `tti_tx_ul` |

Exemple UL réel : `SCHED: UL tx rnti=0x4c, cc=0, pid=0, dci=(1,0), prb=[2, 47), n_rtx=0, cfi=1, tbs=2865, bsr=138208 (152053-13845), tti_tx_ul=2624` (45 PRB, 138 208 octets encore à émettre).

On reconstruit pour chaque TTI :

- les UE qui avaient des données en attente (un UE est « en attente » entre deux de ses allocations quand la première a laissé `buffer_after > 0` en DL, `bsr > 0` en UL) ;
- le **gagnant**, l'UE qui a reçu le plus de ressources. Avec l'allocation gloutonne de srsRAN (`try_dl_newtx_alloc_greedy` en DL, `find_contiguous_ul_prbs` en UL), le premier UE servi prend tout ce dont il a besoin : le gagnant est le premier de la file de priorité.

On ne garde que les **TTI disputés**, où au moins deux UE attendaient, car c'est seulement là que la politique choisit. Les retransmissions HARQ sont ignorées : RR et PF les servent toutes en premier.

### 2.2 Classification en ensemble ouvert

Un classifieur ordinaire range tout log dans la case la plus proche. Une politique inventée serait alors étiquetée PF, par défaut. Ici, chaque politique du catalogue est un **modèle qu'on rejoue** : à chaque TTI disputé, il prédit le gagnant à partir de ce que contient le log, et on compte ses réussites. Un modèle n'est retenu que s'il reproduit les décisions. Si aucun n'y parvient, la politique est déclarée personnalisée.

| Modèle | Ce qu'il prédit | Test d'acceptation |
|---|---|---|
| Round Robin (`sched_time_rr.cc`) | gagnant = fonction fixe de (`tti_tx mod N`, UE en attente) | précision ≥ 95 % en validation croisée, et le numéro de TTI apporte ≥ 90 % de l'information |
| Famille PF (`sched_time_pf.cc`) | gagnant = UE de plus grande priorité $r_i / \bar R_i^{\alpha}$ | précision ≥ 95 % sur au moins 80 TTI décisifs |

**Le modèle RR.** `sched_time_rr` commence son parcours à `tti_tx_dl % ue_db.size()` en DL (`tti_tx_ul % ue_db.size()` en UL, dans `sched_ul_newtxs`) et saute les UE sans données : le gagnant ne dépend que du résidu et de l'ensemble des UE en attente, jamais du passé. L'ordre interne de `ue_db` n'est pas connu, donc la correspondance est apprise sur une moitié du log et évaluée sur l'autre (validation croisée à deux plis). On compare ensuite à une prédiction qui n'utiliserait pas le numéro de TTI : ce gain normalisé (même construction que le kappa de Cohen) empêche qu'un UE qui gagne tout, comme sous Max C/I, passe pour un RR.

**La famille PF.** On rejoue la moyenne $\bar R_i$ exactement comme `save_dl_alloc` / `save_ul_alloc` : moyenne arithmétique sur 100 échantillons, puis exponentielle de pas $\beta = 0{,}01$, alimentée par les octets que le log attribue à chaque UE (0 quand il n'est pas servi). Le débit instantané $r_i$ est le dernier TBS par ressource (RBG ou PRB) observé pour l'UE. Le rejeu est fait pour α ∈ {0 ; 0,25 ; … ; 6 ; ∞}. Deux valeurs extrêmes correspondent à des politiques classiques :

- α = 0 : seule compte la qualité du canal → **Max C/I** ;
- α → ∞ : seul compte le plus petit débit moyen → **Blind Equal Throughput** (débit égal pour tous).

### 2.3 Pourquoi les « TTI décisifs »

À l'équilibre, un PF égalise les priorités des UE : c'est son but. Le gagnant se joue alors sur de petites fluctuations de CQI que le log ne montre pas, et même le bon modèle ne prédit qu'environ 65 % des gagnants. Ces TTI ne prouvent rien, ni pour ni contre. Le modèle est donc jugé sur les TTI où il préfère nettement un UE (meilleure priorité au moins 1,2 fois la deuxième). Ce sont surtout les TTI qui suivent une reprise de trafic. Sur les logs simulés, le bon modèle y obtient 100 %, et un mauvais α ou une politique étrangère au plus 87 %.

α est la valeur dont le rejeu réussit le mieux. Si plusieurs valeurs passent le test, on garde celle dont la précision globale est la plus haute, à condition qu'elle devance les autres d'au moins 5 points. Sinon α est déclaré non identifiable (cas des canaux identiques).

### 2.4 Partage de la bande

RR, PF, Max C/I et BET de srsRAN sont gloutons : le premier UE qui ne peut pas vider son buffer prend toutes les ressources libres. Voir **deux UE servis dans le même TTI et qui gardent tous deux des données** est donc impossible avec ces politiques. Au-delà de 10 % des TTI disputés, le log est classé personnalisé (politique fréquentielle, quotas par UE…) sans même rejouer les modèles.

**Cas de l'UL.** En SC-FDMA, un bloc PUSCH doit être contigu et de taille 2^a·3^b·5^c (précodage DFT). Avec 46 PRB libres (PUCCH sur les PRB 0-1 et 48-49), le premier UE reçoit 45 PRB `[2, 47)` et le PRB restant `[47, 48)` part légitimement à un autre UE, qui garde lui aussi des données. Ce reste n'est pas un partage : en UL, un TTI ne compte comme « bande partagée » que si le deuxième UE non satisfait a reçu au moins 25 % des PRB du premier (`UL_SHARED_BAND_MIN_RATIO` dans `features.py`).

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

### 2.7 Indices d'équité de Jain

Pour n UE de débits $x_i$ : $J = (\sum x_i)^2 / (n \sum x_i^2)$, qui vaut 1 si tous reçoivent la même chose et 1/n si un seul UE reçoit tout. Le rapport en donne quatre, dans les deux voies :

| Indice | Calculé sur | Ce qu'il mesure |
|---|---|---|
| débit sur tout le log | octets de tout l'essai | l'équité vue par l'utilisateur ; dépend du trafic offert |
| débit en pleine contention | octets des TTI où **tous** les UE attendaient (hors 300 TTI après chaque reprise) | l'équité produite par la politique elle-même, à demande égale |
| RBG / PRB en pleine contention | ressources de ces mêmes TTI | le partage des ressources, indépendamment du canal (un RR donne ≈ 1) |
| court terme | moyenne et minimum de J sur des fenêtres de 100 TTI de pleine contention (`--jain-window`) | l'échelle de temps à partir de laquelle la politique devient équitable |

Une fenêtre court terme n'enjambe jamais un TTI avec allocation hors pleine contention. Les TTI **sans aucune** allocation ne la coupent pas : en UL, les grants ne tombent pas à chaque TTI.

Seuls les UE vus dans la voie analysée sont comptés. Le log donne le TBS **alloué**, pas les ACK : il s'agit d'un débit alloué, pas d'un goodput.

## 3. Installation

```bash
cd sched_detector
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Le détecteur n'a aucune dépendance à l'exécution (bibliothèque standard uniquement). La broker multi-UE (section 4.2) a besoin de `pyzmq` et `numpy`, installés par l'extra `dev` ou seuls avec `pip install -e ".[testbed]"`. Python 3.10 ou plus.

Vérifier que tout fonctionne :

```bash
pytest            # 138 tests, ~1 min
ruff check . && mypy src
```

Essai sans radio, sur des logs synthétiques :

```bash
sched-simulate --policy rr --out demo_rr.log
sched-simulate --policy pf --alpha 2 --out demo_pf.log
sched-simulate --policy lqf --out demo_lqf.log      # politique hors catalogue
sched-detect demo_lqf.log

sched-simulate --direction ul --policy rr --out demo_ul_rr.log   # voie montante
sched-detect demo_ul_rr.log
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
mac_level = info          ; indispensable : les lignes SCHED: DL tx / UL tx sont au niveau info

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

La colonne `octets/RBG` (ou `octets/PRB` en UL) du rapport doit montrer un écart d'au moins 25 % entre les UE. En UL, l'émulateur de canal de srsUE n'agit que sur la réception de l'UE : pour dégrader la voie montante, utiliser l'émulateur de canal de l'eNB (`--channel.ul.enable=true --channel.ul.awgn.enable=true --channel.ul.awgn.snr=…`) ou limiter le MCS (`pusch_max_mcs`), sachant que l'émulateur de l'eNB s'applique à tous les UE.

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

**Pour un essai en voie montante**, on inverse le sens : les serveurs côté EPC, les clients dans les namespaces des UE. L'option `-R` d'`iperf3` donne le même résultat sans changer les serveurs.

```bash
iperf3 -s -p 5201 &
iperf3 -s -p 5202 &

# ue2 : émet en continu pendant 60 s (environ 15 à 20 Mbit/s en voie montante dans nos essais)
sudo ip netns exec ue2 iperf3 -c 172.16.0.1 -p 5202 -u -b 30M -t 60 &

# ue1 : 1 s d'émission, 0,5 s de pause, répété
for i in $(seq 40); do sudo ip netns exec ue1 iperf3 -c 172.16.0.1 -p 5201 -u -b 30M -t 1; sleep 0.5; done
```

`172.16.0.1` est l'adresse de l'EPC (`srs_spgw_sgi`) par défaut. Le BSR plafonne vers 150 ko : un champ `bsr` qui reste autour de cette valeur montre que l'UE est bien saturé.

### 4.5 Analyse

```bash
sched-detect /tmp/enb.log --expect rr       # essai avec policy = time_rr
sched-detect /tmp/enb.log --expect pf       # essai avec policy = time_pf
sched-detect /tmp/enb.log --expect custom   # essai avec ta politique
sched-detect /tmp/enb.log --direction ul    # imposer la voie montante
```

La deuxième ligne du rapport indique la voie analysée et, en mode automatique, le nombre de TTI disputés trouvés dans chaque voie. Un rapport UL affiche `octets/PRB`, `PRB en pleine contention` et `PRB (contention)` à la place des colonnes RBG.

`--expect` accepte `rr`, `pf`, `maxci`, `bet`, `custom`. Codes de sortie :

| Code | Sens |
|---|---|
| 0 | verdict obtenu et conforme à `--expect` |
| 1 | fichier absent, ou sans ligne `SCHED: DL` ni `SCHED: UL` pour cette porteuse (vérifier `mac_level = info`), ou voie imposée absente du log |
| 2 | politique détectée différente de `--expect` |
| 3 | indéterminé (cellule pas assez chargée, un seul UE) |

Options : `--direction dl|ul|auto` pour la voie analysée, `--ignore-rnti 0x4b` pour écarter un UE sans trafic (répétable), `--cc 1` pour une autre porteuse, `--nof-ues 3` si un UE connecté n'a jamais reçu de données (le modulo du RR porte sur **tous** les UE connectés, `ue_db.size()`), `--verbose` pour le debug.

Exemple de sortie sur une politique hors catalogue (plus gros buffer d'abord, en UL) :

```
Politique détectée : Scheduler personnalisé  (confiance élevée)
Voie analysée : montante (UL), choisie automatiquement (TTI disputés : 19999 en UL)
Modèle connu le plus proche : aucun (aucun ne fait mieux que désigner toujours le favori)

En bref :
  - Aucune politique connue ne reproduit ses décisions : politique personnalisée.
  - Servi en premier quand plusieurs UE attendent : 0x46 67%, 0x47 33% (canaux différents : 50.0 à
    100.0 octets/PRB).
  - Round Robin : non. A raison 67% du temps, contre 67% en désignant toujours 0x46 : gain +0 pt.
  - Proportional Fair / BET : non (meilleur modèle : Proportional Fair (alpha=0.25)). A raison 13%
    du temps, contre 62% en désignant toujours 0x46 : gain -49 pts.
  - Max C/I : non. A raison 66% du temps, contre 66% en désignant toujours 0x46 : gain +0 pt.

...
Comparaison avec les politiques connues (sur les décisions où chaque modèle a un avis
net) :
  a raison : part des décisions du scheduler que le modèle reproduit ;
  favori   : part qu'on reproduirait en désignant toujours l'UE qui gagne le plus (0x46) ;
  gain     : différence entre les deux. Un modèle ne ressemble au scheduler que s'il a
             raison au moins 95% du temps avec un gain nettement positif.
Politique connue                 décisions  a raison  favori     gain  verdict
Round Robin                          19999       67%     67%    +0 pt  rejeté : pas mieux que le favori
Max C/I                              19500       66%     66%    +0 pt  rejeté : pas mieux que le favori
Proportional Fair (alpha=0.25)       16989       13%     62%  -49 pts  rejeté : pas mieux que le favori
...
```

**Lire les chiffres sans se faire piéger.** Un modèle qui désigne toujours l'UE favori a raison aussi souvent que ce favori gagne : 67 % ici, 90 % si un UE gagne 9 TTI sur 10. Ce pourcentage seul ne prouve donc rien. Chaque score est affiché à côté de sa référence :

| Colonne | Signification |
|---|---|
| décisions | décisions où le modèle a un avis net (tous les TTI disputés pour le RR, les TTI décisifs pour la famille PF) |
| a raison | part de ces décisions où le modèle désigne le même UE que le scheduler |
| favori | part qu'on obtiendrait sur ces mêmes décisions en désignant toujours l'UE qui gagne le plus souvent |
| gain | `a raison − favori`, en points. **+0 pt = le modèle n'explique rien**, quel que soit son pourcentage |
| verdict | validé, rejeté (avec la raison) ou non testable (moins de 80 décisions nettes) |

Un modèle ne ressemble au scheduler que s'il a raison au moins 95 % du temps avec un gain nettement positif. Un vrai `time_rr` entre deux UE donne par exemple « a raison 100 %, favori 50 %, gain +50 pts ». La ligne « Modèle connu le plus proche » n'est remplie que si un modèle fait nettement mieux que le favori ; sinon elle indique « aucun ».

Les indices de Jain sont aussi traduits en nombre d'UE servis à égalité (`J × n`) : avec 3 UE, J = 0,374 donne « ≈ 1,1 UE sur 3 servis à égalité ».

**UE connecté sans trafic.** S'il est compté, la « pleine contention » (tous les UE attendent) n'arrive jamais et les indices de Jain correspondants restent vides. Le rapport le signale (« Astuce : … ») et propose `--ignore-rnti 0x4b` : l'UE est retiré des métriques mais reste compté dans le N du modulo RR, comme dans `ue_db.size()`.

Le tableau de comparaison est la partie la plus utile pour ton rapport : il montre en chiffres que ta politique ne se réduit à aucune politique connue.

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
  log_parser.py   lecture des lignes SCHED: DL et SCHED: UL, dépliage du TTI (rebouclage à 10240)
  features.py     chronologie, TTI disputés, partage de bande, parts, débits
  models.py       rejeu du RR (validation croisée) et de la famille PF (grille de α)
  classifier.py   décision en ensemble ouvert, seuils
  fairness.py     indices d'équité de Jain (4 indices)
  interpretation.py  section « En bref » et chiffres comparés au favori
  detector.py     enchaînement : allocations → chronologie → modèles → verdict, choix de la voie
  cli.py          commande sched-detect
  simulator.py    logs synthétiques DL ou UL, 7 politiques (commande sched-simulate)
  testbed/
    zmq_broker.py   broker ZMQ multi-UE pour srsRAN (commande srsran-zmq-broker)
tests/
  test_log_parser.py   format des lignes, filtrage, rebouclage
  test_classifier.py   bout en bout : catalogue reconnu, hors catalogue → personnalisé
  test_cli.py          rapport et codes de sortie
  test_fairness.py     indices de Jain
  test_uplink.py       voie montante : lignes UL réelles, verdicts, Jain, choix de la voie
  test_interpretation.py  chiffres du rapport face au favori, --ignore-rnti
  data/real_ul_excerpt.log   40 lignes SCHED: UL d'un vrai log srsENB
  test_zmq_broker.py   alignement de la broker avec de faux nœuds srsRAN
```

Le simulateur reproduit la logique de `sched_time_rr.cc` et `sched_time_pf.cc` (ordre de service, allocation gloutonne, moyenne de `save_dl_alloc`). En UL, il alloue des blocs de PRB contigus de taille valide pour la DFT, réserve le PUCCH aux bords et le PRACH un TTI sur dix, et écrit le champ `bsr` comme srsENB. Il ne modélise ni le canal radio ni le HARQ. Il sert à fabriquer des logs dont on connaît la politique, pour vérifier que le détecteur ne se trompe pas.

---

## 7. Limites

- **« Toutes les politiques » veut dire : celles du catalogue sont nommées, les autres sont détectées comme différentes.** Le détecteur ne peut pas deviner la règle d'une politique inconnue. Il dit qu'elle n'est aucune des politiques connues et laquelle s'en approche le plus.
- **Une politique inconnue peut coïncider avec une connue dans une situation donnée.** Par exemple, « plus gros buffer d'abord » avec des files RLC saturées à la même taille revient à une priorité fixe, et si l'UE prioritaire a aussi le meilleur canal, le log est identique à celui d'un Max C/I. Le détecteur juge le comportement observé, pas le code. Varier les conditions (pauses, canaux, charges différentes) réduit ce risque.
- **Cellule saturée obligatoire.** Sans contention, aucune politique n'a de choix à faire : en dessous de 500 TTI disputés, le verdict est « indéterminé ».
- **Canaux identiques.** PF, BET et un α donné ne se distinguent plus (confiance faible, α non identifiable), et un Max C/I y ressemble à une politique inconnue.
- **PF parfaitement régulier sans pause de trafic.** Il alterne comme un RR : verdict RR en confiance faible, jamais élevée (cas couvert par un test).
- **Le rejeu PF suppose que le log reflète les allocations.** Avec beaucoup d'UE, la saturation du PDCCH (`Couldn't find space in PDCCH`) ou de nombreuses retransmissions écartent srsRAN du modèle, et la précision baisse. Analyser une fenêtre où les UE restent connectés : une connexion change `ue_db.size()` et l'ordre du RR.
- **Une voie à la fois.** Le DL et l'UL sont analysés séparément (une politique de srsRAN décide les deux voies indépendamment, avec deux files de priorité). En mode `auto`, une seule voie est analysée ; relancer avec `--direction` pour l'autre.
- **UL : BSR en paliers.** Le BSR est quantifié par l'UE (TS 36.321, table 6.1.3.1-1) et n'est mis à jour qu'à chaque rapport : la fin d'une période d'attente est datée avec moins de précision qu'en DL. Les UE saturés (BSR au plafond) ne sont pas concernés.
- **Validé sur des logs simulés.** Les seuils (95 %, 80 TTI décisifs, ratio 1,2, et 25 % pour la bande partagée en UL) sont calibrés sur le simulateur. Il faut les contrôler sur tes premiers logs réels, en lançant d'abord `time_rr` et `time_pf`, dont tu connais la réponse.
