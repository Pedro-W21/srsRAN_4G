#!/bin/bash
# usage : ./compare.sh NOM_POLITIQUE
DUR=60
pkill iperf3; sleep 1
iperf3 -s -p 5201 -D --logfile /tmp/srv1.log
iperf3 -s -p 5202 -D --logfile /tmp/srv2.log
sleep 1
sudo ip netns exec ue1 iperf3 -c 172.16.0.1 -p 5201 -R -u -b 50M -t $DUR > /tmp/${1}_ue1.txt &
sudo ip netns exec ue2 iperf3 -c 172.16.0.1 -p 5202 -R -u -b 50M -t $DUR > /tmp/${1}_ue2.txt &
wait
pkill iperf3
grep receiver /tmp/${1}_ue1.txt /tmp/${1}_ue2.txt
