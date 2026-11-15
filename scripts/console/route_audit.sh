#!/bin/bash
# S35: is the served cache-aware routing too aggressive? Quality (S18 instrument, 3.18 pack = served pack, k4v4 KV) and
# decode speed (smoke prompts, served decode flags) of the served routing and gentler candidates, in ONE server stop.
# The server and the dashboard are restarted on exit, whatever happens.
#   bash route_audit.sh            (~20 min of downtime)
cd "$(dirname "$0")"
D=$(pwd)
restart() { sudo systemctl reset-failed ps5lm-server ps5lm-dash 2>/dev/null; sudo systemctl start ps5lm-server ps5lm-dash; echo "server restarted $(date +%T)"; }
trap restart EXIT
P318="--pack experts_exl3.pack --kv k4v4"
TIER="--tier /dev/ps5lowmem"
declare -A C
C[exact]="--route-mtp 1"
C[served]="--route-mtp 1 --route-bias 1 --prefetch-max 4 --route-pf $TIER"
C[served_notier]="--route-mtp 1 --route-bias 1 --prefetch-max 4 --route-pf"
C[keep3]="--route-mtp 1 --route-bias 1 --prefetch-max 4 --route-pf --route-keep 3 $TIER"
C[b05p4]="--route-mtp 1 --route-bias 0.5 --prefetch-max 4 --route-pf $TIER"
C[b1p6]="--route-mtp 1 --route-bias 1 --prefetch-max 6 --route-pf $TIER"
ORDER="exact served served_notier keep3 b05p4 b1p6"
echo "== quality $(date +%T)"
for n in $ORDER; do bash $D/score.sh a35_$n $P318 ${C[$n]}; done
SERVE="--spec 3 --pld 4 --max-ctx 262144 --kv-base 8192"
PROSE="Write a detailed, well structured essay on the history of the Silk Road: its origins, the goods and ideas that moved along it, the empires that controlled it, and its influence on culture and trade today."
echo "== decode speed $(date +%T)"
for n in $ORDER; do
  [ $n = served_notier ] && continue
  GEN=300 bash $D/smoke.sh a35r_$n $P318 $SERVE ${C[$n]} | head -1 | cut -c1-150
  grep -a "displaced" ~/ps5lm/bench/sm_a35r_$n.err | tail -1 | grep -o "displaced.*"
  PROMPT="$PROSE" GEN=256 bash $D/smoke.sh a35p_$n $P318 $SERVE ${C[$n]} | head -1 | cut -c1-150
  grep -a "displaced" ~/ps5lm/bench/sm_a35p_$n.err | tail -1 | grep -o "displaced.*"
done
echo "== done $(date +%T)"
