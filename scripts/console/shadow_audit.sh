#!/bin/bash
# S35 option 3: LFU shadow uses for displaced experts (--route-shadow F, --route-shadow-fetch T) against the served
# routing. Same instrument and prompts as route_audit.sh, one server stop, restart on exit whatever happens.
#   bash shadow_audit.sh            (~20 min of downtime)
cd "$(dirname "$0")"
D=$(pwd)
restart() { sudo systemctl reset-failed ps5lm-server ps5lm-dash 2>/dev/null; sudo systemctl start ps5lm-server ps5lm-dash; echo "server restarted $(date +%T)"; }
trap restart EXIT
echo "== exactness of the new binary (flags off) $(date +%T)"
bash $D/smoke.sh s35_exact | head -1 | cut -c1-120
P318="--pack experts_exl3.pack --kv k4v4"
SERVED="--route-mtp 1 --route-bias 1 --prefetch-max 4 --route-pf --tier /dev/ps5lowmem"
declare -A C
C[served]="$SERVED"
C[sh05]="$SERVED --route-shadow 0.5"
C[sh1]="$SERVED --route-shadow 1"
C[sh1f2]="$SERVED --route-shadow 1 --route-shadow-fetch 2"
C[sh1f1]="$SERVED --route-shadow 1 --route-shadow-fetch 1"
ORDER="${ORDER:-served sh05 sh1 sh1f2 sh1f1}"
echo "== quality $(date +%T)"
for n in $ORDER; do
  bash $D/score.sh s35_$n $P318 ${C[$n]}
  grep -a "shadow uses\|background reads" ~/ps5lm/qual/sc_s35_$n.err | tail -2
done
SERVE="--spec 3 --pld 4 --max-ctx 262144 --kv-base 8192"
PROSE="Write a detailed, well structured essay on the history of the Silk Road: its origins, the goods and ideas that moved along it, the empires that controlled it, and its influence on culture and trade today."
echo "== decode speed $(date +%T)"
for n in $ORDER; do
  GEN=300 bash $D/smoke.sh s35r_$n $P318 $SERVE ${C[$n]} | head -1 | cut -c1-150
  grep -a "displaced\|background reads" ~/ps5lm/bench/sm_s35r_$n.err | tail -2 | grep -o "displaced.*\|[0-9.]* background.*"
  PROMPT="$PROSE" GEN=256 bash $D/smoke.sh s35p_$n $P318 $SERVE ${C[$n]} | head -1 | cut -c1-150
  grep -a "displaced\|background reads" ~/ps5lm/bench/sm_s35p_$n.err | tail -2 | grep -o "displaced.*\|[0-9.]* background.*"
done
echo "== done $(date +%T)"
