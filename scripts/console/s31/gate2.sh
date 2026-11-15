#!/bin/bash
# S31 gate 2 (one server stop): rebuild build-tier, smoke A/B: no tier, tier (pinned copy threads), tier with admission
# filters. Restarts ps5lm-server AND ps5lm-dash on exit. Output: ~/ps5lm/s31/gate2.log
#   bash gate2.sh ["variant flags" ...]   (default: the four below)
S=/home/steam/ps5lm/src-tier
B=/home/steam/ps5lm/build-tier
L=/home/steam/ps5lm/s31/${GLOG:-gate2}.log
exec > >(tee $L) 2>&1
restart() { sudo systemctl reset-failed ps5lm-server ps5lm-dash 2>/dev/null; sudo systemctl start ps5lm-server; sudo systemctl start ps5lm-dash; echo "server $(systemctl is-active ps5lm-server) dash $(systemctl is-active ps5lm-dash)"; }
trap restart EXIT
echo "== stop"; sudo systemctl stop ps5lm-server ps5lm-dash
for i in $(seq 60); do pgrep -x flash_decode >/dev/null || break; sleep 1; done
cd $B && ninja flash_decode ps5lm_shaders 2>&1 | grep -E ' error|FAILED|error:' | head -20
ninja flash_decode ps5lm_shaders >/dev/null 2>&1 && echo BUILD-OK || { echo BUILD-FAILED; exit 1; }
lsmod | grep -q "^ps5lowmem " || { echo "module not loaded"; exit 1; }
cd $S/scripts/console
if [ $# -eq 0 ]; then set -- "" "--tier /dev/ps5lowmem" "--tier /dev/ps5lowmem --tier-min-score 1" "--tier /dev/ps5lowmem --tier-min-score 2"; fi
i=0
for v in "$@"; do
  i=$((i + 1))
  echo "== v$i: ${v:-no tier}"
  B=$B bash smoke.sh ${GTAG:-g2v}$i $v
  grep -a -E "lowmem tier:|displaced weight" ~/ps5lm/bench/sm_${GTAG:-g2v}$i.err | tail -2
done
sudo dmesg | grep -ciE "mce|machine check|hardware error" | sed 's/^/MCE lines in dmesg: /'
echo "== done"
