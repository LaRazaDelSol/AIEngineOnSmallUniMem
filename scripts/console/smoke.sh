#!/bin/bash
# Console-side smoke probe: ONE cold 300-token run of the reference C++ prompt with the serving flags.
#   bash smoke.sh [TAG] [extra flash_decode args...]
# Pass = generated-token md5 18179c0e1f51 (EXL3-2.68 greedy reference since S17; every lossless change must keep it).
# Takes ~45 s. Stops ps5lm-server first (two engines cannot share the pool); restart it yourself afterwards.
REF=${REF:-18179c0e1f51}
B=${B:-$HOME/ps5lm/build14}
TAG=${1:-smoke}; [ $# -gt 0 ] && shift
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p ~/ps5lm/bench && cd ~/ps5lm/bench
P="Implement a thread-safe LRU cache in C++20 with O(1) get/put, templated on key and value types. Include unit tests using a minimal assert-based harness and explain every design decision."
sudo systemctl stop ps5lm-server 2>/dev/null
for i in $(seq 30); do pgrep -x flash_decode >/dev/null || break; sleep 1; done
IDS=$(python3 "$HERE/tok.py" "${PROMPT:-$P}")
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
sudo $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --ids $IDS --gen ${GEN:-300} --pool-gib auto --bits 3 --predict 1 \
  --pack experts_exl3_268.pack --spec 1 "$@" > sm_$TAG.out 2> sm_$TAG.err
rc=$?
MD5=$(md5sum sm_$TAG.out | cut -c1-12)
VERDICT=DIFF; [ "$MD5" = "$REF" ] && VERDICT=PASS
echo "$TAG rc=$rc md5 $MD5 $VERDICT | $(grep -a 'tok/s' sm_$TAG.err | tail -1 | sed 's/prompt [0-9]* tok, generated [0-9]* tok: //') | $(grep -a -oP '[0-9]+ slots' sm_$TAG.err | head -1)"
grep -a -E "fatal|terminate|^spec k=|^PLD|pack mirror|tokens/cycle" sm_$TAG.err | head -8
[ $rc -eq 0 ]
