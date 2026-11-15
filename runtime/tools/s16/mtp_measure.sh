#!/bin/bash
# S16: MTP head acceptance on the console (sequential decoding + a D-deep draft chain after every token).
# Output tokens must be identical to plain decoding (md5 vs cu40.out); the drafts are only scored.
#   bash mtp_measure.sh [DEPTH] [GEN] [TAG] [PROMPT]
cd ~/ps5lm/bench
B=~/ps5lm/build14
D=${1:-3}; G=${2:-300}; T=${3:-code}
P=${4:-"Implement a thread-safe LRU cache in C++20 with O(1) get/put, templated on key and value types. Include unit tests using a minimal assert-based harness and explain every design decision."}
sudo systemctl stop ps5lm-server 2>/dev/null
IDS=$(python3 ~/ps5lm/src/scripts/console/tok.py "$P")
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
sudo $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --ids $IDS --gen $G --pool-gib auto --bits 3 --predict 1 \
  --mtp-measure $D > mtp_$T.out 2> mtp_$T.err
echo "rc=$? md5 $(md5sum mtp_$T.out | cut -c1-12) (plain cu40: $(md5sum cu40.out | cut -c1-12))"
grep -a -E "MTP|tok/s|mean accepted|cost|trunk-only|slots\)" mtp_$T.err
