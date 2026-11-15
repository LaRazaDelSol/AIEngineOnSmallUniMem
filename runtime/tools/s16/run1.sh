#!/bin/bash
# S16: one cold run of the 300-token C++ benchmark prompt; prints md5 (plain greedy reference: 9c8d86dd397c) + stats.
#   bash run1.sh TAG [extra flash_decode args...]
cd ~/ps5lm/bench
B=~/ps5lm/build14
T=$1; shift
P="Implement a thread-safe LRU cache in C++20 with O(1) get/put, templated on key and value types. Include unit tests using a minimal assert-based harness and explain every design decision."
sudo systemctl stop ps5lm-server 2>/dev/null
IDS=$(python3 ~/ps5lm/src/scripts/console/tok.py "$P")
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
sudo $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --ids $IDS --gen ${GEN:-300} --pool-gib auto --bits 3 --predict 1 "$@" > s16_$T.out 2> s16_$T.err
echo "$T rc=$? md5 $(md5sum s16_$T.out | cut -c1-12) | $(grep -a 'tok/s' s16_$T.err | tail -1 | sed 's/prompt [0-9]* tok, generated [0-9]* tok: //')"
grep -a -E "fatal|error|MTP depth|mean accepted|MTP cost|trunk-only|verify|^spec|SSD|wait anatomy|by #demand" s16_$T.err | grep -v "^ *$" | head -20
