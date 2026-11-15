#!/bin/bash
# S16: prefetch submitted with the demand misses (S15 default) vs after them (--prefetch-late), cold, same prompt.
cd ~/ps5lm/bench
B=~/ps5lm/build14
P="Implement a thread-safe LRU cache in C++20 with O(1) get/put, templated on key and value types. Include unit tests using a minimal assert-based harness and explain every design decision."
sudo systemctl stop ps5lm-server 2>/dev/null
IDS=$(python3 ~/ps5lm/src/scripts/console/tok.py "$P")
run() {  # tag, extra args
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  sudo $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --ids $IDS --gen 300 --pool-gib auto --bits 3 $2 > late_$1.out 2> late_$1.err
  echo "$1 [$2] md5 $(md5sum late_$1.out | cut -c1-12) | $(grep -a 'tok/s' late_$1.err | tail -1 | sed 's/prompt 56 tok, generated 301 tok: //') | $(grep -a 'reads/tok' late_$1.err)"
}
for cfg in "early1|--predict 1" "late1|--predict 1 --prefetch-late" "late2|--predict 2 --prefetch-late" "early2|--predict 2"; do
  run "${cfg%%|*}" "${cfg#*|}"
done
