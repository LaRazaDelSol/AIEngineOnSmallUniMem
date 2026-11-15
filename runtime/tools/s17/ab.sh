#!/bin/bash
# S17 A/B driver: each argument is one config "TAG|flags..."; one cold 300-token run each (run1.sh), compact summary.
cd ~/ps5lm/bench
for cfg in "$@"; do
  tag=${cfg%%|*}; flags=${cfg#*|}; [ "$flags" = "$cfg" ] && flags=""
  bash run1.sh "$tag" $flags > /dev/null 2>&1
  l=$(grep -a 'tok/s' s16_$tag.err | tail -1)
  printf "%-14s md5 %s | %s tok/s | gpu %s | io %s | hit %s | pure %s | ssd %s\n" "$tag" "$(md5sum s16_$tag.out | cut -c1-12)" \
    "$(echo "$l" | grep -oP '[0-9.]+(?= tok/s)')" "$(echo "$l" | grep -oP 'GPU\+sync \K[0-9.]+')" "$(echo "$l" | grep -oP 'I/O wait \K[0-9.]+')" \
    "$(echo "$l" | grep -oP 'hit \K[0-9.]+')" "$(echo "$l" | grep -oP 'GPU-pure \K[0-9.]+')" "$(grep -a '^SSD' s16_$tag.err | grep -oP 'busy \K[0-9.]+%')"
  grep -a -E "prefetch outcome|fatal|error" s16_$tag.err | head -3 | sed 's/^/    /'
done
