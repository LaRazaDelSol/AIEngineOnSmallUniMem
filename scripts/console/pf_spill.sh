#!/bin/bash
# S24: prefill of one 6,000-token document with the indexer's score-spill path forced (PS5LM_IDX_GMIN=G) or not.
# The spill path is what every prefill beyond 32,768 tokens of context runs; forcing it shows its cost at a short
# context in one minute instead of a 10-minute long-context run.
#   bash pf_spill.sh TAG [G|-] [extra flash_decode args...]        ("-" = default threshold)
B=${B:-$HOME/ps5lm/build14}
TAG=${1:-spill}; G=${2:--}; shift 2 2>/dev/null
cd ~/ps5lm/bench
sudo systemctl stop ps5lm-server 2>/dev/null
for i in $(seq 30); do pgrep -x flash_decode >/dev/null || break; sleep 1; done
IDS=$(sed -n ${DOC:-1}p ~/ps5lm/qual/docs_long.ids)
ENVV=""; [ "$G" != "-" ] && ENVV="PS5LM_IDX_GMIN=$G"
sudo timeout -s TERM ${TMO:-300} env $ENVV ${PFENV:-} $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --ids $IDS --gen ${GEN:-8} --pool-gib auto --bits 3 --predict 1 \
  --spec 1 --max-ctx 262144 --kv q8 --kv-base 8192 --prefill-chunk 4096 "$@" > pf_$TAG.out 2> pf_$TAG.err
echo "$TAG rc=$? md5 $(md5sum pf_$TAG.out | cut -c1-12) | $(grep -a '^prefill [0-9]' pf_$TAG.err | tail -1)"
grep -a -E "prefill anatomy|lmprof|LMPROF" pf_$TAG.err | cut -c1-260 | head -20
