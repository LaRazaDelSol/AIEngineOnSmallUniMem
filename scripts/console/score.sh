#!/bin/bash
# Console-side quality probe (S18 instrument): teacher-forced scoring of the 7 reference documents (3,532 positions)
# against the unmodified W4A16 reference's top-20 (~/ps5lm/qual/ref.txt, made with runtime/tools/s18/ref_logprobs.py).
#   bash score.sh TAG [extra flash_decode args...]
# SREF = the engine-format reference ("doc pos ids", ref3.txt); REF = the full one for the report (ref.txt).
# Prints top-1 agreement / KL20 / ppl for the run (and per document with VERBOSE=1) plus hit rate, reads/token and
# tok/s of the scoring pass. Exact reference point (EXL3-2.68, S18/S19): top-1 0.932, KL20 0.153.
# Takes ~2 min. Stops ps5lm-server first; restart it yourself afterwards.
B=${B:-$HOME/ps5lm/build14}
Q=~/ps5lm/qual
TAG=${1:-score}; [ $# -gt 0 ] && shift
cd $Q || exit 2
sudo systemctl stop ps5lm-server 2>/dev/null
for i in $(seq 30); do pgrep -x flash_decode >/dev/null || break; sleep 1; done
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
sudo $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --pool-gib auto --bits 3 --predict 1 --pack experts_exl3_268.pack \
  --score ${DOCS:-docs.ids} --score-ref ${SREF:-ref3.txt} --score-out sc_$TAG.txt "$@" > sc_$TAG.out 2> sc_$TAG.err
rc=$?
echo "$TAG rc=$rc | $(grep -a 'score: hit' sc_$TAG.err | tail -1) | $(grep -a 'cache-aware' sc_$TAG.err | tail -1)"
if [ "${VERBOSE:-0}" = 1 ]; then python3 score_report.py ${REF:-ref.txt} ${LABELS:-docs.ids.labels} sc_$TAG.txt:$TAG
else python3 score_report.py ${REF:-ref.txt} ${LABELS:-docs.ids.labels} sc_$TAG.txt:$TAG | grep -a -E " all |top-1" | head -3; fi
[ $rc -eq 0 ]
