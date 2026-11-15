#!/bin/bash
# S21: quality of the prefill path. The first M tokens of each long reference document go through prefill (layer-major /
# wide), the rest are teacher-forced and scored against the W4A16 reference (top-1 / KL20 / ppl, score_report.py).
#   bash score_pf.sh TAG M [extra flash_decode args...]
cd ~/ps5lm/qual
B=~/ps5lm/build14
T=$1; M=$2; shift 2
sudo systemctl stop ps5lm-server 2>/dev/null
sudo $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --pool-gib auto --bits 3 --predict 1 --pack experts_exl3_268.pack \
  --max-ctx 8192 --kv q8 --score docs_long.ids --score-ref ids_long20.txt --score-out score_pf_$T.txt --score-prefill $M "$@" > spf_$T.out 2> spf_$T.err
echo "$T rc=$? $(grep -a 'score:' spf_$T.err | tail -1)"
