#!/bin/bash
# S24: speed matrix for cache-aware routing. Three prompts (reference C++ / prose / copy-heavy edit) x the flag sets
# given below; one cold run each through smoke.sh. Prints one line per run plus reads/token and SSD busy.
#   bash route_matrix.sh [TAG]
# The md5 verdict only means something for the exact rows of the reference prompt.
cd "$(dirname "$0")"
TAG=${1:-m}
PROSE="Write a detailed, well structured essay on the history of the Silk Road: its origins, the goods and ideas that moved along it, the empires that controlled it, and its influence on culture and trade today."
EDIT="$(cat prompts/copy_edit.txt)"
run() {  # name, prompt-kind, gen, flags...
  local n=$1 kind=$2 gen=$3; shift 3
  case $kind in
    ref)   GEN=$gen bash smoke.sh ${TAG}_$n "$@" | head -2 ;;
    prose) PROMPT="$PROSE" GEN=$gen bash smoke.sh ${TAG}_$n "$@" | head -2 ;;
    edit)  PROMPT="$EDIT" GEN=$gen bash smoke.sh ${TAG}_$n "$@" | head -2 ;;
  esac
  grep -a -E "^reads/tok|^SSD \(gen|^cache-aware routing: [0-9]+ sel" ~/ps5lm/bench/sm_${TAG}_$n.err | tail -3 | cut -c1-170
}
EX="--route-mtp 1"
B1="--route-mtp 1 --route-bias 0.5"
B3="--route-mtp 1 --route-bias 0.5 --spec 3 --pld 4"
for kind in ref prose edit; do
  gen=300; [ $kind = prose ] && gen=256; [ $kind = edit ] && gen=500
  run ${kind}_exact0 $kind $gen
  run ${kind}_exact $kind $gen $EX
  run ${kind}_b1 $kind $gen $B1
  run ${kind}_b3 $kind $gen $B3
done
