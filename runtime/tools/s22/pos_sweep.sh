#!/bin/bash
# S22: decode speed vs prompt length (the S20/S21 "long-conversation decode 20-23 tok/s at 8k" item).
#   bash pos_sweep.sh TAG "300 2000 4000 8000" [extra flash_decode args...]
# Runs pf_ab-style CLI runs (server stopped, caches dropped) with the appliance's serving flags and prints, per N:
# prefill rate, gen-phase tok/s, spec tokens/cycle, verify anatomy (GPU-pure, gate B, demand misses), hit rate.
cd ~/ps5lm/bench
B=${B:-$HOME/ps5lm/build14}
TAG=$1; NS=$2; shift 2
SRC=${SRC:-$HOME/ps5lm/src/research_notes_long.txt}
sudo systemctl stop ps5lm-server 2>/dev/null
for N in $NS; do
  IDS=$(python3 - "$SRC" "$N" "${OFF:-0}" <<'EOF'
import sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file("/home/steam/ps5lm/model/tokenizer.json")
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
o = int(sys.argv[3])
ids = tk.encode(txt, add_special_tokens=False).ids[o: o + int(sys.argv[2])]
print(",".join(map(str, ids)))
EOF
)
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  O=ps_${TAG}_$N
  sudo PS5LM_VPROF=${VPROF:-} PS5LM_LMPROF=${LMPROF:-} $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --ids $IDS --gen ${GEN:-256} \
    --pool-gib ${POOL:-auto} --bits 3 --predict 1 --pack experts_exl3_268.pack --spec 1 --max-ctx 262144 --kv q8 --kv-base 8192 \
    --prefill-chunk 4096 "$@" > $O.out 2> $O.err
  rc=$?
  echo "== $TAG N=$N rc=$rc md5 $(md5sum $O.out | cut -c1-12) | $(grep -a -oP 'pool [0-9.]+ GiB|[0-9]+ slots' $O.err | head -2 | tr '\n' ' ')"
  grep -a -E '^prefill [0-9]+ tokens|tok/s \(gen phase\)|tokens/cycle|verify anatomy|spec time per cycle|MTP drafts:|^SSD \(gen|reads/tok\(gen\)' $O.err | sed 's/^/   /'
  grep -a -E "fatal|terminate" $O.err | head -3
done
