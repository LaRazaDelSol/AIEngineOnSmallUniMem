#!/bin/bash
# S20 batched-prefill A/B: one long prompt (first N tokens of a text file), --spec 1 decode of GEN tokens.
#   bash pf_ab.sh TAG NTOK [extra flash_decode args...]     (prints md5 of the generated ids + prefill / decode rates)
cd ~/ps5lm/bench
B=~/ps5lm/build14
T=$1; N=$2; shift 2
SRC=${SRC:-$HOME/ps5lm/src/research_notes_long.txt}
IDS=$(python3 - "$SRC" "$N" <<'EOF'
import sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file("/home/steam/ps5lm/model/tokenizer.json")
txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
ids = tk.encode(txt, add_special_tokens=False).ids[: int(sys.argv[2])]
print(",".join(map(str, ids)))
EOF
)
sudo systemctl stop ps5lm-server 2>/dev/null
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
sudo PS5LM_LMSTAT=${LMSTAT:-} PS5LM_LMPROF=${LMPROF:-} PS5LM_IDX_GMIN=${IDXG:-4294967295} $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --ids $IDS --gen ${GEN:-64} --pool-gib ${POOL:-auto} --bits 3 --predict 1 \
  --pack experts_exl3_268.pack "$@" > pf_$T.out 2> pf_$T.err
echo "$T rc=$? md5 $(md5sum pf_$T.out | cut -c1-12) | $(grep -a '^prefill' pf_$T.err | tail -1) | $(grep -a 'tok/s' pf_$T.err | grep -v '^prefill' | tail -1 | grep -oP '[0-9.]+ tok/s \(gen phase\)') | $(grep -a -E 'slots|pool' pf_$T.err | grep -oP 'pool [0-9.]+ GiB|[0-9]+ slots' | head -2 | tr '\n' ' ')"
grep -a -E "fatal|error|terminate|spec gate|mean accepted" pf_$T.err | head -5
