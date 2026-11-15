#!/bin/bash
# S17: predictor training data from the CURRENT engine/pack: per prompt, --dump-moe (MoE input of all 48 layers per
# token, f32) + --trace-out (routing). Diverse text (code / prose / chat / JSON tool calls / math / multilingual).
#   bash pred_data.sh OUTDIR GEN [extra flash_decode args]
OUT=${1:-~/ps5lm/pred17}; GEN=${2:-700}; shift 2
mkdir -p "$OUT"; cd ~/ps5lm/bench
B=~/ps5lm/build14
sudo systemctl stop ps5lm-server 2>/dev/null
P=(
"Write a Python asyncio web crawler with rate limiting, robots.txt handling and retries. Explain the design."
"Explain how a CPU branch predictor works, from two-bit counters to TAGE, with examples."
"Write a short story about a lighthouse keeper who discovers the light is talking to ships from the future."
"You are a tool-using agent. Respond ONLY with JSON tool calls. Task: find the three largest files under /var/log, compress them, and report sizes. Available tools: list_dir(path), stat(path), run(cmd), report(text)."
"Prove that the square root of 2 is irrational, then generalize to the square root of any non-square integer."
"Compare PostgreSQL MVCC with MySQL InnoDB's undo-log approach. Include failure modes and tuning advice."
"Write a Rust implementation of a lock-free single-producer single-consumer ring buffer with tests."
"Summarize the causes and consequences of the 1929 stock market crash for a high-school audience."
"Translate into Spanish and French, then explain the grammar differences: 'If I had known you were coming, I would have baked a cake.'"
"Design a REST API for a library management system. Give endpoints, JSON schemas, error codes and auth."
"Debug this: my React component re-renders forever when I call setState inside useEffect without dependencies. Explain and fix."
"Write a SQL query set for a sales database: top customers per region, month-over-month growth, and cohort retention."
"Explain the transformer attention mechanism mathematically, then implement it in NumPy step by step."
"Plan a 7-day trip to Japan for a family with two kids on a moderate budget, with daily itinerary and costs."
"Write a bash script that monitors disk usage, rotates logs, and emails an alert, with comments."
"Solve: a train leaves at 3pm at 60 mph, another at 4pm at 80 mph on the same track. When and where do they meet? Then generalize."
)
for i in "${!P[@]}"; do
  [ -s "$OUT/moe_$i.f32" ] && continue
  IDS=$(python3 ~/ps5lm/src/scripts/console/tok.py "${P[$i]}")
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  sudo $B/flash_decode --model ~/ps5lm/model --shaders $B/shaders --ids $IDS --gen $GEN --pool-gib auto --bits 3 --predict 1 \
       --dump-moe "$OUT/moe_$i.f32" --trace-out "$OUT/trace_$i.txt" "$@" > "$OUT/out_$i.txt" 2> "$OUT/err_$i.txt"
  echo "prompt $i rc=$? $(grep -a 'tok/s' "$OUT/err_$i.txt" | tail -1 | cut -c1-90)"
done
