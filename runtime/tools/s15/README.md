# S15 study scripts (PC side)

Working files go to `$PS5LM_STUDY_DIR` (default `./ps5lm_s15`); routing traces are read from `$PS5LM_TRACES`
(default `./traces`, text traces written by `runtime/tools/trace_to_txt.py`).

| script | input | output |
|---|---|---|
| `trace_ledger.py` | real routing traces | LRU misses/token per pack geometry, verify-window unions, miss rank, two-tier bytes |
| `coldwarm.py` | real traces | cold vs warm-pool misses for a single request |
| `fetch_experts.py` | HF safetensors (HTTP range) | sample BF16 experts (hot/median/cold) at layers 2/10/30/45 |
| `qstudy.py` | sample experts + `flash_decode --dump-moe` activations + trace | expert-output error per code (PS5Q, RHT, GPTQ/LDLQ, NF levels) vs bpw |
| `predictor.py` | `--dump-moe` activations + trace + router weights | cross-layer predictor precision and LRU miss coverage |
| `export_pred.py` | same | `pred_d{d}_l{l}.bf16` trained predictor matrices (copy to `model/pred_s15/`) |
