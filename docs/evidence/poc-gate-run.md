# PoC evidence — engine end-to-end run (2026-10-26)

All commands: `python -m engine …` from a fresh clone; suite: `python -m unittest discover -s engine/tests` (16 tests).

## 1. Ledger invariants (S3 re-derivation enforced by tests)

```
ledger: G=1.24 GiB/token, store=5.1583999999999985 GiB/s
   hit |    tok/s
  0.00 |     4.16
  0.20 |     5.20
  0.40 |     6.93
  0.50 |     8.32
  0.60 |    10.40
  0.70 |    13.87
  0.80 |    20.80
  0.85 |    27.73
  0.90 |    41.60
  0.95 |    83.20

invariants: h40_is_0_896=OK h60_is_0_931=OK headline_dead=OK
```

## 2. Pack manifest (EXL3-class, 2.05 bpw)

```
$ python -m engine pack build --bpw 2.05 --out pack.json
pack: 26.3 GiB, G=0.565 GiB/token
```

## 3. R2 acceptance gate (agent trace, 8 GiB cache = Linux-track budget, target 20 tok/s)

```
# R2 gate report

**Verdict: PASS** (hit 91.1% vs gate 80%; projected 102.91 tok/s vs target 20.0)

| quantity | value |
|---|---|
| pack size | 26.29 GiB |
| G (expert bytes/token) | 0.565 GiB |
| store bandwidth (input) | 5.1583999999999985 GiB/s |
| cache budget | 8.0 GiB = 7477 slots (one pool, all layers) |
| trace | agent seed=7 tokens=2000 |
| expert reads replayed | 960000 |

*Synthetic trace - a public datapoint, not a console measurement. Replace with probe.py routing dumps per RESEARCH-TRACK R3/R4.*
```
## 4. Same pack, worst-case traffic AND a tight cache (0.25 GiB = PS5-OS-title-class) - the gate discriminates

```
# R2 gate report

**Verdict: FAIL** (hit 0.0% vs gate 80%; projected 9.13 tok/s vs target 20.0)

| quantity | value |
|---|---|
| pack size | 26.29 GiB |
| G (expert bytes/token) | 0.565 GiB |
| store bandwidth (input) | 5.1583999999999985 GiB/s |
| cache budget | 0.25 GiB = 233 slots (one pool, all layers) |
| trace | uniform seed=11 tokens=2000 |
| expert reads replayed | 960000 |

*Synthetic trace - a public datapoint, not a console measurement. Replace with probe.py routing dumps per RESEARCH-TRACK R3/R4.*
```