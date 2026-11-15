# Research Track — novel levers, falsification order, kill criteria

**Created:** 2026-10-21 (S5, creativity pass) · **Status:** ACTIVE — this file is the agenda for
the *research* half of the project. The conservative build path stays in the
[verified plan](../reports/PS5%2013.0%20engine%20plan%20-%20verified%20end-to-end.md) (P0–P6);
where this file and that plan disagree about *what to try next*, this file wins for research and
the plan wins for shipping. Every item is falsification-first: cheapest experiment decides
before any kernel-shaped code is written.

Ground rules inherited from the adversarial pass (`_work/adversarial/STATE.md`):
ground truth = recon JSON + primary sources; repo docs are claim sources; every new number
lands with provenance; every claim that dies gets a kill-list entry, not a silent edit.

---

## 0. What we're assuming dumbly (the four challenges, with the fix each unlocks)

### A1 — The ledger is decode-only; prefill/TTFT was never priced. It is the agentic product-killer.
Agents send 5–20K-token tool outputs. Streamed prefill at small ubatch costs ~1×G per token
(every token needs its 10 experts) → a 10K tool output ≈ 25 min of TTFT at 5.5 GB/s. No gate
measures any of this.

**The unlock (R1):** MoE routing is sub-linear in batch (Cohere data, `speculation_horizons.md`
Q1: batch-32 needs 78.8 unique experts, not 320). At **ubatch 512** a layer's union ≤ 512 ×
2.76 MB = 1.41 GiB — fits the 2–3 GiB slab — so stream the union per layer, compute, advance:

> prefill ≈ 48 × 1.41 GiB ÷ 512 tokens = **0.133 GiB/token → ~40 tok/s prefill at 5.5 GB/s,
> cache-independent** — matching dense-27B's 35–50. A 10K tool output: ~4 min, not 25.

PR #29887's "small batches only (≤32)" is a *VRAM-cache-path* constraint; the streaming path has
the opposite optimum. Zero model surgery — a scheduler policy. **New gate N8** (below).

### A2 — "The cache is 2–3 GiB" is a PS5-OS-title number, silently treated as *the* number.
Under ps5-linux (15 GB usable) a **layered pack** (dense tiers Q3/Q2, PLE left on disk while
speculation is off, experts Q4) leaves **~8 GiB for the expert cache** = ~22% of the REAP-288
store. The #29887 curve says 28.4% cap → 89% hit; discounted to agent-trace priors (~75–80%):

> **Linux track: ~15–22 tok/s decode with ZERO new engine code** (llama.cpp master +
> `--cpu-moe` + mmap — also dodges the #30697 Vulkan crash). The 6–10/12–25 bands describe the
> *native title*, not the platform.

Action: extend the N7 sweep to `--n7-cap-mb 8192 10240` and print this band in the ledger.

### A3 — "MTP OFF until G ≤ 0.45 GiB" used a fresh-draw prior.
The 1.3–1.6× bandwidth multiplier assumes draft tokens pull *fresh* experts. **Prompt-lookup /
n-gram drafts are copied from context** — and agent outputs are repetitive JSON (function
names, keys, tags), so drafted tokens route to *recently-used, already-cached* experts. The
multiplier collapses exactly where the workload is most repetitive. Prompt-lookup needs no MTP
weights and no PLE — the drafter is a string match.

**The unlock (R2):** a **cache-aware speculation gate** — accept a draft batch only if its
projected expert-miss cost < expected accept savings (readable from the cache index).
Testable in hours on the bench (`--dump-routing` + real agent traces).

### A4 — "REAP-288 → G 0.70" was a quant mix-up (S5-F23).
Pruning 512→288 experts shrinks the *store*, not bytes-per-read (router still picks 10/layer →
480 × 2,764,800 B ≈ 1.24 GiB/token at Q4). The 0.70 figure belongs to the **2-bit class**
(full-model Q2 ≈ 0.70 — consistent with the GSQ-RCO Q2_0 rows). sh0wie's shipped pack is
Q4_K_M/Q5/Q8 — **no 2-bit file exists**. So: shipped-pack REAP band ≈ **8–15 tok/s**; **12–25
requires a self-built 2-bit REAP-288 pack** (quality unmeasured). The 6–10 full-pack band is
unaffected. The bench decides; the constant was wrong, not the method.

### A5 — "40–60 is dead" is effort-dead, not physics-dead.
Three doors, each priced *somewhere* in the corpus, never composed:
1. **Basis-delta streaming** (`basis_delta_compression.md` Q3's own verdict: *"open and would
   be novel"*) — arXiv 2512.17073 streams rank-16 deltas at **0.32 MB vs 2.76 MB per expert**
   → G → ~0.15 GiB/token → 36+ tok/s *at any hit rate*; reconstruction compute < 0.2% GPU.
2. **Linux big cache** (A2) → ~20–29 tok/s.
3. **Frequency-tiered pack** (A4 done properly): hot experts Q6_K, cold Q2_K, PLE Q4_0 (free
   per qtum) — cuts G where reads are rare and spends bits where reads are common.

Honest framing: **dead at today's constants with today's effort; alive if any one of the three
lands.** The strategy note's "bragging-tier" positioning stands for effort-adjusted value.

### Killed ideas (checked, dead — kept so nobody re-opens them)
- **GDN state bomb**: 36 linear layers × per-seq state — block-diagonal per-head state =
  48×128×128 ≈ 0.79M params/layer → **~113 MB/sequence fp32**. Ten agents ≈ 1.1 GB. Fine.
- **Network expert server** (PC 990 PRO → PS5 over LAN): 1–2.5 GbE ≈ 0.1–0.3 GB/s vs 5.5 GB/s
  internal. Dead (`bandwidth_moonshots.md` Q7 agrees).
- **CPU expert compute / recompute-instead-of-read / two-PS5 cluster**: GPU is ~95% idle
  already; weights are max-entropy; LAN is the bottleneck. Dead, dead, dead.
- **Kraken-kill generalization**: the entropy argument correctly kills zlib-on-Q4 but does NOT
  apply to basis-delta residuals (structured, low-rank). Kill was right narrowly, wrong broadly.

---

## 1. The layer map — what's new code vs configuration

| Layer | What it is | Our role |
|---|---|---|
| Hypervisor / FW | PS5 13.00 | frozen, never touched |
| OS kernel | ps5-linux 7.1.7 | **configure only** (boot args, ttm) — we never write kernel code |
| Driver | amdgpu / RADV Mesa | **zero work** — real SDMA/compute queues under Linux; the N3 "no async" limitation is PS5-OS-titles-only |
| Runtime | ggml / llama.cpp | **where all our new code lives**: R1/R2 are host-side scheduler logic; R4's reconstruct is one optional ggml op (kernel-free v0: existing dequant + small GEMM, or CPU) |
| Pack format | GGUF / custom expert store | **offline tooling** (R3) — zero runtime risk |
| Serving / agent | llama-server :8081, MCP | exists |

**Rule: all novel research happens on the Linux track** (normal toolchain, real async queues,
gdb, fast iteration); the PS5-OS title is a packaging target, not a research environment.

---

## 2. Research items (falsification-first; cheapest experiment decides)

| # | Item | Hypothesis | Cheapest experiment (in order) | Gate → GO | Kill if |
|---|---|---|---|---|---|
| **R1** | **Two-regime scheduler**: large-ubatch layer-streaming prefill + pinned-cache decode | Streamed prefill at ubatch ≥512 costs ~0.13 GiB/token regardless of hit | Bench TTFT arms: 1K/4K/10K-token agent inputs × ubatch 32 / 512, on Strata (PC) then llama.cpp; measure TTFT + SSD bytes | TTFT at ubatch 512 ≥ 2× better than 32 on ≥1 input size, with no decode regression | No ≥2× prefill win on agent traces, or decode breaks |
| **R2** | **Cache-aware speculation gate** (+ prompt-lookup drafter) | Draft tokens from repetitive agent JSON reuse cached experts → net positive traffic | On/off arms on real agent traces (`--dump-routing` + bytes/token); gate = projected-miss < expected-savings | Net delivered-tok/s gain ≥ +10% at ≤ +10% bytes/token on both trace families | Net ≤ 0 on both families |
| **R3** | **Frequency-tiered REAP pack** (hot Q6_K / cold Q2_K / PLE Q4_0), built from OUR routing dumps | Mixed precision beats uniform at equal average G | Afternoon numpy probe: per-layer expert frequency + correlation + delta entropy from bench dumps → prototype pack (existing GGUF tooling) → bench + quality probe (perplexity + one agentic eval) | ≥1.5× effective-G cut at ≤ REAP-class quality loss | Delta/quant entropy ≈ independent (no compression win), or quality drop > REAP's |
| **R4** | **Basis-delta streaming store** (shared basis resident, per-expert delta streamed) — the corpus's own "open and would be novel" slot | Deltas are low-rank; streaming them beats streaming whole experts | Step 1: reconstruct ONE layer offline (D2-MoE/SLBF recipe), measure PPL/quality vs G saved. Step 2: CPU-reconstruct prototype on the bench. Step 3: ggml op (last) | Effective G saving ≥ 2× at ≤ 1% quality loss on the probe layer | Quality collapses, or saving < 2× |
| **R5** | **OEA resident-bias routing** (τ-nudge at BS=1) | Near-tie redirects onto resident experts cut misses ~free | Needs router-logit access in the runtime; prototype against bench dumps | ≥5% miss reduction, quality-neutral on probe | — (park until R1–R3 land) |

**Keystone:** R3's afternoon numpy probe decides the ROI of R3 *and* R4 before any kernel-shaped
work exists. Do it first.

## 3. Bench + gate additions

- **N8 (new gate): prefill/TTFT** — 1K/4K/10K-token inputs, both ubatch regimes, decode hit
  recorded alongside. *This is the agentic product's first impression; it gates shipping, not
  just research.*
- **N7 sweep extension**: add `--n7-cap-mb 8192 10240` runs (the Linux-track cache question).
- **Speculation arm**: ngram/lookup on/off on real agent traces (hours, existing llama.cpp
  flags) — decides A3/R2 empirically.
- **Ledger**: print the Linux-track band (A2) next to 6–10 / 12–25 / (8–15 shipped-REAP).

## 4. Documentation conventions for this track

- Each R-item's evidence lands in `research/notes/PS5 Flash-Next 40 tok-s frontier/` (or a new
  `research/lab-notebook/` folder when the first register is written) with provenance
  tags, and its verdict is recorded in `_work/adversarial/STATE.md` — never by silently editing
  the hypothesis.
- Killed items move to the kill list in `FLOW.md` with the falsifying number. Nothing is deleted.
- The conservative product path (P0–P6) is untouched by this file; when an R-item graduates
  (gate passed), the verified plan gets one row, not a rewrite.

---

## 5. S6 addenda (2026-10-23) — dated notes, hypotheses above unchanged

Registers: `research/lab-notebook/2026-10-23-exl3-flash-next-compression.md` (F27) and
`research/lab-notebook/2026-10-23-16gb-uma-flagship-precedent-sweep.md` (F28).

- **A3/R2:** Slipstream (npanj) measured hybrid MTP + Prompt-Lookup decoding on THIS model:
  tool-call decode **5.6 → 45+ tok/s**. The fresh-draw prior behind "MTP OFF until G ≤ 0.45" is
  now publicly contradicted for prompt-lookup drafts. R2's cache-aware gate is the unpublished
  remainder; Strata disables MTP whenever streaming, which is the gap the gate fills.
- **A4:** "no 2-bit file exists" is superseded in EXL3 form: 1.66/2.05/2.18 bpw self-calibrated
  branches of Flash-Next exist with KLD/PPL charts; 2.05bpw G ≈ 0.558 GiB/token (vs 1.24 Q4) and
  beats a 6.0bpw 27B on HumanEval+. REAP-288 2-bit GGUF still absent. PS5 runtime for
  trellis-family formats still required (one ggml op + repack) — the R4 op slot already reserved.
- **A5:** door 4 added — extreme compression as the 4th door, now evidence-backed rather than
  hypothetical. Door pricing table lives in the EXL3 register (2.05bpw @ h 0.45–0.72 → 17–33
  tok/s at 5.5 GB/s, pre-scheduler, pre-repack).
- **R1:** two shipped engines validate load/compute overlap on UMA (Slipstream F_RDADVISE,
  1.76× over llama.cpp; Edge0 prerouter, +59%). The 1K/4K/10K × ubatch 32/512 arms on a
  *streaming* engine remain unmeasured by anyone — N8 stays novel and is console-benchable
  (user directive: console-first; VM bench deferred).
- **R3:** production tooling exists for every ingredient (exllamav3 `-rcp` per-tensor recipes;
  Strata `<model>.experts` repack, +17/+30%; peonist-ai/crucible prune/merge toolkit; SGLang
  Expert Pack layout). The afternoon numpy probe remains the keystone — routing dumps now have
  THREE public sources to mimic (`--dump-routing` in Strata, `bench_eviction.py`, exllamav3
  expert_stats).
- **R4:** basis-delta remains unpublished in the 2026 field (checked); EXL3/QTIP trellis is the
  nearest shipped neighbor (shared codebook + compact residual per tensor vs per expert).
- **R5:** parked as before; shipped cousins noted (exllamav3 promote-on-miss + persisted
  per-layer frequency profiles).
- **Console-track tie-in:** PS5LM-Plus (native title: 12 GiB direct memory + ggml Vulkan on
  PS5_Vulkan/RADV) is now the strongest known packaging target for the PS5 endpoint; its 27B
  numbers (20.3–20.7 tok/s) match our published floor. Open question for the next console
  session: does it run on 13.00 Phat (their measurements: FW 6.02–12.70, Pro-heavy).
- **Composed engine (user question, 2026-10-23): feasible by composition** — see
  `research/lab-notebook/2026-10-23-composed-engine-feasibility.md` (F29). R1's
  mechanism already exists upstream as open PR #25294 (Wave-Partitional Prefill + per-layer
  expert slot caches + O_DIRECT pool); prompt-lookup is in-tree (`examples/lookup`); R2's
  cache-aware acceptance gate remains our novel piece. PS5-OS's no-file-mmap constraint selects
  the explicit-I/O pattern for the native title and keeps the layer map intact: research on
  Linux, package for the title, one fork, two I/O backends.
