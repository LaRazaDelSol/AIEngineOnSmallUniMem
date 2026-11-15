# Adversarial review — “One PS5, One SSD, 100B+ MoE Streaming” (plan of 2026-10)

> **Status (2026-10-21):** SUPERSEDED — absorbed into the S2–S5 adversarial pass (_work/adversarial/FLOW.md + w1–w9 registers). Kept for the audit trail.

Reviewed: `research/reports/PS5 MoE streaming engine plan.md` + `research/notes/PS5 MoE streaming engine plan/*`
Date: 2026-10-17. Method: re-derivation of the plan’s own arithmetic, source spot-checks (web),
and looking for failure modes the plan doesn’t schedule a measurement for.

## Verdict

The plan is directionally right and unusually honest about its unknowns. The loader/research layer
holds up under spot-checking (see below). But it has **one real arithmetic error, one mis-specified
gate measurement, one understated platform-stability risk, and a systematic import of BC-250 numbers
onto PS5 silicon that its own notes flag as unresolved**. All four are fixable with measurements, and
the fix is cheap: probe the console before believing anything. That probing is now automated in
`tools/ps5probe/` (see `research/design-history/PS5-PROBE-RUNBOOK.md`).

## What spot-checked out (web, 2026-10-17)

- **pLinux5-NG / Byepervisor-Reborn**: live and current. README matches the plan’s characterization
  (“8.xx through 13.60”, port of ps5-linux-loader to the byepervisor-reborn escape). Reborn released
  ~13 Oct 2026 (4 days ago), 13.60 + 13.20 declared, 13.00 untested. Sources:
  github.com/pLabs5/pLinux5-NG, wccftech 13 Oct 2026, notebookcheck, r/PS5_Jailbreak.
- **TheFlow abandonment**: confirmed, 15–17 Sep 2026, “slop kiddies” reported the last hypervisor bug
  to Sony via HackerOne; GamingOnLinux/TweakTown/Cybernews all corroborate. Repos live, not archived.
- These are the plan’s two highest-volatility facts and they are accurate as of today — but both
  are <5 weeks old. See risk R3.

## Material findings

### A1 — The throughput expectation contradicts the plan’s own IO arithmetic (worst error)

The plan closes with “set expectations at 5–10 tok/s on 122B-A10B.” Its own cited facts refute that:

- Qwen3.5-122B-A10B Q4_K_M = **72.8 GB routed-expert data, 256 experts, top-8** (MoE-Direct spec).
- Bytes streamed per token = K × expert_bytes_per_token = 8/256 × 72.8 GB ≈ **2.28 GB/token**
  (the 8/256 ratio holds regardless of layer count).
- Sustained tok/s ≈ SSD_read_BW ÷ 2.28 GB, with the page-cache hit rate as a hidden multiplier.
  On the plan’s own gate number: **5.5 GB/s → ~2.4 tok/s; 6.5 GB/s → ~2.9 tok/s.**
  To hit 5 tok/s on Q4_K_M you need ~11.4 GB/s sustained — that’s Gen5 desktop territory, not a
  PS5 M.2 slot. The 2.3x-over-mmap MoE-Direct reference machine ran on Gen5 at 5.6 tok/s, which
  is consistent with ~13 GB/s there, not with 5.5.
- IQ3_S (43 GiB) → ~1.35 GB/token → ~4 tok/s at 5.5 GB/s. **Realistic single-PS5 122B expectation:
  1.5–4 tok/s direct-IO, up to maybe 5 with favorable cache locality.** The cluster’s 29.5 tok/s
  was 4 resident boards — a different regime entirely; using it as an anchor was the plan’s
  expectations error.
- **Missing from the plan: prefill/TTFT.** A 512-token prompt touches ~all experts once ⇒
  ≈72.8 GB read ≈ **13–15 s TTFT minimum** (IQ3_S: ~8 s) even with a perfect once-per-request read
  path (the “~3x prefill” trick is still in MoE-Direct’s *release queue*, unpublished). For a
  streaming 122B on one PS5, expect TTFT measured in tens of seconds and decode in single-digit
  tok/s. Phase 5’s QA gate should state this, or the first demo will look like a failure.

**Action:** restate the target as “2–4 tok/s on 122B, 10+ on 35B-A3B, TTFT ≤ ~30 s,” and make K,
quant, and the once-per-request prefill pass first-class Phase 4 levers.

### A2 — Gate 1 measures the wrong thing

“fio sequential read ≥5 GB/s” is a marketing spec (Sony quotes drives the same way), not the
engine’s operating point. The engine reads **~3 MB expert chunks, K-way parallel, mostly cold**,
against a page cache holding ~10–12 of 72.8 GB:

- Random 3 MB reads at qd8 on a Gen4 drive typically run 50–80% of sequential; the ps5-linux M.2
  **LBA-alignment bug** (TheFlow’s May 2026 fix — users had to reinstall to get it) can make random
  far worse than that. A sequential-only gate passes a misaligned install that dies in practice.
- The buffered (page-cache) regime is what flash-moe’s “trust the OS” architecture actually runs on;
  the plan never measures the re-read rate it assumes.
- Cache hit rate is the deciding hidden variable: required SSD bandwidth = tok/s × bytes/token ×
  (1 − hit_rate). flash-moe got ~71% hits with 35 GB cache over 209 GB; on PS5 (~10 GB cache over
  72.8 GB) expect 25–45%. The gate report must compute this, not hope.

**Action:** the gate is now a battery, not a number: seq 1 MiB qd32 O_DIRECT **and** randread 3 MiB
qd8 O_DIRECT **and** buffered re-read of a >cache file. If rand3M < 60% of seq1M ⇒ suspect LBA
alignment ⇒ re-run `m2_init`/`m2_install` with the current tools before failing the project.
All three are implemented in `tools/ps5probe/gate0_1.sh`.

### A3 — Platform fragility is understated

The plan treats “boot pLinux5-NG” as a weekend chore. It is:

- **A 5-day-old, self-described VERY-WIP loader** (only 13.20 confirmed; 13.60 experimental) built
  on a hypervisor bug that was *just* reported to Sony (that report killed ps5-linux-loader). The
  same bug class is already dead on 14.00/14.10. Any public 14.x update ⇒ permanent lockout of
  this path on the existing console (no downgrade — confirmed fact from the notes).
- M.2 on this path is **unconfirmed** (README never mentions it), and if M.2 doesn’t work you get
  “the build is dead before it starts” from a loader whose README also never promised it.
- Single-maintainer risk is now demonstrated, not hypothetical: TheFlow quit in Sep; Byepervisor
  ships no homebrew patches of its own; the whole stack depends on someone maintaining offsets
  against fast-moving FW.

**Action (cheap de-risk):** never PSN/update — FW freeze is the project’s scarcest resource.
Keep the native-OS jailbreak working as the provisioning/transfer channel (Pegasus) while the
Linux experiments run. Run the probe battery before, not after, any reinstall attempt so a failed
experiment leaves a paper trail. And see A5: the native path may not need to die with Gate 0.

### A4 — BC-250→PS5 number-import without an identity gate

The plan’s own notes flag that “PS5 = gfx1013 (RDNA1-class, à la BC-250)” vs “PS5 Oberon = gfx1030
(RDNA2)” is unresolved, yet the phases still import BC-250-derived numbers as levers:

- **The 40-CU unlock contradicts itself on PS5**: DF measures the PS5 at **full 36 CU** (no harvest
  mask); “40-CU unlock verified working in Linux for the ps5” is a DF *comment* that almost
  certainly transplants BC-250 lore (BC-250: 40 physical/24 active). On a 36-CU Oberon the unlock
  does not exist. The plan cites it in Gate 0 tuning — that line is noise at best.
- **The SMU governor (“biggest lever, +29.7%”) does not port**: it exists because BC-250 boards run
  a community BIOS; on PS5 the equivalent is `ps5_control --boost on` (already ships with
  ps5-linux-tools, 2230 MHz GPU). Listing the governor as “mandatory infrastructure” imports a
  nonexistent dependency.
- The ~290 GB/s effective bandwidth, 448 GB/s peak, and rm_kq/smin/fused-kernel deltas are all
  gfx1013-board measurements; on a real Oberon they are unverified (the plan does say A/B the
  shaders — good — but the budget numbers get used unflagged).

**Action:** identity is a 60-second Gate 0 measurement (lspci device id + `vulkaninfo --summary` →
`RADV GFX1013` vs `GFX1030`), and it is now in `gate0_1.sh`. Everything downstream keys off it.

### A5 — Gate 0 as written can kill a project that has a second platform

The plan’s only OS path is Linux (pLinux5-NG). But PS5LM already runs llama.cpp-Vulkan on the
**native** jailbroken OS at 7.00–13.60 — the user’s 13.x — with ~11 GB resident budget, and
Pegasus proves the native OS can read/write the internal NVMe fast. Native streaming has real
obstacles (no raw block access from homebrew, no O_DIRECT, OS page cache behavior under Orbis-like
vfs, no fio) — so it’s not obviously viable, and that’s exactly the point: **the plan never
states why the native path was ruled out, and a binary “no Linux ⇒ stop” gate discards the one
path that works on the console today.** A 30-minute native probe (Pegasus copy-throughput test on a
large file on the internal SSD, implemented as `probe.py native --copytest`) sets a floor for
native-streaming bandwidth before betting everything on a 5-day-old loader.

### A6 — Storage topology and provisioning are unspecified

- ps5-linux puts **Linux itself on the M.2** (exclusive partition, USB stick still needed to boot).
  So the streaming target and the OS are the same device: page cache, journal, and model streaming
  contend. Fine for the gate (fio measures the device), but the engine plan should place expert
  files on the M.2 and keep OS hot path off it (log to tmpfs, etc.) — or run OS from USB and stream
  from a cleanly dedicated M.2. The plan never chooses.
- **Provisioning 72.8 GB has never been costed.** Over 1 Gbps LAN that’s ~10+ min per full image;
  WiFi is hopeless. The workflow should be: build the expert store (unpack + per-expert SHA-256
  manifest, MoE-Direct style) **on the PC**, push once via Pegasus upload / scp, verify hashes on
  target, never re-push. Pegasus `/api/files/upload` + `/api/files/operations` already give the
  pipeline (contract-tested 21/21 in this repo).

### A7 — Memory budget conflates three platforms

11 GB (PS5LM, native OS), ~15 GB usable (DF, ps5-linux), ~13 GiB working ceiling (BC-250, akandr).
The plan picks 13 GiB without measuring on-target. Resident budget = weights + KV + compute on PS5
is a function of loader cmdline (`vram.txt`, `amdgpu.gttsize`), the ttm/ttmpfiles trap, and driver
defaults — all observable in one shot at Gate 0. The probe now dumps: cmdline, both ttm params,
gttsize, `mem_info_vram_*`, `free`, and largest successful Vulkan allocation via vulkaninfo heaps.
The plan should adopt the **measured** number as the KV+resident budget and derive context length
from it.

### A8 — Source-quality load-bearing

Critical-path numbers rest on: EaseUS (5.5/9.9 GB/s — a data-recovery vendor blog), a 2021 Reddit
thread (M.2 ~6.6 GB/s), one DF comment (kernel params; the tmpfiles trap is real and independently
documented by akandr — good), one cluster operator (the 29.5 tok/s and IQ4_XS-parity claims,
unreproduced). The plan flags these in its notes, then quotes them in go/no-go thresholds anyway.
Thresholds should come from **your console** (the probe), with literature only as priors.

### A9 — Kraken upside still overstated even as “optional”

Even if someone drove the Kraken unit from Linux, it only helps if experts are *stored* in a
format the unit decodes (Sony’s I/O decompression is tied to their own container stack; ext4 GGUFs
are not it). Combined with flash-moe’s LZ4 result (−13%), decompression-on-read is the wrong shape
for this engine. **Action: delete it from Phase 4, keep the note, do no work.**

## What the plan gets right (don’t change)

- Loader landscape (repos, FW table, M.2-on-4.xx+ gating, the 3.xx boot-with-M.2 bug, TheFlow,
  Reborn mechanics): verified accurate today.
- Both `ttm.pages_limit` **and** `page_pool_size`, and the systemd-tmpfiles silent-clobber trap —
  probe asserts the live values, per plan.
- pread over mmap (two independent platforms agree), no expert prediction, no speculation on
  Vulkan, serial pipeline on UMA, K as the tunable dial, byte-preserving SHA-256 correctness story,
  A/B the shader changes on-console instead of trusting BC-250 deltas.
- Ordering: measurement gates before engine code. This review only makes the gates more numerous.

## Corrected gate table (as implemented in tools/ps5probe)

| Gate | Measurement | Go | Caution | No-go |
|---|---|---|---|---|
| G0 boot | pLinux5-NG on exact FW, amdgpu bound, login reachable | all | boots, GPU not up | no boot |
| G0 identity | lspci id + vulkaninfo device string | record GFX1013/GFX1030 | — | — (informs A/B) |
| G0 M.2 | second data NVMe enumerates under Linux | present | visible not mountable | absent → USB-enclosure pivot |
| G0 memory | ttm params live == 4194304 (pages_limit **and** page_pool_size) | both | clobbered to 3145728 → tmpfiles fix | can’t raise |
| G0 extmem | `VK_EXT_external_memory_host` in device extensions | present | — | absent → copy-landing fallback (slower, not dead) |
| G1 seq | fio seq 1M qd32 O_DIRECT | ≥5000 MB/s | 3000–5000 → 35B target | <3000 → 100B dead |
| G1 rand3M | fio randread 3M qd8 O_DIRECT | ≥3500 MB/s | <60% of seq → LBA-align reinstall | — |
| G1 cached | buffered re-read ≥10 GB/s | yes | — | OS cache weak → direct-IO engine design |
| N0 native | Pegasus alive, mounts, ≥1 file copy ≥150 MB/s on /data | floor for A5 | — | — |

Projections the report prints after G1: tok/s = BW_eff ÷ (K/E × expert_GB) for the chosen
model/K/quant, plus TTFT ≈ expert_GB (once-per-request prefill). Numbers come from your console,
not from EaseUS.
