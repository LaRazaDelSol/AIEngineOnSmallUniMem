# PS5 MoE Expert Streaming — Storage/Streaming-IO Levers (as of Oct 2026)

> **Status (2026-10-21):** ACTIVE evidence base — numbers are claims; re-derive against the W4 recompute before citing (the parent report had ledger arithmetic errors, W1/FLOW kill list).

Researcher notes for the report writer. All external facts carry inline sources; PS5 platform constants given in the assignment (825GB NVMe ~5.5 GB/s OS reads, 16GB GDDR6 ~448 GB/s, Zen2 8C/16T, no Kraken Linux driver) are treated as inputs, not findings. The target engine is a llama.cpp-Vulkan expert-streaming engine on pLinux5; "target model" streams 10/512 routed experts per token, ~1.2–2.3 GB/token, expert tensors 2–6 MB read K-way parallel.

---

## Q1. Kernel I/O paths: io_uring vs pread, O_DIRECT vs buffered, llama.cpp dio

### Takeaway
For the streaming use-case, the syscall choice is a second-order lever (io_uring ≈ up to 2× libaio at high depth; plain `pread()` with parallel threads is what the best MoE streaming engines actually use); the first-order lever is **explicit slice reads + on-disk layout**, and avoiding O_DIRECT for anything you want re-read (O_DIRECT is 10× slower on repeat loads and llama.cpp's dio mode is widely reported as a footgun). Measured: explicit mincore-checked `pread()` prefetch of expert slices beats mmap page-fault streaming by +13–14% end-to-end.

### Cited Findings
- llama.cpp `--load-mode` now folds the old `--no-mmap` / `--mlock` / `--direct-io` flags; "LOAD_MODE=dio" is the page-cache-bypassing mode meant for pass-through weight loads — [openclawdc](https://openclawdc.com/blog/llama-cpp-flags-explained/); PR context in [llama.cpp discussion #27149 comment](https://github.com/ggml-org/llama.cpp/discussions/27149) (flags `--moe-stream-cache`, `--moe-stream-direct` (O_DIRECT) in PR [#25294](https://github.com/ggml-org/llama.cpp/pull/25294)).
- MoE models ≥100% of RAM are faster with mmap+no-dio than dio; O_DIRECT "takes at least 10x longer" on repeated loads because nothing is cached; man-page itself warns O_DIRECT "in general will degrade performance" (Linux 6.14, x86, Jan 2026 thread) — [llama.cpp discussion #18758](https://github.com/ggml-org/llama.cpp/discussions/18758); gpt-oss-120b reportedly failed to load in one build *without* `--no-direct-io` — same thread.
- Independent load-time measurement: mmap-disabled + direct-I/O made a Llama-4 runner ready in ~18–20 s (the article is about load-time, not steady-state streaming) — [sheldylew.com](https://sheldylew.com/notes/why-llama-4-took-minutes-to-load).
- io_uring measured up to **2× faster than libaio**; best io_uring config 1.4× faster on newer kernels vs older; an IOMMU misconfig can silently negate the gain — [YDB blog](https://blog.ydb.tech/how-io-uring-overtook-libaio-performance-across-linux-kernels-and-an-unexpected-iommu-trap-ea6126d9ef14).
- ClickHouse's storage-engine study: best random-read block size is **4 KB for NVMe** (8 KB SATA, 256 KB HDD); async (io_uring/libaio) matters most at high concurrency; they moved libaio→io_uring for throughput — [ClickHouse blog](https://clickhouse.com/blog/a-journey-to-io_uring-aio-and-modern-storage-devices).
- io_uring survey for DBMSs: registered/fixed buffers and `IORING_OP_URING_CMD` (native NVMe passthrough to device queues) are the two big extra levers; benefit concentrates in high-IOPS, many-thread regimes — [arXiv 2512.04859](https://arxiv.org/html/2512.04859v1); fio/io_uring microbenchmarks on Optane in [AtLarge thesis 2024](https://atlarge-research.com/pdfs/2024-bingimarsson-msc_thesis.pdf); small-random-read io_uring benchmark harness — [zarr-benchmark discussion](https://github.com/zarr-developers/zarr-benchmark/discussions/26).
- flash-moe's engine (M3 Max, 4.4 tok/s @ 209 GB model) uses **parallel plain `pread()`** with "GCD dispatch groups" for K=4 experts (~6.75 MB each); it found `dispatch_io` −70% and `mmap` of expert files −5× (per-page-fault overhead on cold data) — [danveloper/flash-moe README](https://github.com/danveloper/flash-moe).
- VLDB storage paper: Linux md RAID has a hard scalability limit ~15 GB/s; SSD random-read scaling needs multiple queues/threads (relevant for striping, Q2) — [VLDB p2090-haas](https://www.vldb.org/pvldb/vol16/p2090-haas.pdf).

### Inferences
- At 2–6 MB per expert read, K-way parallel, we are bandwidth-bound, not IOPS- or syscall-bound; io_uring's headline gains (IOPS at high depth) barely apply. The realistic delta from io_uring + registered buffers + SQPOLL vs 10 parallel `pread()` threads is small; the win would come only from eliminating copy into page cache + per-IO syscall — worth prototyping with O_DIRECT+io_uring+registered fixed buffers on pLinux5 but not assumed to exceed ~1.1–1.3×.
- flash-moe's −5× mmap result and mbolt's "mmap faults are layout-blind" (Q6) both argue: **use explicit positional reads (pread/io_uring READ at offsets), never mmap, for expert tensors**; keep mmap only for shared/non-expert weights.
- For PS5: buffered I/O + parallel pread is the safe default; O_DIRECT only for a dedicated "cold-stream" path (single-use expert bytes) to avoid evicting the page cache — matching llama.cpp dio's stated intent.

### Gaps
- No published measurement of io_uring (registered buffers/SQPOLL) vs pread specifically on a PS5/Oberon under pLinux5 or of llama.cpp `--load-mode` variants with expert streaming on Vulkan.
- NVMe `num_queues`/`queue_depth` module-parameter tuning deltas on PS5's Samsung/Marvell controller under Linux: nothing published found.

---

## Q2. Two-device striping (internal NVMe + M.2)

### Takeaway
No one has published PS5 dual-SSD streaming. Sony's own cap makes the M.2 slot mirror the internal drive's speed under PS5-OS (~6.5 GB/s benchmark figure), but under pLinux5 both drives would appear as ordinary NVMe devices and app-level striping should give roughly additive read bandwidth (825GB internal ~5.5 GB/s + a Gen4 M.2's native ~7 GB/s ≈ 10–12 GB/s theoretical) — with Linux md RAID0 as the cheap but ~15 GB/s-capped option, and app-level striping (two file trees) as the uncapped option. The blocker is uncertainty whether the internal NVMe is even reachable/performant under the pLinux5 hypervisor stack.

### Cited Findings
- Sony: M.2 expansion drive is capped to "match or not fall below" internal-SSD speed; majority of M.2 drives "will match" internal speed even if rated >5500 MB/s — [playstation.com](https://www.playstation.com/en-us/support/hardware/ps5-install-m2-ssd/); users see Sony's own benchmark report **6525 MB/s** for fast M.2 drives — [reddit](https://www.reddit.com/r/playstation/comments/123zgrr/m2_ssd_says_read_speed_on_ps5_is_6525_mbs_when_it/).
- ps5-linux-loader runs Linux entirely from a **USB drive** (64 GB+, ideally external SSD) — the shipped workflow does not use internal NVMe — [ps5-linux/ps5-linux-loader](https://github.com/ps5-linux/ps5-linux-loader); scene confirmation that pLinux runs from external USB storage — [Hackaday, May 2026](https://hackaday.com/2026/05/01/running-linux-on-the-ps5-with-a-hypervisor-exploit/); users ask whether even a Gen3 SSD is fast enough for Linux-on-PS5 — [r/PS5_Jailbreak](https://www.reddit.com/r/PS5_Jailbreak/comments/1wydud6/do_you_think_a_gen_3_ssd_is_fast_enough_for_linux/).
- pLinux5-NG (Oct 2026) boots on firmware ≤13.60 via Relapse + Byepervisor-Reborn hypervisor exploit; it is a *Linux-under-hypervisor* setup (PS5-OS still owns hardware underneath) — [notebookcheck](https://www.notebookcheck.net/Linux-now-runs-on-PS5-console-jailbreaks-and-July-firmware-with-PC-games-possible.1425575.0.html), [pLabs5/pLinux5-NG](https://github.com/pLabs5/pLinux5-NG).
- Generic dual-NVMe aggregate: Linux md RAID0 tops out ~15 GB/s but scales well below that limit (so two Gen4 drives ≈ 10–12 GB/s aggregate reads is realistic in the kernel stack) — [VLDB p2090-haas](https://www.vldb.org/pvldb/vol16/p2090-haas.pdf).
- No published dual-SSD LLM-streaming prior art found (see Gaps).

### Inferences
- pLinux5 running under the PS5 hypervisor probably does not own the internal NVMe: the bootloader historically passes control to Linux with USB as the only practical mass storage (loader README, Hackaday). Whether `/dev/nvme0` (internal) or the M.2 slot's `nvme1` enumerate under pLinux5's patched kernel is *the* first experiment to run — it decides whether the 5.5 GB/s internal drive is a usable lever or a fiction.
- If both drives enumerate, app-level striping (expert N → device N%2, K-way parallel reads issued to both) beats md RAID0 (no md layer in the ~15 GB/s ceiling path, per-expert contiguity preserved, no resync/parity pathology) — [inference from VLDB md limits](https://www.vldb.org/pvldb/vol16/p2090-haas.pdf) + Q6 layout findings.
- PCIe topology risk: both SSDs feed the same Gen4 x16-derived complex and (on an APU) the same memory controller that the GPU uses; aggregate gain will be <sum of singles and interacts with Q7.

### Gaps
- No measured Linux-on-PS5 (pLinux5) sequential-read numbers for internal NVMe or M.2; nothing found on ps5dev/GitHub issues. Assignment itself notes the M.2-under-Linux figure is unmeasured.
- No published example of LLM weight streaming across two simultaneously-read NVMe devices; only generic storage-literature RAID0 scaling.

---

## Q3. Compression-on-read (CPU and GPU decompression)

### Takeaway
CPU decompression on Zen-class cores comfortably out-runs any single PS5 SSD path (LZ4 ≫4 GB/s/core, zstd-L1 ≈ 0.8–1.5 GB/s/core, ISA-L igzip up to ~10 GB/s multithreaded on gzip streams), so a "read compressed, decode on the 8 idle-ish Zen2 cores" pipeline is arithmetically sound *when cores and memory bandwidth are free* — but flash-moe measured −13% for an LZ4 expert cache in a streaming engine (decode overhead + extra copy beat the bandwidth saved on a fast Apple SSD), and PS5 has the additional twist that its hardware Kraken unpacker, worth "~9× a Zen2 core" per contemporary claims, has no Linux driver (assignment premise).

### Cited Findings
- LZ4: compression >500 MB/s/core, decoder "extremely fast" (typically >2–4 GB/s/core on modern cores; community tables report >3.5 GB/s) — [lz4/lz4 README](https://github.com/lz4/lz4), [dev.to compressor comparison](https://dev.to/konstantinas_mamonas/which-compression-saves-the-most-storage-gzip-snappy-lz4-zstd-1898); mixed lz4-vs-zstd decode ordering depends on data — [HN thread](https://news.ycombinator.com/item?id=36826207).
- zstd decode held ~1 GB/s/core "across all levels" in that comparison — [dev.to](https://dev.to/konstantinas_mamonas/which-compression-saves-the-most-storage-gzip-snappy-lz4-zstd-1898). Multithreaded zstd/lz4 wrappers measured >3 GB/s on 8 threads — [R bloggers](https://www.r-bloggers.com/2018/01/multithreaded-lz4-and-zstd-compression-from-r/). (No measurement found for `zstd --format=bitflip` specifically — gap below.)
- ISA-L/igzip: independently measured **up to 10 GB/s gzip-decompress** (multithreaded, >20 GB/s with prebuilt index) — [HN benchmark thread](https://news.ycombinator.com/item?id=37270722); rapidgzip (parallel gzip, 48–128 cores) 1.2 GB/s indexed-free / much higher indexed — [arXiv 2308.08955](https://arxiv.org/pdf/2308.08955); Gzip 1.14 shipped the same huffman-speedup for both Intel & AMD — [Phoronix](https://www.phoronix.com/news/Gzip-1.14-Released).
- PS5 Kraken hardware unit: decompression "equivalent of ~9× Zen2 cores" (community figure, 297% faster than zlib software) — [NeoGAF thread](https://www.neogaf.com/threads/kraken-vs-zlib-29-smaller-game-sizes-losslessly-297-faster-decompression-on-ps5.1570901/); former Frostbite engineer confirms dedicated Kraken unit, ratio ~1.5:1 typical → games quote ~5.5 GB/s raw as ~8–9 GB/s effective — [gamingbolt](https://gamingbolt.com/former-frostbite-software-engineer-explains-ps5s-kraken-compression-unit), [r/PS5 cbloom writeup](https://www.reddit.com/r/PS5/comments/iza3u3/how_oodle_kraken_and_oodle_texture_supercharge/); ryg: Kraken decodes ~4× faster than deflate — [fgiesen blog](https://fgiesen.wordpress.com/2024/08/08/oodle-kraken-etc-misconceptions/).
- **Negative result**: flash-moe tried an LZ4-compressed expert cache and it was **−13%** ("decompress overhead > warm cache savings") — [flash-moe README](https://github.com/danveloper/flash-moe).
- SSD-vs-decompressor arithmetic in the "SSD slower than CPU decode" regime is standard in storage engineering (decompression runs far faster than disk, so compressed storage is transparent win when CPU-free) — [ZFS discussion measuring on 8-thread Ryzen 1700](https://zfsonlinux.topicbox.com/groups/zfs-discuss/T228f2ac87c8f6d0f/zfs-lz4-and-zstd-compression-benchmarks-on-a-8-thread-ryzen-1700-was-re-zfs-lz4-and-zstd-compression-benchmarks-on-a-8-thread-ryzen-1700); academic MoE-SSD energy study notes severe energy penalty of SSD-resident experts (second-order argument for fewer bytes moved) — [alphaXiv 2508.06978](https://www.alphaxiv.org/abs/2508.06978).
- GPU-side: NVIDIA RTX IO/Deflate work exists; **no RADV/ROCm hardware-deflate or GPU-decompress-on-AMD for weights found** (gap below). No ozip-on-console GPU path found.

### Inferences
- PS5 adaptation: on the 8×Zen2 with the GPU busy on dequant, spare decode throughput is likely 3–8 GB/s (LZ4 across a few pinned cores) — enough to make compression a *capacity* tool (fit 2× more experts in page cache / cut streamed bytes by ~1.4–2× for 4-bit quant data with LZ4/zstd-1 block-caches) but flash-moe's −13% warns it is only net-positive when the bottleneck is genuinely SSD bandwidth and caches are cold, and only if decode runs on cores that don't contend with GPU-feeding DMA. Quant weights (4-bit packed) compress poorly; the win must come from per-block scale/bias and dead-slab (zero) blocks — measure before committing.
- The Kraken unit (Q4) dominates every CPU codec: ~1.5:1 ratio at hardware speed with zero CPU cost. If a driver existed, it would be the single best lever; without it, zstd-level-1 with a small per-tensor dictionary, on 2–4 cores, is the pragmatic CPU pick (best ratio-per-GHz for already-quantized tensors), LZ4 when bandwidth-per-core matters.

### Gaps
- No `zstd --format=bitflip` (transposition prefilter for ML data) decode-throughput measurement found.
- No QPL/igzip benchmark run specifically on Zen2 found (igzip numbers above are Intel-tuned but run on AMD via QPL; the 10 GB/s figure is multi-core).
- No AMD GPU-side (Vulkan/RADV) deflate decompression project found; NVIDIA-only hardware-deflate (RTX IO) exists.

---

## Q4. Kraken / PS5 I/O unpacker reverse-engineering status

### Takeaway
The PS5 I/O unpacking unit (Kraken/Oodle hardware decompressor) remains un-driven from Linux or homebrew in any shipped, public project: software decoders for the Kraken bitstream exist in the compression community (ozip-lineage work discussed on encode.su), the format itself is documented-enough to re-encode, and scene chatter says an FPKG-backup alternative drives kraken decompression **via the A53 co-processors**, but no ps5dev payload or Linux driver exposes the dedicated unit. Feasibility of Kraken-encoded experts under pLinux5 today = software decode only.

### Cited Findings
- PS5 ships a hardware decompressor supporting zlib + Kraken (Sony dev-doc-derived info) — [encode.su thread](https://encode.su/threads/3364-sony-PS5-comes-with-an-inbuild-quot-Kraken-compression-quot); unit's nature/ratio explained by ex-Frostbite engineer — [gamingbolt](https://gamingbolt.com/former-frostbite-software-engineer-explains-ps5s-kraken-compression-unit).
- Scene report (video, unverified): a PS5 game-backup loading method routes **Kraken decompression through the A53 coprocessors** — [YouTube scene update](https://www.youtube.com/watch?v=xb5ujsKBet4). This is the only concrete public claim of *someone* (homebrew side) using hardware decompression outside PS5-OS games.
- Homebrew ecosystem has a compression tool for MKPFS/game packing (experimental) — [PS5-Game-Compressor](https://github.com/juma-sayeh/PS5-Game-Compressor), [r/PS5_Jailbreak MKPFS thread](https://www.reddit.com/r/PS5_Jailbreak/comments/1tyvuct/ps5_mkpfs_compression/) — but nothing exposing the unpacking unit to payloads.
- ps5-linux/pLinux5 exposes no documented interface to the unpacker; its storage stack is USB block devices under a hypervisor (see Q2 sources).
- The assignment's premise (no Linux driver) is consistent with everything found.

### Inferences
- The A53 reports, if real, imply the decompressor is reachable from *some* PS5-side context — a future payload could plausibly mediate read+decompress requests to it, but that's homebrew-folklore-grade evidence; nothing is shippable.
- Kraken-encoded experts today = decode on Zen2 with a software Kraken decoder (ozip-style, encumbered by licensing questions for Oodle) at ~1–2× LZ4-class speed — strictly worse than plain zstd/LZ4 unless ratio (>1.3–1.5:1 on quant weights) is decisive. Estimate: not worth it; use zstd.

### Gaps
- The ozip repository (SonicMastr's reverse-engineered Kraken/Mermaid decoder) did not appear directly in this session's searches — treat "ozip exists" as community knowledge needing verification before quoting in the report.
- No ps5dev.org threads found via search about driving the unpack unit from a payload — either none exist or the site is poorly indexed.

---

## Q5. Cache management: page cache vs pinned user-space caches, mlock, fadvise

### Takeaway
The strongest published result in this space is counter-intuitive: **trust the OS page cache** — flash-moe found every custom cache slower (deleting their LRU gave +38%; page cache hit ~71% for free), while the win comes from *pinning the right experts*: text-calibrated pinning raises expert hit rate from 6% (chance) to 42–56%, and mincore+explicit-prefetch beats fault-driven loading by +13–14%.

### Cited Findings
- flash-moe: "Trust the OS" principle; OS page cache (~35 GB) achieves ~71% hit; **Metal LRU deleted → +38% tok/s**; malloc cache and LZ4 cache slower; F_NOCACHE helped 2-bit only (+3%); prefetch −0% — [flash-moe README](https://github.com/danveloper/flash-moe).
- tinygiant: random-profile pinning = 6% hit (≈chance); **text-calibrated pinning (from 10 real tokens) = 42% hit, +LRU = 56%**; pipelined reads hide I/O at ≥88% hit (4.10 tok/s at 88% vs 3.47 sequential) — [llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149) and [tinygiant](https://github.com/jerryjokesalot/tinygiant).
- mbolt: explicit pread of non-resident expert slices (mincore-checked, sorted/merged ranges) into page cache before matmuls fault: **+13–14% end-to-end** vs fault streaming on both stock and relaid-out files — [mbolt](https://github.com/doramirdor/mbolt), reported in [llama.cpp discussion #18758](https://github.com/ggml-org/llama.cpp/discussions/18758).
- llama.cpp `--mlock` exists for pinning; users report mmap+no-dio (page cache) beats dio 10× on reloads (Q1); aggressive full-model prefetch causes thrash and was patched off — [llama.cpp discussion #18758](https://github.com/ggml-org/llama.cpp/discussions/18758).
- cgroup v2 MemoryMax as a hard residency ceiling: MemVanta (mmap GGUF + bounded cache) survives a 3584 MiB cgroup where pinned llama.cpp OOM-dies at 3840 MiB (47.5% less RSS) but is much slower — [MemVanta, in #18758 comments](https://github.com/sauravsingla/MemVanta).
- imatrix-guided initial expert pinning ("loads fast, gets faster with use") — [Guanaco, in #27149 comments](https://github.com/Atomic-Germ/Guanaco); llama.cpp issue [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) proposes SLRU GPU+RAM two-tier expert cache; pure CPU offload without cache = 0.5–1 tok/s on that hardware.
- OS page cache as accidental expert streaming is "doing expert streaming badly" but free — [#27149 body](https://github.com/ggml-org/llama.cpp/discussions/27149).

### Inferences
- PS5 (16 GB unified, engine budget maybe 6–8 GB to page cache after weights+KV): the plan is (a) let the page cache hold LRU experts (no user-space LRU), (b) add text/imatrix-calibrated **mlock of the top-frequency experts** to resist pressure — tinygiant's data says calibration is where the 6→42% jump lives; (c) never full-model prefetch (thrash, confirmed in the wild); (d) readahead: for 2–6 MB contiguous expert reads, set per-file readahead small (the read is already large; kernel default 128 KB readahead adds ~nothing but wastes bandwidth on K-parallel scattered reads — set `posix_fadvise(READNOW)` only for the *predicted* next-token experts, if prediction is kept, per mbolt's negative result on predictors).
- On PS5 the GPU also uses the same 16 GB — user-space pinned buffers must be allocated *outside* the Vulkan-visible pool or they steal GPU memory; mlock of files is the cheapest form of "pinning" because evictions then can't reclaim from it, which is exactly what hot experts need.

### Gaps
- No measured posix_fadvise(READSEQL/NOWIN/READAHEAD) sweep for MoE expert streaming anywhere published.
- No measured comparison of mlock-pinned expert pools vs page-cache on **Linux x86 UMA** specifically (flash-moe's is Apple; #27149's is Apple CPU-only).

---

## Q6. Expert-file layout, chunk size, filesystem

### Takeaway
Co-activation-aware layout is the biggest measured pure-I/O lever found: mbolt's profile-guided packing cut reads/token 1418→775 (co-activation order) and →~370 (interleaved up|gate|down per expert), = **2.23× faster cold-decode I/O**; tinygiant's expert-contiguous relayout cut mmap page-fault amplification **36×** (6912→192 pages/expert). GGUF's stacked 3-D expert tensors already make each expert a contiguous byte slice — a seek+read of ~2–6 MB with no format change. Filesystem choice (f2fs vs ext4) had no published MoE-relevant measurement; chunk size is not a lever at 2–6 MB (NVMe bandwidth saturates ≥128 KB reads).

### Cited Findings
- mbolt measured: co-activation-ordered file → 1418→775 reads/token; interleaved up|gate|down per expert → ~370; **2.23× cold I/O**; but on fast Apple NVMe the I/O win was "eaten by compute (42 ms/token)" — pays on 1–2 GB/s drives — [mbolt via #18758](https://github.com/ggml-org/llama.cpp/discussions/18758).
- tinygiant: GGUF stacked experts interleaved in one tensor ⇒ 1 expert via mmap touches 6,912 pages; contiguous relayout ⇒ 192 pages (36× fewer faults); "SSD: 3.0+ GB/s from contiguous layout" — [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149), [tinygiant relayout tool](https://github.com/jerryjokesalot/tinygiant/blob/main/tools/nws_expert_relayout.py).
- GGUF needs **no format change** for selective expert reads: `blk.N.ffn_gate_exps.weight` is stacked 3-D, each expert = contiguous slice; Qwen3-30B-A3B per-layer: all-experts 345 MB → 8-active 30 MB (11.4× I/O cut) — [#27149 body](https://github.com/ggml-org/llama.cpp/discussions/27149).
- flash-moe counterpoint: "Expert file clustering 0% — NVMe ignores scatter at 7 MB granularity" (on M3 Max's 17.5 GB/s SSD with only ~4×6.75 MB reads/token) — [flash-moe README](https://github.com/danveloper/flash-moe). Effect size of layout scales inversely with SSD speed and per-token read count.
- NVMe block-size behavior: optimal random-read size 4 KB per ClickHouse, but large sequential-ish streams (tinygiant 286 MB layer chunks → 2.97 GB/s) reach full bandwidth — [ClickHouse](https://clickhouse.com/blog/a-journey-to-io_uring-aio-and-modern-storage-devices); PS5-OS itself batches 256 KB–2 MB reads to hit 5.5 GB/s (Mark Cerny's presentation — widely reported, see [gamingbolt](https://gamingbolt.com/former-frostbite-software-engineer-explains-ps5s-kraken-compression-unit) context).
- Unrelated-but-adjacent engine datapoint: "Strata" reportedly runs Qwen3.8-Flash-Next q2 (model > 32 GB RAM) at 200 tok/s prefill / 20 tok/s decode on 32 GB RAM + 6 GB VRAM — expert-streaming works well on x86 with dGPU — [Dampfinchen in #27149](https://github.com/ggml-org/llama.cpp/discussions/27149).

### Inferences
- PS5 reads 1.2–2.3 GB/token (10 experts × K-parallel 2–6 MB): we are firmly in the regime where mbolt's layout math pays (many large scattered reads per token on a 5.5 GB/s device); co-activation-aware packing + per-expert-interleaved gate/up/down should be implemented in this project's packer (profile by capturing routing on representative prompts, then reorder expert file blocks to maximize expected contiguity of each token's 10-expert set — turning ~10 reads into ~5–6 merged ranges).
- 2–6 MB chunks are above the size-sensitivity knee; splitting experts into 512 KB pieces buys nothing, but *co-locating* predicted pairs/triples into one contiguous "super-expert" region (so K reads merge) is the same lever with a bigger denominator, consistent with mbolt's 775→370 reads/token trajectory.
- f2fs vs ext4: choose ext4 (pLinux5 default, no evidence of MoE-specific difference — keep, don't re-litigate).

### Gaps
- No published f2fs-vs-ext4 large-file random-read benchmark found.
- No published defrag/extent-locality (e4defrag) study for MoE streaming; flash-moe's 0% suggests it's negligible on a freshly-written drive.

---

## Q7. x86-UMA revisit of the Mac negative results (NVMe DMA vs GPU bandwidth contention)

### Takeaway
The −73%/prefetch-useless result is real but *Apple-specific in mechanism*: flash-moe's GPU dequant saturates ~418 GiB/s of the M3 Max's ~400 GB/s unified fabric, leaving no headroom for concurrent SSD DMA; on PS5 the same arithmetic is even tighter (GDDR6 448 GB/s theoretical), so **serializing GPU-compute with SSD-stream** (the flash-moe "hardware-optimal serial pipeline") is the right default on PS5 — with the caveat that overlap demonstrably *works* when compute is CPU-side or bandwidth-light (tinygiant CPU: +2.09× pipelining; Strata on x86 dGPU: 20 tok/s streaming), so overlap should be adaptive, not forbidden.

### Cited Findings
- flash-moe (M3 Max, 48 GB unified, ~400 GB/s): "SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped. GPU dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes through memory controller arbitration" → F_RADVISE prefetch net 0% ("SSD DMA slows GPU −73%"); serial GPU→SSD→GPU pipeline declared hardware-optimal — [flash-moe README](https://github.com/danveloper/flash-moe).
- Counterpoint on the same architecture when compute is **CPU-side**: tinygiant M1 CPU-only pipelined I/O beats sequential at every cache-hit level (2.92→3.81 tok/s @75% hit; 3.47→4.10 @88%) — [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149): contention, not overlap, is the enemy; overlap pays when the consumer (CPU or unsaturated GPU) leaves fabric headroom.
- Predictive prefetch is independently negative on Mac (temporal −18%, MLP 31% acc, speculative early routing −38% — [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149) / [flash-moe](https://github.com/danveloper/flash-moe)) and mbolt ("predictive prefetch did NOT work", [mbolt](https://github.com/doramirdor/mbolt)).
- x86 UMA analog (Strix Halo, 256-bit LPDDR5X ≈ 215–256 GB/s real-world): iGPU inference fully bandwidth-bound, fine-grained CPU/GPU buffer sharing — [Framework community](https://community.frame.work/t/ryzen-ai-max-not-so-much/77999), [Chips&Cheese](https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling); host–GPU data movement named the latency-critical bottleneck in LLM serving generally — [arXiv 2512.16056](https://arxiv.org/html/2512.16056v2).
- x86 discrete-RAM+GPU streaming works at speed: Strata 20 tok/s Qwen3.8-Flash-Next q2 on 32 GB DDR5+6 GB VRAM ([#27149 comment](https://github.com/ggml-org/llama.cpp/discussions/27149)); HOBBIT mixed-precision expert offload on llama.cpp across edge devices — [arXiv 2411.01433](https://arxiv.org/html/2411.01433v2); ktransformers-style CPU-offload guides — [HF blog](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide).
- No direct measurement found of NVMe-DMA-vs-iGPU arbitration on AMD APUs (gap below).

### Inferences
- PS5 is a Strix-Halo-like UMA with *less* bandwidth (448 theoretical, likely ~350–400 real) and 16 GB total: a Vulkan-dequant expert matvec will likely saturate a large share of fabric bandwidth, so background 5 GB/s SSD DMA plausibly costs single-digit-to-double-digit % of GPU throughput via both fabric share (~1–1.5% of 448 GB/s in bytes, but arbitration/row-activation effects historically cost 3–10× the byte-share on AMD iGPUs — Strix Halo's ~215/256 ≈ 84% efficiency hints at how much mixed traffic costs — [Framework](https://community.frame.work/t/ryzen-ai-max-not-so-much/77999)). Expect the truth between flash-moe's −73% (worst case, fully-saturated dequant) and ~0: **measure on-device**.
- Engineering answer: implement the serial pipeline as baseline, with an env-gated overlap mode (I/O for layer N+1 under layer N compute) and a cheap saturation detector (I/O-complete timestamp jitter), defaulting per measurement. If compression-on-read (Q3) is used, decode overlaps with GPU naturally (CPU cores, no GPU-fabric share).

### Gaps
- Zero direct measurements of background-SSD-DMA impact on iGPU inference on AMD APUs (890M/Strix Halo) found — only the general UMA-bandwidth literature.
- No zenodo/PS5-specific UMA APU inference methodology paper found.

---

## Q8. Exotic: ZNS, pmem, SPDK, CXL, USB

### Takeaway
The only exotic lever with real evidence is SPDK-class userspace NVMe (up to ~10× lower latency at low depth, but not uniformly faster than libaio and needs exclusive device access — hard on PS5's hypervisor-owned hardware); CXL is physically impossible on PS5 (needs PCIe 5.0+ PHY and memory expanders); USB gives at most ~1 GB/s (10 Gbps ports, ~850–1000 MB/s real) — 5–6× below internal NVMe, so USB/USB4-enclosure storage is a capacity fallback, not a bandwidth lever. Nothing found for ZNS-LLM or pmem tricks in this context.

### Cited Findings
- SPDK: Intel's own figures claim up to **10× lower latency** vs the kernel NVMe driver (userspace polling); SPDK issue #661 documents cases where SPDK is *slower* than libaio at low queue depth; SPDK requires binding the device away from the kernel (VFIO/UIO) — [Intel](https://www.intel.cn/content/www/cn/zh/developer/articles/technical/accelerating-your-nvme-drives-with-spdk.html), [spdk#661](https://github.com/spdk/spdk/issues/661); io_uring-SPDK hybrid papers show kernel-bypass gains concentrate in IOPS/latency, not bulk bandwidth — [IJRASET](https://www.ijraset.com/best-journal/evaluating-the-performance-of-spdkbased-io-uring-and-aio-block-device).
- PS5 USB: 10 Gbps (USB 3.2 Gen 2) ports; real external-SSD throughput ~850–1000 MB/s (Seagate Game Drive: 871 MB/s 2MB writes / 850 MB/s reads; DF testing: PS5 has 10 Gbps vs Xbox's 5) — [StorageReview](https://www.storagereview.com/review/seagate-game-drive-ps5-ssd-review), [DF via ResetEra](https://www.resetera.com/threads/df-ps5-external-ssd-testing-sata-vs-nvme-vs-hard-drive-back-compat-titles-read-write-speed-tests.411057/), [ADATA guide](https://www.adata.com/us/quikTips/unlock-more-space-ultimate-external-ssd-guide-for-ps5/). PS5 has no USB4/TBT.
- pLinux5 runs from USB SSD today — so the 0.85–1 GB/s USB path is the *floor* configuration, and a model streamed from the boot drive would cap around ~0.5–0.7 tok/s at 1.2–2.3 GB/token ([arithmetic on](https://www.storagereview.com/review/seagate-game-drive-ps5-ssd-review) those figures).
- ZNS for LLM streaming, pmem/DAX tricks: no relevant published work found.

### Inferences
- SPDK on PS5: blocked in practice — VFIO binding the internal NVMe under a hypervisor that still owns the platform is unlikely to work, and polling-mode SPDK burns cores that Zen2 needs for decode; io_uring + O_DIRECT captures most of the user-visible benefit in-kernel. Maturity: research-only for PS5.
- CXL: n/a confirmed by physics of the platform (PCIe Gen4 APU, no CXL PHY or expanders) — no search source needed beyond PS5 hardware being PCIe 4.0 class ([Lexar upgrade guide mentions Gen4/Gen5 drive support but PS5 APU link is Gen4](https://americas.lexar.com/ps5-ssd-upgrade-guide/)); flag as impossible, not "unpromising".
- The realistic exotic win list for PS5, in order: two-device app-level striping (Q2) > expert-co-activation layout (Q6) > calibrated mlock pinning (Q5) > slice-read prefetch (Q1/mbolt) > optional zstd compression-on-read (Q3). USB and SPDK are non-levers.

### Gaps
- No ZNS/zoned or pmem/DAX measurements for MoE weight streaming found.
- No prior art of a USB4/external-SSD PS5-for-Linux streaming benchmark found (the question is moot given ~1 GB/s ceiling).

---

## Cross-cutting priority list for the report writer (all sourced above)
1. Serial GPU↔SSD pipeline default, adaptive overlap behind a flag (Q7: −73% Apple vs +2.09× CPU-side evidence).
2. Explicit `pread`-of-slices streaming, no mmap for experts (Q1: −5× mmap, +13–14% slice reads).
3. Co-activation-aware expert packing + interleaved gate/up/down (Q6: 2.23× cold I/O).
4. Page-cache-as-LRU + calibrated mlock pinning of hot experts (Q5: 6%→56% hit).
5. Router-primed READNOW prefetch only if Q7 measurement shows fabric headroom (Q1/mbolt predictors: negative).
6. Prototype io_uring + registered fixed buffers (+ O_DIRECT cold path); expect ≤1.3× (Q1: 2× only at high-depth libaio comparisons).
7. Verify internal-NVMe enumeration under pLinux5, then two-device app-level striping (Q2: 10–12 GB/s theoretical, md capped ~15 GB/s — use app striping).
8. Optional zstd/LZ4-on-2–4-cores compressed stream when capacity-bound (Q3: −13% warning, decode ≫ disk).
9. Kraken: software-decode-only; prefer zstd (Q4: no driver; A53 rumor).
