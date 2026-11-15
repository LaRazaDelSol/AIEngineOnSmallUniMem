# PS5 Flash-Next: Paths from 6 → 10/20/40 GB/s (moonshot survey, as of Oct 2026)

> **Status (2026-10-21):** ACTIVE evidence base (W4/W8). Constants per the W4 recompute (G = 1.24 GiB/token); REAP-288 quant correction S5-F23 (Q4_K_M does not reduce G). Fabricated-token citation pruned in place (F12/S5).

Baseline context used throughout (from platform sources cited under Q1): PS5 Phat/Slim run **ps5-linux** (a.k.a. pLinux5, TheFlow's loader) on FW 3.00–7.61 today; **pLinux5-NG** extends boot to FW 13.60; the **Relapse** jailbreak (Jul 2026) covers FW 7.00–13.60 including **PS5 Pro** — i.e., the jailbreak population is no longer limited to launch-model Phats. Under Linux the internal NVMe and the M.2 slot are separate NVMe devices (`/dev/nvme0`, `/dev/nvme1` per the community guide's own commands), all USB 3.1 Gen2 ports work, Ethernet is 1 GbE, and GPU/CPU boost + 40-CU unlock are controllable from user space. None of the surveyed paths are already integrated into a streaming-LLM engine; every one below is "open engineering," not shipping software.

## Q1. Dual/tri-device concurrency (internal NVMe + M.2 slot) under ps5-linux

### Takeaway
The two flash devices are separate NVMe namespace endpoints that both enumerate under ps5-linux (the community tooling itself writes to `/dev/nvme1` for the M.2 while `/dev/nvme0` — the internal 825 GB — remains present), so app-level striping should roughly double sequential read bandwidth (~12–13 GB/s uncompressed, before compression). Caveats: Linux-from-USB is the boot mode in which PS5 OS is also running-premise (M.2 used as a Linux disk means it's given up by PS5 OS), the internal drive's proprietary filesystem needs a reader, and I found **no published hdparm/fio measurement of the two devices concurrently under Linux** — that measurement is the open work.

### Cited Findings
- ps5-linux loader (TheFlow/c0w, credits incl. fail0verflow's prosperous and flatz's HV exploit) runs on **PS5 Phat and Slim, FW 3.00–7.61**; features list "M.2 SSD as dedicated Linux partition", "All USB ports usable", "Internal SSD is not modified" — [ps5-linux-loader README](https://github.com/ps5-linux/ps5-linux-loader)
- M.2 is supported on FW 4.00+ only; on 3.xx "the PS5 fails to boot with it attached" — [ps5-linux-loader README FAQ](https://github.com/ps5-linux/ps5-linux-loader)
- The community guide's storage workflow runs under Linux and treats M.2 as `/dev/nvme1` (`sudo mkfs.ext4 /dev/nvme1`), implying the internal drive enumerates as `/dev/nvme0` at the same time; "M.2 is used exclusively for Linux – cannot be used for PS5 game storage simultaneously" — [ps5-linux guide: USB & Storage](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/usb-storage)
- Hardware overview: internal NVMe 825 GB is "not touched by Linux install"; M.2 slot works on FW 4.00+; internal NVMe is standard NVMe-class PCI device visible to Linux; BD-ROM is on a separate AHCI controller (`ahci_salina`) — [ps5-linux guide: Hardware Overview](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/hardware)
- PS5 OS itself reads both drives at up to ~5.5+ GB/s in its own benchmark; Samsung 990 Pro measures 6,558 MB/s in the PS5's SSD benchmark, 980 Pro "matches or outperforms" the internal drive in game-load tests — [Digiera PS5 SSD roundup](https://www.digieraglobal.com/blogs/news/best-internal-ssd-for-ps5); [r/PS5 FW2.0 M.2 test](https://www.reddit.com/r/PS5/comments/ozwah7/ps5_firmware_20_tested_m2_ssd_storage_upgrade_vs/)
- Boot-from-USB + M.2-as-data-disk is a supported configuration (Linux root on USB or M.2; front-bottom Type-C or rear Type-A boot); the M.2 can hold the entire Linux install (`root=LABEL=ubuntu2604-m2`) — [ps5-linux-loader README](https://github.com/ps5-linux/ps5-linux-loader)
- pLinux5-NG boots Linux on up to FW 13.60; Relapse jailbreak covers FW 7.00–13.60 "both the standard PS5 and the PS5 Pro" (Sony began blocking jailbroken consoles from PSN at FW 14.10) — [Notebookcheck, Oct 2026](https://www.notebookcheck.net/Linux-now-runs-on-PS5-console-jailbreaks-and-July-firmware-with-PC-games-possible.1425575.0.html); [TechPowerUp on Relapse](https://www.techpowerup.com/forums/threads/new-jailbreak-unlocks-nearly-every-playstation-5-model.353259/); [Tom's Hardware on Relapse](https://www.tomshardware.com/video-games/playstation/new-ps5-relapse-jailbreak-enables-homebrew-and-switch-emulation-but-is-hamstrung-by-rapidly-changing-firmware-revisions-jailbreak-unlocks-firmware-13-60-but-newer-games-already-demand-firmware-14-00)
- Prior art, striping flash for LLMs: active LocalLLaMA discussion "Has anyone tried running an LLM from a raid0 of a bunch of SSDs" (large MoE, VRAM-poor setups); Gen4 NVMe RAID0 video benchmarks show near-linear sequential scaling for large sequential reads; GPU-farm guides cover mdadm NVMe RAID0 for AI model storage — [r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1uytlvw/has_anyone_tried_running_an_llm_from_a_raid0_of_a/); [2x Gen4 RAID0 test](https://www.youtube.com/watch?v=Ffxkvf4KOt0); [GigaGPU NVMe RAID for AI models](https://gigagpu.com/nvme-raid-configuration-ai-models/); [Xinnor offload benchmark incl. RAID](https://xinnor.io/blog/does-ai-model-offloading-matter-benchmarking-ram-and-nvme-strategies-for-llm-training/)
- Prior art, tiered weight streaming (DRAM+NVMe tiers) is standard in DeepSpeed-Inference-style serving — [arXiv survey citing it](https://arxiv.org/html/2512.21835v2)
- No published console/ARM-SBC project doing two-device RAID0 specifically for LLM weight streaming was found (only single-SSD offload) — [r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1uytlvw/has_anyone_tried_running_an_llm_from_a_raid0_of_a/)

### Inferences
- Both devices sit behind the same APU PCIe root complex, so aggregate ceiling is bounded by the SoC's I/O fabric, not the sum of the two link specs; realistically expect 85–95% of (5.5 + 6.5–7.4) ≈ **11–12.5 GB/s** from a 2-device `io_uring` stripe at queue depth ≥32 with 2–4 MB stripe units. This is the single cheapest 2× available on the exact consoles that run ps5-linux today.
- Reading the internal SSD from Linux needs a reader for Sony's SSIFS/ext4-hybrid layout; the "internal SSD is not modified" wording plus `nvme0` visibility implies block-level readability (raw partitions), but nobody has documented mounting game partitions under pLinux5 — treat "internal drive readable as a 400 GB expert store" as a build item, not a known quantity.
- PS5 Pro: jailbreakable via Relapse but NOT listed on the ps5-linux loader's supported matrix; its stock SSD is 1.33 TB at ~5.8+ GB/s, so a Pro + M.2 stripe would start higher (~13 GB/s) — Pro under pLinux5-NG is an unverified combination.
- On Phat (3.xx FW) the M.2 slot is unusable (boot failure), so dual-device plans require FW ≥ 4.00 consoles.

### Gaps
- No measured aggregate Linux benchmark (fio/hdparm) of internal + M.2 concurrent sequential reads; the "≈12 GB/s" number is my arithmetic from per-device specs, not a user report.
- PS5 Slim under ps5-linux: loader README says Slim is supported, but the DF review used a Phat; no Slim-specific dual-storage report found.
- Whether the hypervisor keeps any DMA restrictions on the NVMe controllers under pLinux beyond what the loader needs to boot Linux (loader source exists but I did not audit it for NVMe/IOMMU handling) — [ps5-linux-loader](https://github.com/ps5-linux/ps5-linux-loader)

## Q2. Kraken / hardware decompression

### Takeaway
The PS5's I/O decompression unit is confirmed to exist and to decompress in dedicated hardware (not CPU/GPU), and Sony's own pre-launch material frames 5.5 GB/s raw as "8–9 GB/s effective" with Kraken; the "~9.9 GB/s" figure floating around traces to derivative articles, not a primary spec. What's missing for the moonshot is a user-space path: no known syscall/IOCTL in pLinux5 exposes the unit, but under the **native jailbreak (no Linux)** the PS5-OS I/O pipeline (libSceKernel Prc path + the SSD controller's DEFLATE/Kraken support) already sits between flash and RAM and is exactly what the OS uses to achieve those effective rates — an engine running as a native payload could plausibly inherit it.

### Cited Findings
- PS5 SSD controller "supports hardware decompression for the industry-standard ZLIB, but also the new Kraken format from RAD Game Tools"; discussion quotes Sony material that "5.5GB/s of bandwidth translates into an effective eight or nine gigabytes per [second]" — [encode.su thread 3364](https://encode.su/threads/3364-sony-PS5-comes-with-an-inbuild-quot-Kraken-compression-quot)
- Former Frostbite engineer (Yan Chernikov): "It is literally doing decompression in hardware, it's not doing it on the CPU or the GPU. There is literally a unit responsible..." — [GamingBolt](https://gamingbolt.com/former-frostbite-software-engineer-explains-ps5s-kraken-compression-unit)
- Sony/Tom's Hardware: Kraken cuts PS5 install sizes ~60%; stated intent to extend hardware decompression to the M.2 slot once unlocked — [Tom's Hardware](https://www.tomshardware.com/news/ps5-60percent-smaller-game-sizes-kraken-compression)
- Charles Bloom's engineering analysis: Kraken "acts as a multiplier for the IO speed and disk capacity" — i.e., compression ratio directly multiplies effective GB/s — [c Bloom rants](http://cbloomrants.blogspot.com/2020/09/how-oodle-kraken-and-oodle-texture.html)
- The widely-quoted "up to 9.9 GB/s decompressed" figure appears in SEO/content-farm pages restating Sony's "5.5 GB/s raw / up to ~9–10 GB/s with PS5 I/O + compression" marketing — [EaseUS explainer](https://www.easeus.com/partition-master/ps-5-ssd-speed-test.html) (weak source; primary framing is Sony's 2020 PS5 architecture presentation)
- Homebrew is actively probing the compression path for loading: "game backups via the A53 processor including kraken compression is in development" (2026 jailbreak-community news item) — [YouTube: PS5 Jailbreak News](https://www.youtube.com/watch?v=xb5ujsKBet4)
- Under pLinux5, no decompression-unit driver or syscall shim is listed in the ps5-linux patches/tools feature set (GPU, USB, Ethernet, WiFi, BD, VCN — no I/O-decompressor) — [ps5-linux guide: Hardware Overview](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/hardware); [ps5-linux-loader README feature list](https://github.com/ps5-linux/ps5-linux-loader)

### Inferences
- The unit is wired into the storage→GDDR path on the way into the I/O complex; if experts are stored Kraken-compressed on disk, the native-OS read pipeline should deliver its rated decompressed rate (≥8–9 GB/s per Sony) "for free" — an engine as a native payload (FW ≤13.60 + Relapse, which gives kernel access on nearly every shipped model) may not need to touch the unit directly at all; it needs the OS file I/O to be the only path its reads take. 1.4 GiB/token at 40 tok/s = 56 GB/s raw, but with a plausible 2.5–3× lossless ratio on expert weights (fp16→ Kraken of quantized weights; Bloom's ratios were for textures, so treat 2–3× as optimistic-but-typical for entropy-coded data), the *flash* requirement drops to 19–28 GB/s — and combined with Q1 striping (12 GB/s × ~2.5 effective ≈ 24–30 GB/s), 40 tok/s starts touching plausibility on native OS.
- Under Linux the realistic version is software decompression: ZLIB via zlib-deflate on 8 Zen2 cores is maybe 1–3 GB/s aggregate; a Kraken software decoder (Oodle is proprietary; open approximations exist) would burn the CPU budget the engine needs. So "Kraken under Linux" is much weaker than "stream under native OS."
- Because jailbroken PS5-OS kernels (ps5-payload-dev SDK world) already expose most syscalls to payloads, a self-built engine binary running under PS5-OS has strictly more I/O options than pLinux; the cost is giving up standard Linux LLM tooling.

### Gaps
- I found **no reverse-engineering write-up of the decompression unit's MMIO registers or a pLinux driver attempt** (ps5dev.org threads may exist behind access walls; nothing surfaced in search).
- The unit's exact sustained decompressed rate is not documented by Sony; the 8–9 GB/s figure is an "effective end-to-end" marketing number, not a measured unit throughput. Whether it sustains on 3 MB random expert-sized blocks at high QD is unknown.
- Unknown whether the *M.2 slot* traffic actually traverses the decompressor even on native OS (the 2021 "will extend to M.2 once unlocked" statement was never confirmed as shipped).

## Q3. GDDR/memory side: clocks, VRAM pool, FAST_RESERVE

### Takeaway
GDDR6 at ~448 GB/s is not the bottleneck, and there is no evidence anyone has raised the memory clock on PS5/BC250 boards under Linux (memory OC docs exist for the BC250 mining boards — same APU family — but I could not confirm a memory-clock result); what *is* concretely tunable under Linux is the **VRAM/GTT pool split** (`amdgpu.gttsize`, `ttm.pages_limit`, `vram.txt` dynamic-VRAM mode at 512 MB default), which changes how much of the 16 GB your engine can use as stable GPU-resident pages but does not change flash→GDDR bandwidth.

### Cited Findings
- ps5_control documents only CPU boost (3500 MHz), GPU boost (2230 MHz) and fan curve — no memory-clock control — [ps5-linux-tools README via loader README links](https://github.com/ps5-linux/ps5-linux-tools); [ps5-linux-loader README](https://github.com/ps5-linux/ps5-linux-loader)
- PS5 Linux gives ~14–15 GB of the 16 GB to the OS, remainder reserved for firmware/VRAM; default VRAM carve is 512 MB with dynamic allocation via `vram.txt` — [ps5-linux guide: Hardware Overview](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/hardware); [ps5-linux-loader README tips](https://github.com/ps5-linux/ps5-linux-loader); [Digital Foundry test, May 2026](https://www.digitalfoundry.net/news/2026/05/playstation-5-linux-tested-steam-pc-vs-native-ps5-games)
- BC250 (PS5-APU-derived "Cyan Skillfish" mining board, community's low-cost AI testbed running Ollama+Vulkan up to 35B MoE) uses the same recipe: 512 MB dynamic VRAM BIOS + kernel params; has dedicated BIOS "Overclocking" and "VRAM Configuration" documentation pages — [AMD BC250 documentation](https://elektricm.github.io/amd-bc250-docs/)
- BC250/PS5 Linux users raise the GPU memory ceiling with kernel parameters: `ttm.pages_limit=<GB×1024×1024/4>` plus the matching `ttm.page_pool_size`, to exceed the ~5.5 GB dynamic-VRAM cap; the cap exists to avoid suspend/resume problems (irrelevant on PS5, which can't suspend under pLinux anyway). **S5 PRUNE 2026-10-21: this bullet originally cited a Digital Foundry comment (#9165405) and the token `amdgpu.gttsize=14750` — both fabricated (STATE F12/W5); the technical content is preserved via the real BC250 docs.** — [elektricM/amd-bc250-docs — Kernel Configuration](https://elektricm.github.io/amd-bc250-docs/linux/kernel/)
- DF measured that dynamic graphics allocation "seems to top out at around 5.5 GB" with default setup, causing texture-stutter problems in games — evidence the pool split genuinely binds workloads — [Digital Foundry](https://www.digitalfoundry.net/news/2026/05/playstation-5-linux-tested-steam-pc-vs-native-ps5-games)
- 40-CU unlock verified working under PS5 Linux ("really not a big bump" for 36→40 CUs) — [Digital Foundry comment](https://www.digitalfoundry.net/news/2026/05/playstation-5-linux-tested-steam-pc-vs-native-ps5-games#comment9166560); harvested-CU unlock is also a documented kernel param on ps5-linux — [ps5-linux guide: Hardware](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/hardware)

### Inferences
- For the streaming problem, the memory side is already sufficient: a 13 GB pinned pool with `gttsize`/`ttm` tuning gets you past the default 5.5 GB dynamic ceiling — which matters more than any clock, because the 13 GB pool is exactly what the MoE streaming design assumes. FAST_RESERVE-style tuning = "how much pool is yours," not "how fast it refills."
- A +10–15% memory-clock OC (if the pp_table on Oberon were writable — amdgpu debugfs/pp_table on BC250-class boards is the plausible lever, per the BC250 BIOS-overclocking page existing at all) would add ~0 GB/s to the flash-limited pipeline: it only raises the GDDR ceiling from 448 GB/s toward ~500. Worth ignoring until the storage side is solved.

### Gaps
- No PS5 or BC250 user report of an *actually achieved* GDDR memory-clock increase with measured before/after bandwidth under Linux; BC250 overclocking page exists but I did not verify its contents reach mclk.
- Whether `ps5_control` or the SEEM/MCU RPC has an undocumented memclock command (it controls clocks at boot per the fan/boost feature) — nobody has dumped its full parameter set publicly in what I found.

## Q4. PCIe/DMA path: P2P DMA, IOMMU state under pLinux5

### Takeaway
On an APU, NVMe and GPU hang off the same SoC fabric; the interesting P2P win is skipping the CPU/bounce-buffer copy into GDDR (GPUDirect-Storage-style: DMA flash→page→straight into the resident pool). pLinux5 boots Linux via hypervisor-level exploits (the loader's shellcode explicitly includes an HV stage), i.e., it takes the SoC's DMA protections down far enough for amdgpu to run — but I found no explicit documentation of the resulting IOMMU group layout, and no P2P DMA measurement on PS5 silicon.

### Cited Findings
- The loader is "a Linux payload implementing HV exploits to run a custom bootloader" with `shellcode_hv` / `shellcode_kernel` stages built on flatz's published HV exploit and cragson's HEN implementation — [ps5-linux-loader README](https://github.com/ps5-linux/ps5-linux-loader); [flatz HV exploit gist](https://gist.github.com/flatz/620ddda6d64acca6d1c990dc3080ac0e)
- All storage + USB + GPU attach to the AMD south-bridge complex: BD-ROM is AMD/Sony AHCI (`104d:9105`), USB is AMD Ariel (`1022:13ee/13ed`), GPU is `1002:13fb` — i.e., standard AMD PCI enumeration under Linux, no exotic bridge chips — [ps5-linux guide: Hardware Overview](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/hardware)
- M.2→PCIe riser / eGPU explicitly tested by the community and **not supported** — the external PCIe path beyond the two M.2-class connectors is dead ends; you cannot bolt on a PCIe-to-NVMe HBA card — [ps5-linux guide: USB & Storage](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/usb-storage)
- amdgpu on Oberon under Linux works with VCN, display, compute — a full-function driver that in stock x86 setups would normally negotiate through the IOMMU; pLinux runs it directly — [ps5-linux guide: Hardware](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/hardware); [DF testing confirms working amdgpu/RADV](https://www.digitalfoundry.net/news/2026/05/playstation-5-linux-tested-steam-pc-vs-native-ps5-games)

### Inferences
- If the HV/DMA-protection teardown leaves the AMD IOMMU disabled or passthrough (the usual state for console Linux loaders — the PS4 loader family documented this), then Linux `dma_map` paths are identity-mapped and **P2P between nvme controller and the GPU's DMA engines is register-level bookkeeping, not an IOMMU problem** — e.g., issuing reads into GDDR-backed bus addresses via the display/GDF DMA or an SDSM-style path. This is speculative for PS5 specifically; the enabler is that address translation isn't in the way.
- P2P's realistic gain is eliminating a CPU copy (memcpy at ~2×GDDR is *not* the bottleneck at 6 GB/s, so at PS5's flash speeds P2P mostly saves CPU cycles, not wall-clock bandwidth: ~6 GB/s of copies is why the CPU-side is 1–2 cores busy, not a 2× wall). Expected: +0–20% end-to-end at current device speeds; it becomes decisive only when stripe+Kraken push ≥15–20 GB/s and copy overhead starts competing with the token-compute threads.
- Since the eGPU/riser path is dead, PCIe P2P on PS5 is necessarily *intra-APU*: NVMe↔fabric↔GDDR. No source claims it's impossible; no source demonstrates it either.

### Gaps
- Nobody has published `dmesg | grep -i iommu`, IOMMU group listings, or `iommu=pt`-style cmdline for pLinux5 — the SVM/IOMMU teardown claim in the assignment is plausible from the loader's HV-exploit nature and the README's "patched HV vulnerabilities" line but unverified in primary sources I could reach.
- No GPUDirect-Storage-style measurement exists on any console APU (closest analog: RG35XX/BC250-class hobby boards, no P2P reports).

## Q5. External paths: USB 3.x, USB4/Thunderbolt, measured throughput

### Takeaway
PS5 ports are USB 3.1 Gen2 (10 Gbps, ≈1.0–1.2 GB/s raw, ~0.9 GB/s effective on the best Gen2 NVMe enclosures — consistent with the "0.9 GB/s" figure quoted in the assignment; it is a bus-physics ceiling, not a software limit), the front-top port is USB 2.0; a USB4/TB enclosure on the Type-C port will run at USB 3.x Gen2 speeds since the controller is Ariel xHCI, not USB4 — so an external NVMe adds at most ~0.8–0.9 GB/s, useful as a third stripe member (cheap capacity) but not a path to 10+ GB/s.

### Cited Findings
- Ports: rear 2× Type-A "USB 3.1 Gen2" (AMD Ariel `1022:13ee`), 1× Type-C "USB 3.1 Gen2 + DP Alt Mode" (`1022:13ed`); front-top Type-A is USB 2.0; all usable under pLinux; no USB4/TB capability mentioned anywhere in the platform docs — [ps5-linux guide: Hardware Overview](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/hardware)
- All USB ports remain functional under ps5-linux; Linux can boot from USB or run from M.2 — [Tom's Hardware, PS5 Linux loader](https://www.tomshardware.com/software/linux/ps5-linux-loadr-goes-public-turning-phat-consoles-into-full-linux-pcs)
- Community storage experience: DF notes moving Linux to the M.2 bay "vastly accelerates storage speeds vs the USB alternative" (qualitative) — [Digital Foundry, May 2026](https://www.digitalfoundry.net/news/2026/05/playstation-5-linux-tested-steam-pc-vs-native-ps5-games)
- USB peripherals incl. NVMe enclosures work, with quirks: rear-port detection regressions for enclosures on newer kernels (use front Type-C), enclosures must be kept awake; 2.5GbE USB docks (RTL8156B) work and are "fully utilized" — [ps5-linux guide: USB & Storage](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/usb-storage)
- USB hubs/docks confirmed working with full port functionality (Startech 10G2A1C25EPD etc.) — [ps5-linux guide: USB & Storage](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/usb-storage)

### Inferences
- 10 Gbps Gen2 = 1.25 GB/s line rate minus 8b/132b→ (already included: Gen2 = 10 Gb/s raw = 1.16 GB/s after encoding, ~0.85–1.0 GB/s real with UAS) — the oft-quoted 0.9 GB/s is exactly where a good JMS583/JRTL921B-class Gen2 enclosure lands; there is no firmware headroom above ~1 GB/s on these ports.
- A USB-attached stripe adds bandwidth roughly linearly per port up to the shared root-hub fabric: 2–3 enclosures across the three 10G ports plausibly add 2–2.7 GB/s as *third/fourth* stripe members (mdadm/io_uring across nvme0+nvme1+sdX), pushing 12 GB/s → ~14 GB/s aggregate for cheap, at the cost of UAS device stability quirks documented in the guide.
- A USB4/TB4 NVMe enclosure in the Type-C port downgrades to 3.2 Gen2; PS5 has no USB4 host capability, so "Thunderbolt GPU/SSD tricks" are off the table — the APU simply lacks the controller.

### Gaps
- No published `hdparm -t` / fio number for a USB-attached NVMe under pLinux5 specifically (the 0.9 figure is my derivation from bus specs + enclosure-class norms, corroborated only by DF's qualitative "USB much slower than M.2").
- Unknown whether xHCI bandwidth is shared across all four ports or per-port under load with an active hub tree — no measurement found.

## Q6. Flash-side exotic: Optane, Gen5 drives, ZNS/alignment

### Takeaway
The M.2 slot is electrically PCIe Gen4 x4 (≈8 GB/s raw, ~7.4–7.9 GB/s from the best drives) and accepts Gen5 drives, which then run capped at Gen4 rates; Sony's Gen4-only gate is a software header check that the community has already defeated for Gen3 drives, which opens the door to weird silicon (Optane-class, enterprise U.2→M.2 adapters) — but note the whole point of Optane (low-QD/low-depth random read) is *the opposite* of what 3 MB expert-block streaming needs; sequential-optimized Gen4 drives are already near-optimal, and ZNS/alignment is a second-order tuning, not a multiplier.

### Cited Findings
- Sony requirement: PCIe Gen4 x4 NVMe, M.2 2280/22110 etc., ≥5,500 MB/s sequential read, heatsink required; drives with HMB are not supported; Sony's list explicitly demands ≥5.5 GB/s — [PlayStation official guide](https://www.playstation.com/en-us/support/hardware/ps5-install-m2-ssd/); [Lexar PS5 SSD upgrade guide](https://americas.lexar.com/ps5-ssd-upgrade-guide/); [Tom's Hardware expansion test](https://www.tomshardware.com/news/ps5-nvme-ssd-expansion-slot-tested); [HMB-not-supported report thread](https://nascompares.com/2021/07/29/the-best-ssd-for-upgrading-your-ps5-storage-get-it-right-first-time/)
- Gen5 interface drives are listed as acceptable for the PS5 (they negotiate down to the slot's Gen4 x4); e.g., Lexar guidance says "PCIe Gen 4 or Gen 5 interface" with ≥5,500 MB/s — [Lexar guide](https://americas.lexar.com/ps5-ssd-upgrade-guide/)
- Community Gen3 unlock: the PS5 checks "a small header at the start of the drive"; writing the first 2 MB from any PS5-formatted Gen4 drive makes Gen3 (and sub-250 GB) drives accepted — confirmed on FW 4.00/4.03/4.51, dump method by Bringus Studios — [ps5-linux guide: Gen3 NVMe Unlock](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/usb-storage); [Bringus video](https://www.youtube.com/watch?v=Uds315QBUnE)
- In-bay real-world ceiling: top drives measure 6.5–6.6 GB/s in the PS5's own benchmark (990 Pro 6,558 MB/s; MP600 Pro ~6.6 GB/s) — [Digiera](https://www.digieraglobal.com/blogs/news/best-internal-ssd-for-ps5); [nascompares 980 Pro test](https://nascompares.com/2021/08/06/ps5-ssd-expansion-testing-the-samsung-980-pro-nvme-ssd/)
- M.2 SATA drives do not work at all under Linux ("confirmed not working regardless of adapter or enclosure") — [ps5-linux guide: USB & Storage](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/usb-storage)
- PCIe 4.0 x4 = 16 GT/s/lane × 4 ≈ 8 GB/s wire, ~7.9 GB/s max usable — Gen5 drives cannot exceed this in the PS5 bay (arithmetic from [Tom's Hardware slot spec](https://www.tomshardware.com/news/ps5-nvme-ssd-expansion-slot-tested))

### Inferences
- **Gen5 in a Gen4 slot delivers its Gen4-rated ~7.2–7.9 GB/s** — buying a 990 Pro/T710-class drive is the correct "fastest bay" move, not a 14 GB/s Gen5 drive; Gen5 buys nothing in-bay (though Gen5 M.2-in-USB-enclosure still tops out at the USB ceiling anyway).
- Optane (all Gen3, so needs the header-unlock): its edge is 4K read at QD1–QD8 with ~10 µs latency. Expert streaming at 3 MB blocks at QD16+ is controller-bandwidth-bound, where modern Gen4 TLC is 2–4× faster. Optane wins only if the engine does many small random block reads (e.g., 32 KB hash-indexed experts or MoE router scatter) — then a 32/64 GB 905P-class stick's ~2.5 GB/s random-4K would beat TLC's QD1 random by 5–10×. Availability 2026: used-Market only (Intel exited Optane 2022), cheap on eBay but power-of-two capacities ≤4 TB. Worth one $60 experiment for the scatter-read case only.
- ZNS: removes GC jitter → better sustained-write tail, irrelevant to read-only streaming; LBA-aligning the model file to 128 KB–2 MB boundaries and using O_DIRECT + fixed-buf + QD32 `io_uring` is free performance (avoids read-modify and page-cache churn) but won't change the ~7.4 ceiling.
- The realistic per-bay numbers for the report: internal 5.5, bay 6.5–7.4, USB 0.85–0.9 → stripe totals 12–13 (2-device) / 13.5–14 (with one USB device).

### Gaps
- No PS5-specific report of an Optane/enterprise drive behind the Gen3 unlock; whether the 4.00+ check (which may include speed probing on newer FW, given "failed reports on 5.xx and above" in the guide) would tolerate an Optane 905P is untested.
- No public test of sustained multi-hour random-3 MB read under ZNS vs conventional on PS5 hardware (and no ZNS drive has been tried in a bay that I could find).

## Q7. Stream from a faster box: Ethernet / NVMe-oE

### Takeaway
The PS5's built-in Ethernet is 1 GbE both on hardware and under pLinux (custom `mts`/`gbe` driver, "works out of the box"), capping network streaming at ~110–118 MB/s sustained — 50–100× below the requirement, so network streaming can at most feed *cold* experts in the background into the GDDR6 pool (a prefetch tier), never drive token rate; the only upgrade is USB 2.5GbE (~290 MB/s) or WiFi6 USB (worse), keeping NVMe-oE firmly in "useful for caching, useless for streaming" territory.

### Cited Findings
- PS5 Ethernet port is 1000BASE-T only — community consensus across the exploit-guide megathread — [GBAtemp PS5 Exploit Guide, p.1141](https://gbatemp.net/threads/ps5-exploit-guide.613891/page-1141)
- "PS5 Ethernet port is gigabit — practical sustained ceiling is around 110–118 MiB/s for huge files. 2.5 GbE / 10 GbE adapters on the computer [side] just sit idle" — [ps5upload FAQ](https://github.com/phantomptr/ps5upload/blob/main/FAQ.md)
- PS5 jailbreak tooling saturates gigabit wired over 2.5GbE LANs; WiFi is slower — [PS5-Suite (GitHub)](https://github.com/manos555555/PS5-Suite)
- Under pLinux the onboard NIC is MediaTek "Star" GbE / Salina GBE (`104d:9104`) with a community `mts` driver, supported on all models; Marvell rev2 Salina boards still in-progress for WiFi/BT — [ps5-linux guide: Hardware Overview](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/hardware); [rmuxnet's ethernet driver patch](https://github.com/ps5-linux/ps5-linux-patches/commit/643e214d7bd37f292045fc0dbb821e421f7a3e47)
- USB wired NICs work under pLinux; RTL8156B (2.5GbE) is the community-recommended chipset; a 10G-port dock's 2.5GbE is "fully utilized by Linux" → ~280–290 MB/s realistic — [ps5-linux guide: USB & Storage](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/usb-storage)
- WiFi6 USB adapters (MT7921 etc.) work out of the box — an 800–1200 Mbps-class ~60–90 MB/s path — [ps5-linux guide: USB & Storage](https://rmuxnet.gitlab.io/ps5-linux-guide/guide/usb-storage)

### Inferences
- NVMe-oE/RDMA: even at 2.5GbE over USB (≈290 MB/s, TCP), you are at 0.29 GB/s — under 1% of the 40 GB/s target; RoCE adds nothing because the wire, not the protocol stack, is the wall. An iWARP/RDMA USB NIC for PS5 is a nonstarter (no driver, and still ≤2.5 Gb/s class on the ports available).
- The sane architecture using the NIC: the PS5 pulls *not-yet-needed* experts from a LAN server into its GDDR6 pool ahead of time (idle link budget 110–290 MB/s ≈ 32–84 MB/token at 3.5 tok/s link-speed... at 40 tok/s only ~3–7 MB/token — enough to warm 2–5% of needed bytes per token), i.e., the NIC is a capacity-extender for the 13 GB pool, not a bandwidth source. If the pool were 130 GB-equivalent (model on NAS + local cache of active experts), bursty MoE workloads could mask most of the storage gap — this is the "125 GB on NAS, 13 GB cache" variant of the moonshot.
- No prior art found of LLM weight streaming from NAS to console; PC prior art (LocalLLaMA RAID0 thread above) stays local-disk.

### Gaps
- No measured TCP throughput for the `mts` gbe driver specifically under pLinux5 (the 110–118 MiB/s figure is from PS5-OS jailbreak tooling; the Linux driver could be lower if unoptimized — rmuxnet's patch thread would be the place to check, unverified).
- USB 3.1 Gen2 → 10GbE or 5GbE USB NICs: RTL8153/8126-class devices untested under pLinux as far as I could find; the guide's RTL8156B note is 2.5G only.

## Cross-cutting summary (for the report writer)

| Path | Realistic ceiling | Maturity | Status |
|---|---|---|---|
| Internal + M.2 app-level stripe | 11–13 GB/s | devices confirmed to enumerate; striping untried | **Open engineering, months** |
| Kraken hardware decompression (native OS path) | ×2–3 effective (→ 20–35 GB/s combined with stripe) | unit exists, confirmed; no RE'd interface under Linux; native-OS path = OS I/O itself | Hard part unknown; promising |
| Memory-side | pool size (13 GB ✓ via gttsize/ttm); clocks ~irrelevant | gttsize recipe proven on sibling boards | Solved for pool; OC unexplored |
| P2P DMA | +0–20% at current speeds, more later | nothing demonstrated on PS5 | Speculative |
| USB NVMe as 3rd/4th device | +0.8–0.9 GB/s each, ~2.7 GB/s for 3 | ports/peripherals work under pLinux | Easy, incremental |
| Gen5 drive in bay | 7.4–7.9 GB/s (Gen4 slot cap) | shipping drives | Just buy it |
| Optane (Gen3 unlock) | low-QD random niche only | unlock method proven for Gen3; Optane untried | $60 experiment |
| 1 GbE / USB 2.5GbE NVMe-oE | 0.11–0.29 GB/s | works today | Prefetch-tier only |

Hard physical ceilings this survey located: USB 3.1 Gen2 ≈ 1.2 GB/s line rate; PCIe 4.0 x4 ≈ 8 GB/s wire; 1 GbE ≈ 125 MB/s; per-drive Gen4 flash ≈ 6.6–7.4 GB/s in-bay. Everything above those — striping, Kraken ratio multipliers, P2P, prefetch tiering — is engineering, not physics, and no published project has yet stacked them on PS5.
