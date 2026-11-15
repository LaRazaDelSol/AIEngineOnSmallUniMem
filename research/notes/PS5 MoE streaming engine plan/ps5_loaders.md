# PS5 Linux Loaders — Current State (October 2026)

> **Status (2026-10-21):** ACTIVE WITH CORRECTIONS — fabricated tokens pruned in place (STATE F12, S5).

## State of ps5-linux-loader (ps5-linux org) — firmware, consoles, M.2, memory carve-out

### Takeaway
ps5-linux-loader is now maintained under the `ps5-linux` GitHub org (not the user "ps5-linux-loader"), and its README (current as of Oct 2026) supports PS5 **Phat and Slim** on firmware **3.00–7.61**, with M.2 SSD-as-Linux-partition support on 4.xx–7.61 (3.xx lacks M.2 because the PS5 fails to boot with it attached). TheFlow (Andy Nguyen) publicly abandoned the project on 15 Sep 2026 after the underlying hypervisor bug was reported to Sony, but the repos remain live and unarchived.

### Cited Findings
- Project README states ps5-linux "leverages patched HV vulnerabilities" for **PS5 Phat and Slim** on **3.00–7.61**; features listed: HDMI 4K60, M.2 SSD as dedicated Linux partition, all USB ports, BD drive via custom ahci driver, internal Bluetooth via custom xhci driver, Ethernet via custom gbe driver — [Source](https://github.com/ps5-linux/ps5-linux-loader)
- Firmware support table (exact): **3.00, 3.10, 3.20, 3.21 without M.2 support**; **4.00, 4.02, 4.03, 4.50, 4.51 with M.2 support**; **5.00, 5.02, 5.10, 5.50 with M.2 support**; **6.00, 6.02, 6.50 with M.2 support**; **7.00, 7.01, 7.20, 7.40, 7.60, 7.61 with M.2 support** — [Source](https://github.com/ps5-linux/ps5-linux-loader)
- 1.xx/2.xx "may be added in the future, but we will not prioritize this effort"; FAQ explicitly answers "Will higher >=8.00 firmwares be supported? A: No" — [Source](https://github.com/ps5-linux/ps5-linux-loader)
- FAQ: "Why can I not use M.2 on 3.xx? A: Because the PS5 fails to boot with it attached." — [Source](https://github.com/ps5-linux/ps5-linux-loader)
- M.2 install flow: `m2_init` (creates fake MBR/GPT header to satisfy PS5 OS), `m2_install.sh` (formats M.2 ext4, copies image), `m2_exec.sh` (kexec into Linux on M.2); M.2 is exclusive to Linux (cannot dual-use for PS5 game storage); still requires a USB FAT32 stick to boot — [Source](https://github.com/ps5-linux/ps5-linux-tools) and [Source](https://github.com/ps5-linux/ps5-linux-loader)
- Memory carve-out: kernel cmdline `vram.txt` in FAT32 partition; "By default, it uses 512MB (0x20000000) which enables Dynamic VRAM allocation" (per AMD BC250 docs) — [Source](https://github.com/ps5-linux/ps5-linux-loader)
- Digital Foundry (4 May 2026): ~15GB usable memory, full 8 cores/16 threads Zen 2, full 36 CU GPU; boot clocks CPU 3.2GHz / GPU 2.0GHz, boost mode CPU 3.5GHz / GPU 2.23GHz; dynamic graphics allocation "seems to top out at around 5.5GB" — [Source](https://www.digitalfoundry.net/news/2026/05/playstation-5-linux-tested-steam-pc-vs-native-ps5-games)
- BC250-derived kernel params to raise dynamic VRAM/GTT ceiling: `ttm.pages_limit=<GB×1024×1024/4>` + matching `ttm.page_pool_size` (recipe per elektricM BC250 docs). **S5 PRUNE 2026-10-21: this bullet originally cited `amdgpu.gttsize=14750` and a DF comment — fabricated token + unverifiable citation (STATE F12/W5).** The "40 CU enable verified" claim is BC-250-only and worth ~0 on PS5 (verified-plan C9) — [elektricM/amd-bc250-docs — Kernel Configuration](https://elektricm.github.io/amd-bc250-docs/linux/kernel/)
- TheFlow's abandonment: announced on X 15 Sep 2026, "stepping away from the ps5 scene and stopping all my work on ps5 linux," including plans to finish PS5 Pro support and release in 2027; repos still live, not archived — [Source](https://itsfoss.com/news/ps5-linux-lead-quits) and [Source](https://www.tweaktown.com/news/113593/developer-abandons-ps5-linux-project-after-people-found-a-hypervisor-bug-and-reported-it-to-sony/index.html)
- Repo stats (Oct 2026): ps5-linux-loader 282 stars / 72 forks / 68 commits; ps5-linux-tools 12 stars; ps5-linux-image 43 stars — [Source](https://github.com/ps5-linux/ps5-linux-loader)
- Pre-built image is `ps5-ubuntu2604.img.xz` (Ubuntu 26.04); builder supports Ubuntu 26.04, Arch, CachyOS (Gamescope+Steam), Fedora, and a multi-distro 32GB image with kexec switching — [Source](https://github.com/ps5-linux/ps5-linux-image)

### Inferences
- The build spec's "3.xx–4.xx / Phat only" understanding is **outdated**: the project now supports through 7.61 and includes Slim, and M.2 support is gated to 4.xx+ firmware.
- Because TheFlow halted PS5 Pro work, any Pro support is unconfirmed and never reached the repo; the "console support" column should list Phat + Slim only (confirmed), Pro as unconfirmed/abandoned.

### Gaps
- No quantitative figure from the project for M.2 throughput; only the qualitative DF claim (see throughput section).

---

## State of pLinux5-NG (Byepervisor-Reborn hypervisor escape port)

### Takeaway
pLinux5-NG (pLabs5) is a port of ps5-linux-loader to the new Byepervisor-Reborn hypervisor escape, targeting **high firmware 8.xx–13.60**. It is explicitly **VERY work-in-progress**: as of Oct 2026 only **13.20** is confirmed booting; 13.60 is the headline/experimental target, and 14.00/14.10 are blocked because they lack a public kernel exploit and the Reborn/Relapse bypass no longer works on them.

### Cited Findings
- pLinux5-NG README: "boot linux on PS5 consoles running high firmware (8.xx through 13.60)"; a port/rework of ps5-linux-loader (which works 3.00–7.61) to the byepervisor-reborn hypervisor escape; boot path (bzImage/initrd relocation, boot_params/e820, hardware bring-up, SVM/IOMMU teardown, jump to startup_64) brought over from ps5-linux-loader; the escape changed to a khook on the S3-resume `HV_MP_STARTUP_CALL` path per Byepervisor-Reborn — [Source](https://github.com/pLabs5/pLinux5-NG)
- README banner: "this project is still **VERY** Work-in-progress, currently only 13.20 is supported until i can get offsets for other FW's / full linux support" — [Source](https://github.com/pLabs5/pLinux5-NG)
- Jailbreak flow uses RELAPSE (DNS `45.56.67.85`, User's Guide page, "Jailbreak" button → elfldr), then sends `pLinux5-NG.elf` on port 9021; AGPL-3.0 license; 191 stars / 28 forks / 19 commits — [Source](https://github.com/pLabs5/pLinux5-NG)
- Byepervisor-Reborn (PS5Dev): "PS5 hypervisor exploit for 6.xx all the way up to 13.60"; initial implementation lists support for **13.60 and 13.20**, plus an **untested 13.00** configuration; underlying vulnerability could in principle apply to 6.xx but that is not current tested support — [Source](https://wccftech.com/byepervisor-reborn-cracks-ps5-hypervisor-13-60-not-useful-regular-users-yet) and [Source](https://ixbt.games/en/news/2026/10/13/sony-prosto-ne-daiut-pokoia-dlia-ps5-vysel-novyi-eksploit-byepervisor-reborn-otkryvaiushhii-perspektivy-dlia-linux-i-nelicenzirovannyx-prilozhenii.html)
- Exploit mechanism: targets a vulnerability in the console's recovery from Rest Mode (S3 resume) — the `startup_MP` hypercall invoked on core 0 to initialize secondary cores — gaining hypervisor root-mode access and disabling execute-only-memory (XOM) protections, enabling kernel dump/patch; bug initially found by TheFlow, subsequently rediscovered by others with LLM assistance; released ~12–13 Oct 2026 — [Source](https://www.youtube.com/watch?v=WZYeqqCkXC0) and [Source](https://wccftech.com/byepervisor-reborn-cracks-ps5-hypervisor-13-60-not-useful-regular-users-yet)
- 14.00/14.10 blockers: "Firmware 14.00 and 14.10, released in September and October, block the Relapse exploit and Reborn hypervisor bypass"; PSN also no longer accessible on 13.60 (Sony demands latest software to log in) — [Source](https://www.notebookcheck.net/Linux-now-runs-on-PS5-console-jailbreaks-and-July-firmware-with-PC-games-possible.1425575.0.html)
- Developers "suggest a modified implementation could work on 14.00, but none is provided, and that firmware lacks a public kernel exploit anyway"; Reborn ships without homebrew-enabling patches — [Source](https://wccftech.com/byepervisor-reborn-cracks-ps5-hypervisor-13-60-not-useful-regular-users-yet)
- Experimental 13.60 Linux build tested (Sonic-Iso) on original and Slim PS5 consoles on 13.60; upstream ps5-linux firmware list does not include 13.60 — [Source](https://www.youtube.com/watch?v=oDnIUKDty8U)

### Inferences
- pLinux5-NG is pre-release: do not treat 13.60 as confirmed working; only 13.20 is confirmed, and even that is WIP. Hardware confirmed working so far is original/Slim PS5 on 13.60 (experimental), not Pro.
- The deciding "which firmware boots" question for a plan should reflect: 13.20 confirmed; 13.60 experimental; 13.00 untested; 14.00/14.10 blocked (no public kernel exploit + Reborn/Relapse neutralized).

### Gaps
- No evidence of pLinux5-NG M.2 support or throughput; README does not mention M.2 — M.2 behavior on high firmware is unconfirmed.

---

## Measured M.2 SSD throughput under Linux on PS5 (the deciding unknown)

### Takeaway
**No published measured M.2 SSD throughput figure (fio or similar) under Linux on a PS5 was found** in any searched source as of Oct 2026. The only relevant datapoints are qualitative (DF: "vastly accelerating storage speeds vs the USB alternative"), a TheFlow M.2 performance fix (LBA alignment) with no numbers, and PS5-OS-level M.2 read figures (~6.5–6.6 GB/s) that are not Linux measurements. This remains an unresolved gap.

### Cited Findings
- Digital Foundry (4 May 2026): "Linux can be moved to an SSD that sits in the PlayStation 5's m.2 bay, vastly accelerating storage speeds vs the USB alternative — though a USB stick is still required to boot the OS." No numeric throughput given anywhere in the article — [Source](https://www.digitalfoundry.net/news/2026/05/playstation-5-linux-tested-steam-pc-vs-native-ps5-games)
- TheFlow committed an "M.2 performance fix by aligning the LBA" (~May 2026, per Modded Warfare 25 May 2026); users must reinitialize/reinstall Linux on the M.2 to apply it; no before/after numbers reported — [Source](https://www.youtube.com/watch?v=fM1bJfnI2YU) and [Source](https://github.com/ps5-linux/ps5-linux-tools/activity)
- Gamers Nexus (2026) booted Linux from an **external USB M.2 enclosure** (their PS5 was on 3.xx firmware which lacks internal M.2 support): "We didn't directly notice any IO speed problems... we also don't have a way to check it one way or the other" — no measurement — [Source](https://www.youtube.com/watch?v=I2OsvSc8VeE)
- PS5-OS-level M.2 read throughput (context, not Linux): PS5 OS M.2 speed tests report ~6.5–6.6 GB/s reads (e.g. 6634 MB/s, 6.6 GB/s on 980 Pro); internal 825GB SSD write ~250 MB/s; EaseUS cites 5.5 GB/s uncompressed / 9.9 GB/s Kraken-compressed — [Source](https://www.reddit.com/r/PS5/comments/ozwah7/ps5_firmware_20_tested_m2_ssd_storage_upgrade_vs), [Source](https://www.youtube.com/watch?v=j-hz4QeOJkA), [Source](https://www.easeus.com/partition-master/ps-5-ssd-speed-test.html)
- The PS5 M.2 slot is PCIe Gen4x4 NVMe; Sony recommends ≥5,500 MB/s sequential read; M.2 Linux partition uses ext4 (per m2_install.sh) — [Source](https://www.playstation.com/en-us/support/hardware/ps5-install-m2-ssd) and [Source](https://github.com/ps5-linux/ps5-linux-tools)

### Inferences
- The deciding unknown cannot be resolved from public sources: any throughput claim in the build spec must be flagged as unverified. The most defensible statement is "Linux M.2 is qualitatively faster than USB (DF), but no measured fio figure has been published; PS5-OS M.2 reads are ~6.5–6.6 GB/s."

### Gaps
- No fio/hdparm/dd throughput figure for the PS5 M.2 slot under Linux exists in any located source. This is the single unresolved deciding unknown.

---

## PS5 thermals under sustained load in Linux (ps5_control, boost, overheating)

### Takeaway
The only documented thermal control is the `ps5_control` tool (via `/dev/icc` and `/dev/mp1`) which toggles fan and CPU/GPU boost; the project's explicit guidance is to **always enable the fan when boosting** because the console cooling was designed for Sony's own power management. No specific sustained-load temperature/overheating datapoints under Linux were found.

### Cited Findings
- `ps5_control.c`: "Allows turning on/off fan and boost via /dev/icc and /dev/mp1"; systemd services `ps5fan.service` and `ps5boost.service` run `ps5_control --fan on` / `--boost on` at boot; install.sh enables both — [Source](https://github.com/ps5-linux/ps5-linux-tools)
- README: "Always turn on fan when your turn on boost, as this is what the official PS5 OS does." Boost raises CPU to 3500MHz and GPU to 2230MHz with adjustable fan curve — [Source](https://github.com/ps5-linux/ps5-linux-loader)
- Tom's Hardware: "Nguyen warns users should always enable the fan profile when boosting, as the console's cooling was designed for Sony's own power management" — [Source](https://www.tomshardware.com/software/linux/ps5-linux-loadr-goes-public-turning-phat-consoles-into-full-linux-pcs)
- Digital Foundry: boost mode sets CPU 3.5GHz / GPU 2.23GHz; "The tool also allows you to increase cooling to compensate" — [Source](https://www.digitalfoundry.net/news/2026/05/playstation-5-linux-tested-steam-pc-vs-native-ps5-games)
- Gamers Nexus: overclocking to 3500/2230 MHz gave "roughly 5–7% performance improvements"; no thermal readings reported — [Source](https://daily.dev/posts/we-hacked-overclocked-a-ps5-to-install-linux-and-steam-hg7hwyx9u)
- No Linux-specific overheating reports (e.g. shutdowns under boost) were located; general PS5 overheating sources cover PS5-OS gaming, not Linux.

### Inferences
- For a sustained-load plan, the safe assumption is: boost requires fan-on to avoid thermal risk; the fan curve is user-adjustable; no measured thermal headroom under Linux has been published.

### Gaps
- No measured temperature/fan/throttle behavior under sustained Linux load (especially boost) was found in public sources.

---

## Linux access to the PS5 hardware I/O decompressor

### Takeaway
The PS5's Kraken hardware decompressor is a confirmed hardware feature, but **no evidence was found that Linux (ps5-linux projects or drivers) accesses the PS5's hardware I/O decompressor**. Linux runs the M.2/SSD as plain NVMe/ext4 with no decompressor driver, and no upstreamed decompressor work was found.

### Cited Findings
- PS5 hardware decompression is confirmed: a dedicated "Kraken" decompression unit (a zlib-like format, "literally doing decompression in hardware," not CPU/GPU) — [Source](https://gamingbolt.com/former-frostbite-software-engineer-explains-ps5s-kraken-compression-unit)
- EaseUS: PS5 reads uncompressed data up to 5.5 GB/s and Kraken-compressed data up to 9.9 GB/s via its SSD interface — [Source](https://www.easeus.com/partition-master/ps-5-ssd-speed-test.html)
- ps5-linux upstreamed changes are limited to `drm/amd: fix dcn 2.01 check` and `amd/addrlib: Add more GFX1013 GPUs` — no decompressor driver in the upstreamed list — [Source](https://github.com/ps5-linux/ps5-linux-loader)
- No reference to a hardware decompressor/compressor driver in ps5-linux README, tools, or image builder was found.

### Inferences
- Linux on PS5 currently treats storage as plain NVMe; the Kraken decompressor is not exposed to Linux in any documented form. Any plan relying on Linux-side hardware decompression is unsupported by current evidence.

### Gaps
- No source confirms or denies whether the PS5's hardware decompressor can be driven from Linux; the absence of any driver/mention is the only signal.
