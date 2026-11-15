# PS5 probe runbook — Gate 0/1 measurements over the network

> **New session? Start at `docs/PS5-PROBE-SESSION-BRIEF.md`** (entry point:
> state of the world, run order, safety, and what a successful session ends with).
> Rewritten 2026-10-21 to the W6/W7 facts: upstream **ps5-linux/ps5-linux-loader**
> boots 13.00 Phat/Slim with no firmware update; **pLinux5-NG is archived** — never
> send `pLinux5-NG.elf` (maintainer's own branch note: brick risk, "factory reset to
> recover"). This file's prior revision still said "pLinux5-NG, 13.20 confirmed;
> 13.00 experimental" — that was always the wrong repo's matrix.

Goal: turn the plan's open questions into measured facts on **this** console
(PS5 **Phat CFI-1000, FW 13.00**, never PSN, `10.0.0.113`), without ever trusting a
BC-250/EaseUS number. Everything below runs from this PC; the console needs to be
jailbroken and on the LAN.

Software: `tools/ps5probe/probe.py` (PC side, Python stdlib) + `tools/ps5probe/gate0_1.sh`
(pushed over ssh to the console; also runnable by hand). Raw evidence lands in
`recon/ps5probe/`, verdicts in `recon/gates/ps5-gate-<ts>.md`.

## Step 0 — PC + console prep (once, ~30 min)

1. **Freeze firmware.** Never sign in to PSN, never update while the project lives.
   13.00 is the supported target for BOTH tracks (Relapse JB 7.00–13.60; upstream
   ps5-linux-loader lists Phat/Slim 13.00 natively). 14.x = dead end, say so and stop.
2. PC-side build (Linux/WSL — Windows alone can't build payloads):
   ```
   # ps5-payload-sdk (binary release, works in WSL Ubuntu):  https://github.com/ps5-payload-dev/sdk
   git clone https://github.com/ps5-linux/ps5-linux-loader && cd ps5-linux-loader && make
   ```
   **HEAD must be ≥ `73f10194`** (PR #54, the 13.00 WLAN offset fix, merged 10-20).
   The v2.5 **release ELF lacks 13.00** — if it says "make sure your fw is
   supported", you have the wrong ELF; rebuild from `main`.
3. Flash `ps5-ubuntu2604.img.xz` from the ps5-linux-image `latest` release
   (rolling; 2026-08-29 build; sha256 `1959069…35c6555b`) to a **≥64 GB USB SSD**.
   Plug into **front-bottom Type-C or a rear Type-A** (front top A is USB 2.0, slow).
   Console settings: Settings → System → Power Saving → Features Available in Rest
   Mode → **Supply Power to USB Ports = Always**; disable **HDMI Device Link**;
   auto-updates off.
4. Fixed DHCP lease for the console (router) so probes don't chase IPs.
5. `ssh` client required (Windows 10+ ships it). Python ≥3.8 on PATH.
   Sanity — everything closed means "not jailbroken yet":
   `python -I tools/ps5probe/probe.py discover --subnet 10.0.0.`

## Step 1 — Phase A: PS5LM baseline (HEN stack ON, ~30 min)

Do this FIRST — PS5LM *requires* kstuff-lite + etaHEN + ShadowMountPlus loaded,
which are exactly what the Linux loader needs OFF. One jailbreak session cannot
serve both (this is the sequencing trap: skip it and the loader "freezes" in a way
that looks like an upstream bug but is self-inflicted).

1. Jailbreak per RELAPSE flow (Settings → Network → DNS `45.56.67.85` → User's
   Guide page → accept cert → **Jailbreak**), **with kstuff-lite + etaHEN +
   ShadowMountPlus autoload**. elfldr listens on :9021.
2. Install PS5LM v0.9.1 **by manual FTP copy** (`ftpsrv` :2121; unzip the release,
   copy the `PPSA99575` folder to `/data/homebrew`; issue #34 shows webui installs
   failing — don't use one). ShadowMountPlus puts the tile on the home screen.
   v0.9.1 asks the HEN to open `/data` at start (OnionHEN: add `PPSA99575` to
   `app_jailbreak.exact_title_ids` in `/data/OnionHEN/config.ini`).
3. Also run the native storage pass while Pegasus is up:
   `python -I tools/ps5probe/probe.py native --host 10.0.0.113 --copytest --bwtest`
   `--bwtest` self-creates an 8 GiB file on /data (streamed, ~6 min once, reused
   after) and same-mount copies it → **the internal-SSD r+w floor at 13.00 without
   Linux**. Plain `--copytest` alone only measures whatever ≥1 GiB file happens to
   exist — on this console that was the USB stick (82.5 MB/s is a USB number).
4. Launch PS5LM, load **Qwen3.8-27B UD-IQ2_XXS or UD-Q2_K_XL**, record tok/s
   (reference: 20.7 / 20.3 on a PS5 Slim — Phat should match; OUR number is the
   point). Check the API: `:8081/v1/chat/completions`, `/v1/messages`, `/health`,
   `/metrics`. **Do not scrape ps5-exporter :9100 during timing** (issue #31 —
   scraping stalls generation).
5. Record the full FW string + kit type from the health dump (this session's
   findings may be the first public 13.00-Phat datapoints — treat every number as
   publishable).

## Step 2 — Transition + Phase B: boot Linux at 13.00 (one reboot, ~15 min)

1. **Reboot the console** (the JB dies with it — that's the feature).
2. Re-run Relapse (same flow) with **kstuff/etaHEN autoload OFF** — clean,
   elfldr-only session. No ShadowMountPlus needed for this phase.
3. Send the loader built in Step 0.2:
   `socat -t 99999999 - TCP:10.0.0.113:9021 < ps5-linux-loader.elf`
4. Expected good path: payload → auto **rest mode** → solid orange → press power →
   **white LED + HDMI** = Linux. Ubuntu OOBE on the TV: set your login (no default
   creds exist), enable SSH, then per README `apt-mark hold linux-*`, and set
   `vram.txt` (FAT32 p1) = `0x20000000` (512 MB Dynamic VRAM) for engine work.
5. **Triage:**
   - "…make sure your fw is supported" / never enters rest mode → wrong ELF (rebuild from `main` ≥ #54) or rest-mode USB power off.
   - Freezes **before** rest mode → kstuff/etaHEN still loaded. Reboot, re-JB clean, retry.
   - Rest mode but **never wakes** (dead buttons) → Safe Mode → Restart (no data loss). **Log model + full FW string** — a miss here is still a first public 13.00-Phat datapoint.
   - White-but-blackscreen → drop `video=DP-1:…` from `cmdline.txt`; try `amdgpu.force_1080p=1`; `RADV_DEBUG=nohiz`.
6. Immediately after boot, raise GPU memory and defeat the tmpfiles silent-clobber:
   ```
   sudo modprobe ttm 2>/dev/null
   echo 4194304 | sudo tee /sys/module/ttm/parameters/pages_limit /sys/module/ttm/parameters/page_pool_size
   printf 'options ttm pages_limit=4194304 page_pool_size=4194304\n' | sudo tee /etc/modprobe.d/ttm-gpu-memory.conf
   printf 'w /sys/module/ttm/parameters/pages_limit - - - 4194304\nw /sys/module/ttm/parameters/page_pool_size - - - 4194304\n' | sudo tee /etc/tmpfiles.d/ttm-gpu.conf
   cat /sys/module/ttm/parameters/pages_limit   # must read 4194304 after a reboot, not 3145728
   ```
   (S5 fix 2026-10-21: the tmpfiles type must be `w` — "write the value to the file". The
   old `x` means "ignore a path during cleanup", which wrote nothing; the `modprobe.d`
   line above is the durable fix and this re-asserts it if ttm loads after tmpfiles.)
7. Fan + boost BEFORE any measurement: `ps5_control --fan on && ps5_control --boost on`
   (always fan first — the loader's own guidance). The battery records governor/
   boost/PCIe state into the JSON either way (W2 #15) — numbers without clock state
   are unattributable.

## Step 3 — SSH from the PC (once)

1. Find the Linux IP: `probe.py discover` (22 will be the new open port), or the router.
2. Key it so BatchMode works (no sshpass on Windows — one-time typing):
   ```
   ssh-keygen -t ed25519           # if you don't have a key
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <user>@<IP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
   ```
   (user = whatever you chose in OOBE; update `--user` accordingly.)

## Step 4 — Phase C: run the full gate battery (~6 min; N7 store +~2 min first run)

```
python -I tools/ps5probe/probe.py linux --host <IP> --user <user> --n7
```

- Pushes `gate0_1.sh` over `ssh … bash -s`, collects JSON, evaluates every gate,
  writes `recon/gates/ps5-gate-<ts>.md` + `.json` and `recon/ps5probe/linux-<ts>.json`.
- Covers in one pass: **G0** (boot/GPU identity/M.2 present/ttm/clock state),
  **G1** (seq/rand~2.6 MiB/rand-4k/buffered on a **100 GiB** SLC-honest scratch file
  with settle time; **the device it measured is printed in every G1 row** — W2 #17),
  **G0+** (both NVMe enumerated), **N6** (concurrent dual-device stripe — dead
  branch on this console until an M.2 is bought), **N7** (`--n7`: expert-cache
  replay, synthetic Zipf s=1.1 with **distinct** top-10 draws over a 40 GiB store
  behind a min(35% RAM, 3 GiB) cgroup cap; store persists — `N7_KEEP=0` deletes it).
- Physics uses **G = 1.24 GiB/token** (config.json recompute; the 1.40 figure was
  ~13% high); override with `--gib-per-token`, `--expert_gb`, `--k`, `--experts`.
- `report recon/ps5probe/linux-<ts>.json` re-evaluates a saved run without redoing
  the battery.
- Gates NOT measurable from the network (engine phase): **N3** (overlap),
  **N4/N4a** (MTP — zero instrumentation, the projection row prints its ×1.44 as
  *UNMEASURED* for a reason), **N5** (real-pack hit rate; N7 is the stand-in).
  Per the S2 audit, **N5/N7 are curve measurements, not pass/fail gates**: report
  `hit% vs cache capacity`, then compute tok/s = BW/(G×(1−h)).

## Step 5 — Read the verdict

Physics (W4 recomputed from config.json): `tok/s = SSD_BW ÷ (1.24 GiB × (1 − hit))`,
480 expert reads of 2,764,800 B/token. Published hit-rate reality: agent traces
0.40–0.72, static ceiling 0.79. **40–60 tok/s needs hit 90–93.5% at single-NVMe
BW** — not planable; the honest bands are **6–10 (full pack) / 12–25 (REAP-288 +
overlap)** with MTP OFF until N4 measures acceptance on agent traffic. External
anchor as of today: Strata on a 16 GB Mac mini (closest shipped analog): 11.7→27.2
tok/s at Q2_0, SSD streaming, MTP auto-off.

| Observation | Meaning | Do |
|---|---|---|
| G1 seq ≥ 5000 MB/s (device = internal 825!) | single-drive BW as modeled | feed the projection row |
| G1 seq 3000–5000 | lower bound; misses need SSD at (1−h) rate | recheck SLC/settle, re-run once |
| G1 row names the M.2/USB device | you measured the wrong disk | re-run with `--g1-target <mount>` — every G1 row prints the device. **First find the mount:** on the console run `findmnt -t ext4`; `/data` is the *native-OS* path and may not exist under ps5-linux. If the internal 825 is mounted nowhere, say so — record "G1 measured on boot drive" as a caveat |
| rand ~2.6MiB < 60% of seq | LBA-alignment bug | check partition start; ps5-linux-tools `m2_init` (only if an M.2 is installed) |
| ttm row CHECK (3145728) | tmpfiles clobber | Step 2.6 fix, reboot, re-run |
| G0 M.2 row CHECK | no M.2 hardware (known) | N6/G0+ are a ~$100–150 purchase branch, not a gate to pass tonight |
| N7 prints hit at 2–3 GiB cache | the actual missing curve | plot capacity sweep (re-run with `--n7-cap-mb 1024` then `2048`, `3072`, `6144` — the store persists, each re-run is ~2 min) — this row IS the project's key measurement |
| `vk_ext_host` CHECK/0 | zero-copy landing unavailable | engine uses host-visible pool copies; budget the memcpy |
| GPU identity ≠ GFX1013 | lore adjustment | note actual device_id; re-check notes/ps5_loaders |

## Step 6 — Provisioning (Phase 2+, first real transfer)

Build the expert store **on the PC** (unpack GGUF → one file per expert + SHA-256
manifest, MoE-Direct layout), push once, verify on target — re-pushes over LAN are
the budget-killer (~10 min per 72.8 GB at gigabit): `scp` (Step 3 key) or Pegasus
`PUT /api/files/upload?directory=…` on the native side, `rsync` over ssh on the
Linux side. On target: `sha256sum -c manifest.txt`.

## Safety rails

- `probe.py native` writes only inside `ps5-probe/` sandboxes and deletes after;
  destructive Pegasus routes are used only on that path. All ops are single-filed
  with freshness-checked awaited deletes (the Pegasus concurrent-ops hazard is real
  — never queue two ops at once).
- `gate0_1.sh` is read-only except scratch under the target mount; N7 store
  persists (`<mount>/ps5probe-scratch/n7.bin`) — reclaim manually if abandoned.
- fio wear is now ~100 GiB create per run (~1–2 write-equivalents on the OEM
  drive; still negligible, but don't loop it).
- One-way door: a PSN/update kills the hypervisor JB forever (the HV bug Relapse
  uses is patched in 14.10; PSN hard-requires 14.10). Firewall the console's VLAN,
  not just "remember not to".

## How this maps to the tools in this session

- **shell + python** — run probe.py, parse JSON, compare thresholds; Steps 0–4 are
  one command each.
- **webfetch/websearch** — re-verify loader READMEs before Step 2 (the landscape
  moved twice in five weeks; it can move again — check ps5-linux-loader `main`
  commits ≥ `73f10194` are still HEAD or read what changed).
- **REA** — only if a route stops matching the captured Pegasus API samples in the
  sibling WingsBrowser project.
