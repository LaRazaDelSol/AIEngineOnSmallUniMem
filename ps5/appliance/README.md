# ps5/appliance — the PS5 Linux console as an inference appliance (S15)

One script, two boot modes. Everything here was measured on the console (S14b/S15); the S12 drafts
(`cmdline.conf`, sysctl/udev files, `ps5lm-strip.sh`, CPU-layout + `ps5lm-runtime` units) were removed in S15:
they assumed the wrong SMT map (siblings are (2k, 2k+1)), 1 GiB hugepages, a root label and a binary that do not
exist, and they contradicted `ps5lm-min`.

| file | installed as | what |
|---|---|---|
| `ps5lm-min.sh` | `/usr/local/sbin/ps5lm-min` | `on` / `off` / `status` (runtime, reversible) · `cmdline` (boot params) · `install` · `boot ai\|game` |
| `ps5lm-min.service` | `/etc/systemd/system/` | AI mode: `ps5lm-min on` at boot |
| `ps5lm-server.service` | `/etc/systemd/system/` | AI mode: OpenAI + Anthropic server on :8081 (slow restart policy: pinned memory must drain) |
| `getty-tty1-ps5lm-ai.conf` | `/etc/systemd/system/getty@tty1.service.d/zz-ps5lm-ai.conf` | AI mode: the tty1 autologin (gamescope + Steam, ~1.1 GB) never starts |
| `etc-ps5lm/server.env` | `/etc/ps5lm/server.env` | `PS5LM_BITS`, extra server args |

All three units carry `ConditionKernelCommandLine=ps5lm.ai=1` (the getty drop-in the negation), so **game mode
boots exactly as before** and AI mode is one kernel flag:

```
sudo bash ps5/appliance/ps5lm-min.sh install     # from a repo checkout on the console, once
sudo ps5lm-min cmdline                           # kho=off udmabuf.size_limit_mb=16384 swiotlb=4096 ... (once)
sudo ps5lm-min boot ai && sudo systemctl reboot  # AI appliance from now on (soft reboots)
sudo ps5lm-min boot game && sudo systemctl reboot  # normal console again
sudo ps5lm-min status                            # memory budget, boot mode, server state
```

Cold boots read the loader USB's `cmdline.txt`, not `/boot/efi/cmdline.txt`, so a cold boot is a guaranteed way
back to game mode (and needs `ps5lm.ai=1` on the USB copy to come up in AI mode). Soft reboot only, never
power-cycle.

Boot parameters in use (`ps5lm-min cmdline`): `kho=off` (KHO kept 1.5 GiB as CMA scratch the pinned slab cannot
use), `udmabuf.size_limit_mb=16384`, `swiotlb=4096` (64 → 8 MiB bounce pool; peak use 0.5 MiB), `nowatchdog
nmi_watchdog=0 init_on_alloc=0 init_on_free=0`, plus the console's own `ttm.pages_limit/page_pool_size=4194304
amdgpu.gttsize=14336 mitigations=off preempt=full`. Not used, on purpose: `isolcpus`/`nohz_full` (the engine is one
host thread + io_uring; nothing to isolate from), 1 GiB hugepages (the engine reserves 2 MiB hugetlb pages itself and
returns them at exit), APST off (unmeasured on the SN810).
