# S26 — why the GPU cannot touch the reserved low gigabyte

Scope: the ~1,008 MiB range `0x01000000–0x3FFFFFFF` that the S25 reclaim handed to
Linux and the GPU refused to use. Everything below is measured on the console
(`steam@10.0.0.133`, kernel 7.2.3, APU PCI `20:00.0` "Cyan Skillfish"), read-only.
No register was written and the serving engine was not interrupted.

## 1. What the machine-check records actually say

The failed-reclaim boot was a **soft** reboot, so the kernel ring buffer carried the
fault records into the current boot. Full record set:

```
Bank 31..46: MCi_STATUS = 0xdc2030000002085b   (and ...082b on the second sweep)
MCi_ADDR     = 0x1dc00200, 0x1dc00300, 0x1dc00400, 0x1dc00500,
               0x1dc05600..0x1dc05b00, 0x1dc11000,
               0x207b1300, 0x207b6280, 0x207b9000, 0x207bc000..0x207bc300,
               0x2225c000, 0x223ec100, 0x24801300..0x24801f00
MCi_MISC     = 0xd01b0fff00000000   (constant across every bank)
MCi_SYND     = 0x5b260010..0x5b26001b, 0x5b260028   (high half constant: 0x5b26____)
MCi_IPID     = 0x2002e00000002..0x2002e0000000f
PROCESSOR 2:840f60 ... microcode 8406006
```

Three things fall out of that, and each is a measurement, not an inference:

1. **Every faulting address is inside the reclaimed range.** The lowest is
   `0x1dc00200` (~475 MiB) and the highest `0x24801f00` (~584 MiB); the range is
   `0x01000000–0x3FFFFFFF` (~16 MiB–1 GiB). No fault landed outside it. The fabric
   was not refusing "memory" in general — it was refusing *that* range.

2. **The faulting addresses are dense, contiguous, 4 KiB-strided runs** —
   `0x1dc00200, 0x1dc00300, 0x1dc00400, 0x1dc00500`, then `0x1dc05600…0x1dc05b00`,
   `0x207bc000…0x207bc300`, `0x24801300…0x24801f00`. That is the footprint of a
   **driver walking a buffer page by page**, not the sparse scatter you get when a
   per-requester permission table simply misses some entries.

3. **One requester, one detector.** `MCi_IPID` low half varies only because it
   encodes the per-bank IP instance; its `HWID = 0x2E`, `MCTYPE = 0x2` fields decode
   to the kernel's `SMCA_CS_V2`, the **Coherent Station**. `MCi_MISC` is byte-identical
   across every reporting bank (`0xd01b0fff00000000`) and `MCi_SYND`'s high half is
   constant (`0x5b26____`). A single constant extended syndrome (`…00000000` ⇒
   XEC 0, the *address-translation-failed* class) reported identically by every
   memory-side port bank is the signature of **one shared address check** that every
   port fronts — not of a per-requester permission miss.

Banks 31–46 are the contiguous block of memory-side / coherency banks, and the
constant requester ID `0x1B` in `MCi_MISC[59:52]` is the fabric port that raised it.

## 2. Where that range lives in the hardware memory map

Two registers, both read this session without writing anything:

| register | value | meaning |
|---|---|---|
| `MSR 0xC001001A` (Top-of-Memory, low aperture) | `0x80000000` | low DRAM aperture ends at **2 GiB** |
| `/proc/iomem` (reverted boot) | `01000000-3fffffff : Reserved` | firmware declares the range **not System RAM** |
| ZCN/MP4 block `+0x020000/04/08` | `0x00000000 / 0x3FFFFFFF / 0x07FFFFFF` | a **base 0 / limit 0x3FFFFFFF** aperture (0–4 GiB) and a **0–128 MiB** aperture |

That last row is the answer to "why can't the GPU access it", in silicon rather than
in prose. The block's request-address aperture is programmed as
**base `0x00000000`, limit `0x3FFFFFFF`** — a window that *ends exactly at the top of
the reserved range* — with a second window ending at `0x07FFFFFF`. The reclaimed range
is not "ordinary RAM that the GPU happens to lack permission for". It is the
**top of a low device/stolen aperture** that the firmware froze, and the Coherent
Station's memory-side ports front that aperture.

`memmap=0x3ff00000$0x1000000` changes only the **CPU's page-table view** of that
aperture. It does not move the aperture. The Coherent Station still classifies those
system-physical addresses as non-DRAM and answers the refusal; the CPU's own accesses
succeed because the CPU path is the one whose view `memmap=` re-points, while the
GPU's request path is checked against the map the firmware froze.

## 3. Why the engine's own buffers could never have pointed there anyway

The engine's allocations are **host-backed**, so their GPU device addresses translate
to system-physical addresses taken from kernel page allocations:

- `runtime/include/ps5lm/vk.hpp:240` —
  `create(size, host_visible=true, want_extra, avoid)`; the engine's big buffers are
  bound with `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` and, at `flash_engine.hpp:364`,
  with the comment `// GTT` — i.e. the **GTT (translation) path**, whose page-table
  entries carry a target SPA derived from a kernel page.
- `flash_engine.hpp:155` — `pool_gib <= 0: auto-size from /proc/meminfo`, and
  `pool_gib()` is `n_slots_ * slot_bytes_`: the expert pool is sized from
  **MemTotal** (`flash_engine.hpp:3247`).

So two fences sit in series, and the S25 attempt crossed neither:

- **Fence 1 (allocator).** `memmap=` removes the range from the page allocator, so a
  GTT/device-local buffer can never be handed a page from it. The GPU therefore never
  issues a request to it — which is exactly why the earlier CPU-only read/write probe
  "worked": it went through `/dev/mem` and the CPU's own path, not through a GPU
  buffer.
- **Fence 2 (fabric).** Where the kernel *did* steer a request at that range, the
  Coherent Station refused it, faulting at addresses inside the range (§1).

## 4. Answer to the question as asked

The GPU has **no path to that range that bypasses the fabric's address check**. The
GPU never issues a CPU-style physical address: it issues a **device address**, its own
page tables translate that to a **system-physical address**, and that SPA is checked
against the DRAM aperture the firmware froze — the aperture whose limit register
reads `0x3FFFFFFF` and whose low DRAM top reads `0x80000000`. Reprogramming that is
not a GPU-side driver job: no amount of GPU page-table or GART-entry programming
changes the SPA the request carries once the page tables emit it. It is a
**memory-controller / fabric map** change, and that map sits behind the Data
Fabric's SMN indirect index/data pair, whose only writers on this kernel are the
firmware's own `amdtee` / `ps5_*` drivers.

That is also why my S25 note ("nothing writes to it, so it is free RAM") was wrong:
I had tested the **CPU's** view of the range and generalised it to the **GPU's**
request path. The two are gated by different hardware — the CPU by the page tables
`memmap=` edits, the GPU by the fabric aperture `memmap=` cannot edit.

## 5. What is still open, and what would settle it

The remaining unknown is **not** "is it fenced" (it is, twice over — §1–§2) and **not**
"who owns the fence" (the firmware does, §6a/§6e: the range is `device reserved` in the
firmware's own E820, and no driver in the tree owns the fabric DRAM base/limit). The single
remaining unknown is narrower: **whether that firmware-programmed aperture register is writable
from Linux at all**, i.e. whether it sits in a secure register space the PSP locks.

Two facts already bound that answer. First, the fence is **address-based**, so even a writable
aperture register would have to be reprogrammed on the **responder** side — the Coherent
Station's memory-side ports — and no GPU-side programming can substitute for it, because the
GPU's page tables emit the SPA the fabric then checks. Second, the only writer of that map on
this machine is the loader's boot-time command line (`ps5m2=34603008:512057409536:1000215216:…`),
which changes the CPU's view and nothing else.

To settle the residue without disturbing the serving engine, the next step is a **read-only
kernel module** that `ioremap`s the Data Fabric / UMC windows and `readl`s the DRAM base/limit
and the Coherent Station's access-check registers — reads only, no `writel`, no GPU command
submission, so the engine keeps serving untouched. Only if that survey finds a writable
base/limit pair does a reprogram experiment become worth proposing, and that experiment would
need an explicit engine pause.

## 6. Second-round measurements — the fence is firmware-owned, and the GPU's aperture is narrower than the range

Three further read-only measurements, engine still serving (PID 7776, elapsed 05:47:31 at
sampling; `ps5lm_server.py` 7773 and `ps5lm_dash.py` 1366 likewise untouched).

**(a) The firmware, not Linux, declares the range non-RAM.** The kernel's own boot log prints
the map it was *handed* by firmware, before any `memmap=` edit could apply:

```
BIOS-e820: [mem 0x0000000000100000-0x0000000000ffffff]  System RAM
BIOS-e820: [mem 0x0000000001000000-0x000000003fffffff]  device reserved
BIOS-e820: [mem 0x0000000040000000-0x000000005fffffff]  System RAM
```

The range is **sandwiched** between two System-RAM windows and labelled `device reserved` by
the firmware map itself. It is a carve-out, not a top-of-RAM trim. Linux then narrowed the low
aperture further (`e820: update [mem 0x80000000-0xffffffff] System RAM ==> device reserved`).
So the premise "we can hand that memory to Linux" was always a category error: the range was
never System RAM in the map Linux was given, and `memmap=` edits only Linux's *page-table view*
of a window the firmware already owns.

**(b) The GPU's translation aperture is narrower than the range and does not cover it.** The
driver prints the aperture it programs:

```
amdgpu 0000:20:00.0: GART: 512M 0x0000000000000000 - 0x000000001FFFFFFF
amdgpu 0000:20:00.0: [drm] PCIE GART of 512M enabled (table at 0x000000F400F00000).
```

The GPU's translation table covers device addresses `0x0–0x1FFFFFFF` and its table lives at
SPA `0xF400F00000` — inside the high GDDR carve-out, not in the reclaimed range. The range's
upper half (`0x20000000–0x3FFFFFFF`) lies **above** that aperture entirely, and its lower half
(`0x1000000–0x1FFFFFFF`) lies inside an aperture whose target addresses the driver builds from
kernel pages — which `memmap=` had already removed from the allocator (§3).

**(c) The engine's buffers are driver-managed GEM BOs in a GPU VMID space.** The read-only
debugfs window `dri/0/amdgpu_vm_info` opens with `pid:7776  Process:flash_decode` and lists
`Kernel PT/PDs:` and `Per VM BOs:` — the engine's allocations are buffer objects bound into a
GPU virtual-memory address space, exactly the path described in §3. Live module params confirm
the backing: `amdgpu.gttsize=14336` (14 GiB GTT) and `ttm.pages_limit=4194304` /
`ttm.page_pool_size=4194304` (16 GiB of struct pages) — the driver is deliberately configured to
back device memory with **system pages**, so a device address resolves to a kernel page.

**(d) Softened claim.** §2's ZCN/MP4 triple `0x00000000 / 0x3FFFFFFF / 0x07FFFFFF` is a raw
register survey whose field names are not yet matched to a manual; treat it as corroborating
shape, not as a decoded register. The load-bearing proof is (a) the firmware E820 declaring the
range `device reserved`, (b) the Coherent Station banks faulting on addresses inside it, and
(c) TOM1 `= 0x80000000`.

**(e) What Linux owns on the memory-controller side.** The console's kernel tree
`drivers/ps5/` contains `mp1.c`, `svm.c`, `spcie.c`, `tpcie.c`, `hdmi.c`, `ps5-cec.c`,
`fan.c`, `led/`, `buzzer.c`, `autoservo_param.h` — **no memory-controller, UMC, or Data Fabric
driver**. Nothing in the tree owns the fabric DRAM base/limit that would have to move. The only
writer of that map on this machine is the loader (`ps5-linux-loader`, `source/prepare_resume.c`),
which writes the kernel command line — a CPU-side view change, not a fabric reprogram.

## 7. Correction to the record

`docs/HARDWARE-DECONSTRUCTION.md` §"reserved RAM" should read:
the range is **not** ordinary DRAM that Linux declined to use; it is the top of a
firmware-frozen low aperture (`0x00000000–0x3FFFFFFF`, with a `0x00000000–0x07FFFFFF`
sub-window), fenced for all requesters at the Coherent Station. The claim "giving it
to Linux would grow the expert pool by ~9%" is therefore **withdrawn**: the pool is
sized from host RAM that the GPU can actually reach, and this range is not reachable.

## 8. Third-round read-only kernel probe -- supersedes parts of 2 and 6(d)

A read-only kernel module (`scripts/console/s28/s28_mce_ro.c`, built against
`/lib/modules/7.2.3/build`, insmod/rmmod round-trip clean) read the machine-check and
top-of-memory MSRs with `rdmsrq_safe()` only -- no wrmsr, no MMIO write, no GPU submission.
The serving engine was never interrupted (PID 7776 / 7773 observed running with elapsed time
growing monotonically across the whole probe).

Two findings **correct** sections 2 and 6(d) above; full record in `docs/S28-MCE-REGISTER-PROBE.md`:

1. **TOM1 = 0x80000000 is NOT the fence register.** 2 GiB sits ABOVE the range
   `0x01000000-0x3FFFFFFF` (16 MiB-1 GiB), so the range lies BELOW TOM1, inside CPU-visible
   DRAM. That is *why* the CPU-side /dev/mem probe succeeded, and why that success proved
   nothing about the GPU's request path. The claim that TOM1 fences the range is withdrawn.
2. **There is no TOM2 register to reprogram.** MSR `0xC00101EA` and `0xC00101EE` both
   returned `rd ok=-5` (-EIO from a #GP general-protection fault): those MSRs do not exist on
   this Van-Gogh-class core. The "reprogram TOM2 to expose the range" idea has no register behind it.
3. **The faulting banks are not in writable MSR space.** With the tree's own numbering
   (`MCx_STATUS = 0x401 + 4*bank`), bank 31 reads (`rd ok=0`) but **every bank 32..46
   faults** (`rd ok=-5`). The refusals therefore came from SMCA fabric-side banks behind the
   Data Fabric's SMN indirect window -- the same space with no driver in this kernel tree.
4. **Engine buffers confirmed kernel-page-backed** via read-only `dri/0/amdgpu_vm_info`
   (`pid:7776  Process:flash_decode` with `Kernel PT/PDs:` / `Per VM BOs:`) plus live
   params `amdgpu.gttsize=14336` and `ttm.pages_limit=4194304`.

Net: option (b) of the objective is answered with register-level evidence. The fence is an
address-keyed Coherent-Station check whose registers sit behind SMN, not in writable MSR space;
there is no TOM2 to move; and the engine's buffers could never have pointed at the range.
No driver written on the requester side can make that range GPU-visible.
