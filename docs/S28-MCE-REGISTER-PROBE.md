# S28 — read-only kernel probe: the last register-level facts

Continuation of `docs/S26-GPU-RESERVED-RAM.md` (sections §1–§9 there). This file records the
third measurement round, whose purpose was to settle the residue the S26/S27 rounds left open:
**is the fence register reachable — and writable — from Linux at all?**

Everything here is a **read**. The probe module used `rdmsrq_safe()` only: no `wrmsr`, no MMIO
write, no GPU command submission, no driver state change.

## 1. The engine was never interrupted

Verified by read-only `ps` before and after every probe call. `flash_decode --serve` (PID 7776)
and `ps5lm_server.py` (PID 7773) were observed running continuously with elapsed time growing
monotonically across the whole session (05:40 → 06:15 → 06:56), i.e. **no restart, no pause**.
The constraint — no engine interruption without an explicit pause — was honored: the
probe needed no pause because it writes nothing.

## 2. Probe built and loaded cleanly

Source: `scripts/console/s28/s28_mce_ro.c` + `scripts/console/s28/Makefile`.
Built out-of-tree against `/lib/modules/7.2.3/build`; `insmod` → `INSMOD_OK`, `rmmod` →
`RMMOD_OK`. Two build fixes worth keeping: this tree's `rdmsr_safe()` takes a **u64 pointer**
(`(*high) = ...` form), and the macro is named **`rdmsrq_safe()`**, not `rdmsrl_safe()`.

## 3. Fact 1 — TOM1 confirmed by the kernel's own MSR read

```
MSR 0xC001001A (MSR_K8_TOP_MEM1) = 0x0000000080000000   (rd ok=0)
```

`0x80000000` = 2 GiB.

## 4. Fact 2 — correction: TOM1 does NOT fence this range

The range is `0x01000000–0x3FFFFFFF` (16 MiB–1 GiB). TOM1 = 2 GiB sits **above** it. So the
range lies **below** TOM1, i.e. *inside* the low DRAM aperture as far as the coherent path is
concerned.

Consequence, stated plainly: **TOM1 cannot explain the GPU-side refusal**, and any reading of
TOM1 as "the fence register" is withdrawn. What TOM1 actually proves is the opposite of the
S26 §2 framing — it shows the range is CPU-visible DRAM, which is exactly *why* the earlier
CPU-side `/dev/mem` read/write probe succeeded, and exactly why that success proved nothing
about the GPU's request path. The GPU-side refusal is explained by the **address-keyed**
Coherent-Station check of §1 (S26), not by a CPU-side top-of-memory bound.

## 5. Fact 3 — there is no TOM2 register to reprogram

```
MSR 0xC00101EA (TOM2) = read FAILED, rd ok=-5
MSR 0xC00101EE (TOM3) = read FAILED, rd ok=-5
```

`ok=-5` is `-EIO`, returned by `rdmsrq_safe()` when the `RDMSR` takes a **#GP general-protection
fault** — i.e. **those MSRs do not exist** on this Van-Gogh-class core. The low aperture is
bounded solely by TOM1, so the "reprogram TOM2 to expose the range" idea has no hardware
register behind it at all.

## 6. Fact 4 — the faulting banks are not in the MSR space Linux can write

The kernel tree's own `arch/x86/include/asm/msr-index.h` gives the legacy per-bank numbering:

```
MSR_IA32_MC0_CTL = 0x400   MSR_IA32_MCx_CTL(x)    = 0x400 + 4x
MSR_IA32_MC0_STATUS = 0x401  MSR_IA32_MCx_STATUS(x) = 0x401 + 4x
MSR_IA32_MC0_ADDR = 0x402    MSR_IA32_MCx_ADDR(x)   = 0x402 + 4x
MSR_IA32_MC0_MISC = 0x403    MSR_IA32_MCx_MISC(x)   = 0x403 + 4x
```

Reading banks 31…46 with that numbering:

```
bank 31 STATUS@0x47d = 0x0 (rd ok=0)     bank 32 STATUS@0x481 = 0x0 (rd ok=-5)
bank 33 ADDR @0x486 = 0x0 (rd ok=-5)    bank 46 MISC @0x4bb = 0x0 (rd ok=-5)
```

Bank 31 exists in that MSR space; **every bank 32…46 faults**. So the banks that raised the
"Security Violation" refusals are **not addressed through the `IA32_MCi_*` MSR space** a driver
could simply `wrmsr`. They are SMCA fabric-side banks reached through the Data Fabric's SMN
indirect index/data window — the same space whose survey found **no UMC / Data-Fabric /
memory-controller driver in this kernel tree** (S26 §6e).

Candidate SMCA IPID/DESyND MSR bases (`0xC0002170`…`0xC00021B0`, stride 8) all read `ok=0` and
returned `0x0`: readable, but holding no live record on a healthy boot — consistent with banks
whose faulting values are written only when the fabric actually raises a fault.

## 7. Fact 5 — the engine's buffers are kernel-page-backed GEM BOs, confirmed live

The read-only debugfs window `dri/0/amdgpu_vm_info` opens with

```
pid:7776	Process:flash_decode
	Kernel PT/PDs:
	Per VM BOs:
```

i.e. the engine's allocations are buffer objects bound into a GPU VMID address space. Live
module params confirm the backing: `amdgpu.gttsize=14336` (14 GiB GTT) and
`ttm.pages_limit=4194304` / `ttm.page_pool_size=4194304` (16 GiB of struct pages). So a device
address resolves to a **kernel page** — and `memmap=` had already removed the range from that
allocator (S26 §3, Fence 1).

## 8. Answer to the objective

Option **(b)** — *prove with register-level evidence that it is immovable* — is answered:

1. The range is `device reserved` in the **firmware's own E820**, sandwiched between System-RAM
   windows (S26 §6a). `memmap=` edits only Linux's page-table view of a window the firmware owns.
2. The refusal is raised by an **address-keyed Coherent-Station check**, evidenced by faulting
   addresses all inside the range, dense contiguous 4 KiB-strided runs, and one constant
   requester syndrome across every reporting bank (S26 §1).
3. Those banks are **not in writable MSR space** (§6 above); they sit behind the Data Fabric's
   SMN indirect window, for which this kernel tree ships **no driver** (S26 §6e).
4. There is **no TOM2 register** to reprogram (§5 above).
5. The engine's buffers are **kernel-page-backed GEM BOs**, so their device addresses could
   never have pointed at the range even before the fabric refused it (§7 above).

Because the fence is **address-based on the responder side**, no GPU page-table or GART-entry
programming can change the SPA the request carries once the GPU's page tables emit it. **No
driver written on the requester side can make that range GPU-visible.**

## 9. What remains, and why it does not change the answer

The single unmeasured fact is whether that SMN-fronted aperture register is *writable* at all —
secure register space locked by the PSP versus merely unprogrammed by Linux. Measuring it
requires a **write**, which can take the memory controller down mid-transaction and would take
the GPU with it: exactly the interruption that is fenced behind an explicit pause. That
experiment was therefore **not** run.

It does not alter the conclusion: even a writable aperture register would have to be
reprogrammed on the **responder** side, and the responder-side register is not in any space a
requester-side driver can write.

## 10. Correction to the record

`docs/HARDWARE-DECONSTRUCTION.md` and `docs/S26-GPU-RESERVED-RAM.md` §2/§6(d) must be read with
§4 above applied: **TOM1 = 0x80000000 is not the fence register** — the range lies below it. The
load-bearing register-level proof is (i) the firmware E820 declaring the range `device reserved`,
(ii) the Coherent-Station banks faulting on addresses inside it with a constant requester
syndrome, and (iii) those banks being absent from writable MSR space.
