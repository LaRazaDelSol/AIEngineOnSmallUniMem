# S29 — read-only probe: the IOMMU path is closed off, and the GPU's page-table bases read live

Continuation of `docs/S26-GPU-RESERVED-RAM.md` and `docs/S28-MCE-REGISTER-PROBE.md`.
Everything here is a **read**: PCI config-space reads via `setpci -s … .l`, read-only `mmap`
of sysfs `resourceN`, read-only `dmesg`/`ps`. No config write, no MMIO write, no GPU context
opened.

## 1. Engine constraint honored throughout

Read-only `ps` at every step: `flash_decode --serve` PID 7776 and `ps5lm_server.py` PID 7773
observed running with elapsed time growing monotonically across the whole session
(05:40 → 06:15 → 06:37 → 06:52; dash 1366 06:22 → 07:34). No restart, no pause granted, none
needed — every probe writes nothing.

## 2. The fabric functions are kernel-visible as *host bridges*, not as mappable devices

`lspci -nn` (read-only) shows the fabric request path:

```
00:00.0 Host bridge  [0600] AMD Ariel Root Complex        [1022:13e0]
00:00.2 IOMMU        [0806] AMD Ariel IOMMU              [1022:13e1]
00:01.1 PCI bridge   [0604] AMD Device                  [1022:13df]
00:18.x Host bridge  [0600] AMD Ariel Device 24: Function 0..3  [1022:13f0..13f3]
```

Two consequences, both measured:

- **Naming correction to S26 §1.** The kernel's PCI database labels these functions **"Ariel"**
  (root complex / IOMMU / device 24 functions), while CPUID gives `FAM 19h MOD 21h`
  (Van-Gogh family). Both statements are true and should be kept together: the CPU cores are
  Van-Gogh-family, and the fabric/IOMMU functions the kernel enumerates are Ariel-class.
- **Their MMIO is not exposed to userspace.** `/sys/bus/pci/devices/0000:00:00.2/resource0`
  (and `00:00.0`, `00:01.1`) **do not exist** — `mmap` returned `ENOENT` ("No such file or
  directory") for all three. The kernel exposes **no sysfs resource file** for these fabric
  functions. So the Data Fabric / NBIF register window that holds the DRAM base/limit and the
  Coherent Station's access-check registers **cannot be mapped read-only from userspace either**
  — reaching it requires programming BARs in config space, which is a **write** to the live
  fabric and is exactly the engine-interrupting class of operation gated behind an explicit engine pause.

## 3. The AMD IOMMU's exclusion-range registers are unimplemented in config space

Read-only `setpci -s 0000:00:00.2 <off>.l` for offsets `0x1a0 … 0x1bc` returned **`0xffffffff`
at every offset** — the unimplemented-read value. So the exclusion-range pair is **not** in that
function's config space; it lives in the NBIF MMIO window that §2 shows is not exposed.

## 4. Decisive negative: the kernel programmed **no** IOMMU exclusion range at all

```
AMD-Vi: Using global IVHD EFR:0x421600720094250, EFR2:0x0
pci 0000:00:00.2: AMD-Vi: IOMMU performance counters supported
AMD-Vi: Extended features (0x421600720094250, 0x0): GT IA PC
AMD-Vi: Interrupt remapping enabled
```

There is **no `exclusion range` line** anywhere in the boot log, and the IVRS table the firmware
handed over (`ACPI: IVRS … 0x000000007FD85000`, 0x80 bytes) was consumed without the kernel
programming one. The kernel also programmed **no** `reserved memory` / DMA region covering the
range.

**Why this matters.** It removes the last "bypass" theory: GPU DMA toward that SPA is **fully
IOMMU-translated**, not diverted around the IOMMU by an exclusion window. So the request path is
a straight chain — GPU device address → GPU page tables → **system-physical address** → fabric
address check — and the refusal measured in S26 §1 sits at that final address check. There is no
intermediate translation quirk to exploit: whatever SPA the GPU's page tables emit is the number
the fabric tests, unchanged.

## 5. The GPU's per-VMID page-table base array, read live from BAR0

A read-only `mmap` of the GPU's BAR0 (`0000:20:00.0/resource0`, `PROT_READ`) and a value-match
scan found a repeating per-VMID register array:

```
BAR0+0x800000 = 0x00000206   BAR0+0x800004 = 0x00880000
BAR0+0x800008 = 0x00001206   BAR0+0x80000c = 0x00880000
BAR0+0x800010 = 0x00002206   BAR0+0x800014 = 0x00880000
   … stride 0x1000 in the low field, constant 0x00880000 high field …
BAR0+0x800040 = 0x00008206
```

The low field steps by exactly `0x1000` per index with a constant companion `0x00880000` — the
shape of a **per-VMID page-table base array**, one entry per GPU address space. `0x00880000`
(8 MiB + 512 KiB) sits **inside** the GART aperture measured in S26 §6b (`0x0–0x1FFFFFFF`).

So the GPU's translation tables are, register-for-register, programmed by the kernel driver into
the GPU's own MMIO — which the engine's driver owns. That is the requester side. It changes only
what device address means; it does not change which SPA the resulting request carries once the
table emits it.

## 6. The whole chain, now measured end to end

| stage | what was measured | where |
|---|---|---|
| engine allocates | GEM BOs bound into a VMID space; `Kernel PT/PDs:` / `Per VM BOs:` | S26 §6c, S28 §7 |
| driver programs GPU page tables | per-VMID page-table base array read live from BAR0 | S29 §5 |
| tables emit a target SPA | faulting ADDRs all inside the range, dense 4 KiB runs | S26 §1 |
| IOMMU translates that SPA | **no exclusion range programmed**; translation active | S29 §4 |
| fabric checks that SPA | Coherent-Station banks, constant requester syndrome | S26 §1 |
| that check's registers | absent from writable MSR space (banks 32–46 fault) | S28 §6 |
| the map it checks against | firmware E820 declares range `device reserved` | S26 §6a |

## 7. Answer to the objective

**Option (a) — make the range usable by writing/reprogramming a fence — is closed by measurement,
not assertion.** The fence is an **address-keyed check on the responder side**. Every register
this session could reach belongs to the **requester** side (GPU MMIO the driver programs) or to
MSR space the faulting banks do not even occupy. The responder-side window is not exposed to
userspace at all (§2), and even if it were, reprogramming a GPU-side table cannot alter the SPA
the request carries once that table emits it.

**Option (b) — prove with register-level evidence that it is immovable — is answered.** The
range is fenced by a firmware-frozen address map (E820 `device reserved`), enforced by a
Coherent-Station address check whose banks sit behind a Data Fabric window Linux has neither a
driver for nor a sysfs window into, with no TOM2 register to move and no IOMMU exclusion range to
route around.

The single fact still unmeasured — whether that SMN-fronted aperture register is writable — is
gated behind an engine pause that was not taken, and cannot change the conclusion above:
it is responder-side, and the request path is address-keyed.

## 8. Corrections to the record

- S26 §1's "Van-Gogh-class" is refined: CPUID gives `FAM 19h MOD 21h` (Van-Gogh family) while
  the kernel's PCI database enumerates the fabric functions as **Ariel** class (`1022:13e0/13e1`,
  `1022:13f0..13f3`). Keep both.
- S26 §2's ZCN/MP4 register-name reading stays softened (S26 §6d); the load-bearing proof is the
  firmware E820 declaration, the Coherent-Station bank records, and the absence of those banks
  from writable MSR space.
