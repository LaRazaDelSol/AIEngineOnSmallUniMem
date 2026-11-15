# Hardware deconstruction of the console (S25, 2026-11-10)

Every device the kernel can see on this PS5, what it is at the register level, what it could add to the engine, and
why. Sources: `recon/s25/` (full PCI configuration space, memory map, boot log, register surveys, RAM samples, all
taken on the console in this session), the PS5 driver sources of the running kernel (`~/kbuild/linux/drivers/ps5/`),
and earlier measurements cited by ledger number (R-numbers refer to the development ledger).

Marking: **[M]** measured here, **[S]** read in driver source, **[I]** inferred (the evidence is given), **[R..]** an
earlier measurement.

## 1. Result

| component | state under Linux | what it could add | verdict |
|---|---|---|---|
| **1,008 MiB of RAM at 0x01000000-0x3FFFFFFF** | marked reserved by the loader; holds the frozen PlayStation OS image; the CPU can read and write it, **the GPU cannot** [M] | not the expert pool. CPU-only uses (a RAM swap device) need a small block driver | **tried as general RAM on 2026-11-10 and reverted**: GPU access to it raises machine checks (section 3) |
| GPU, 40 CUs | all 40 compute units on (36 + 4 harvested, S15b) | nothing more: the clock is capped by the power loop | at the limit of this firmware [R48] |
| CPU, 8 Zen 2 cores | 3.5 GHz, 1 core used by the engine | 4% of the GPU's arithmetic at best, from the same power budget | control and I/O only (section 5) |
| M.2 SSD path | Gen4 x4 through the Sony bridge, drive at its rated 6.6 GB/s | a 7.3 GB/s-class drive: +8% (the uplink then caps at 7.1) | drive-bound, then uplink-bound (section 4) |
| internal 825 GB flash (CXD90062GG) | no driver, decoding off, holds the PS5 system software | at most +0.5 GB/s: it shares the M.2's uplink | not usable, and not worth a driver (section 4) |
| decompressor / I/O coprocessor (`ZCN/MP4`) | alive, two register pages, no driver | nothing: the pack has 7.9993 bits of entropy per byte | no input for it (section 6) |
| crypto coprocessor (CCP) | only its random-number register answers | AES / SHA / RSA / zlib engines: no multiply-accumulate datapath | not a compute device (section 6) |
| audio coprocessor (ACP / Tempest) | powered down: every register reads 0 | about 1% of the GPU even at full specification | no (section 6) |
| Salina SATA port | controller up, link down (nothing attached) | 0.55 GB/s if a drive could be attached inside the console | no |
| USB 10 Gbps ports (3) | working | +0.88 GB/s per NVMe enclosure, additive [R6] | needs a drive that is not the PS5-formatted SSD (`--pack-mirror` is built) |
| Ethernet / Wi-Fi | 1 Gbit/s, Wi-Fi 6 | 0.11 GB/s | no |
| 13 "unclassified" Sony functions | empty slot functions of the bridge | nothing behind them | section 2 |

The engine already uses the two devices that carry the workload (GPU and M.2) at 55-65% of the GPU's arithmetic peak
and 100% of the drive's rated bandwidth. Nothing else on the board can do matrix arithmetic, and the one block of
unused memory is fenced off from the GPU.

## 2. The bus, function by function [M]

```
00:01.1  root port, 16 GT/s x4 (Gen4), MaxPayload 256 of 512           the ONLY I/O link of the main chip
  40:00.0  Sony CXD90062GG, class 0180 (mass storage, not NVMe)        internal flash controller. Mem-, no driver.
  40:00.1  Salina PCIe Controller (driver tpcie, "Titania")            UART console at BAR4+0x1010100, MSI glue [S]
  40:00.2  Salina Ethernet (driver mts)            link 8 GT/s x1      1 Gbit/s
  40:00.3  Salina SATA AHCI                                             PHY init fails (-110): no port
  40:00.4  Salina SATA AHCI secondary (ahci_salina) 1 port, 6 Gbit/s   link down
  40:00.5  Salina glue (driver spcie)                                   ICC mailbox: fan, thermal, power, HDMI, LEDs [S]
  40:00.6  Salina xHCI                                                  USB 2 + Bluetooth
  40:00.7  NXP IW620 Wi-Fi 6 / BT 5.3              link 2.5 GT/s x1
  40:01.0 - 40:02.4, 40:02.6, 40:02.7  Sony 90f3-9102 (13 functions)   same serial number and link as 40:00.0, three
                                                                        BARs each (1M, 1M, 256 B) never assigned, Mem-
  40:02.5  WD SN810 NVMe (driver nvme)             link 16 GT/s x4      the M.2 slot
00:08.1  internal port, 16 GT/s x16
  20:00.0  GPU (amdgpu)   20:00.1  HDMI audio
  20:00.2  Cryptographic Coprocessor 1022:13ec     20:00.3  ZCN/MP4 1022:13ef
  20:00.4/.5  USB 3.1 Gen2 xHCI (2 Type-A, 1 Type-C)                    20:00.6  HD Audio Coprocessor 1022:13eb
```

What this says about the board:

- **Everything outside the main chip hangs off one Gen4 x4 link.** The flash controller chip presents itself as one
  multi-function endpoint (ARI) and forwards the devices behind it as its own functions: the southbridge (Salina, on a
  Gen3 x1 link of its own), Wi-Fi, and the M.2 slot. That is why the M.2 drive appears as function `40:02.5` in the
  middle of a row of Sony device IDs: `40:01.0`-`40:02.7` are sixteen slot functions of the bridge, the M.2 occupies
  one, the other thirteen are empty (identical capability lists, serial `00-00-00-00-04-00-01-02`, BARs unassigned,
  memory decoding off) [M, I]. There is no hidden device behind them to write a driver for.
- No function on either bus has a class code of a processing accelerator, and none exposes shared memory large enough
  to hold weights except the Salina controller's 512 MiB BAR2 (`a0000000`), which is an address window of the bridge,
  not memory [I: the driver maps only BAR4 and uses it for register glue].

## 3. The reserved gigabyte [M]

The memory map the loader hands to Linux marks `0x01000000-0x3FFFFFFF` (1,008 MiB) as reserved. Read through
`/dev/mem`:

| test | result |
|---|---|
| content (4,032 pages sampled, one per 256 KiB) | 29% all-zero pages, 24% high-entropy, the rest structured; strings such as `libSceSysmodule.sprx`, `SceSysAvControlIpcSrv`, a Sony build path |
| does anything write to it | 0 of 4,032 pages changed in 30 s and again in 90 s |
| is it writable RAM | 5 of 5 all-zero pages held a test pattern and were restored |
| is a hypervisor running from it | Linux itself is the SVM host: `kvm_amd` initialises, and `drivers/ps5/svm.c` executes `VMRUN` to run a firmware routine in a guest [S] |

It is ordinary DRAM holding the memory image of the PlayStation OS that was running when the loader took over. The
reservation is one hard-coded entry written by `ps5-linux-loader` (`source/prepare_resume.c`, found in S14), not a
property of the hardware. (The two smaller reserved ranges are different: `0x60000000` reads all ones, and
`0x62800000` holds the dormant hypervisor's code; they stay reserved.)

**S26 correction to the last sentence.** "Not a property of the hardware" is true of the *CPU's* view and false for the
GPU's. The range is the top of a low aperture that the firmware freezes in the fabric's request-address map, and that
aperture — not a permission table the GPU consults — is what refuses the access. Two registers, both read read-only on
the console, place the fence in silicon:

| register | value | what it fixes |
|---|---|---|
| `MSR 0xC001001A` (Top-of-Memory, low aperture) | `0x80000000` | low DRAM aperture ends at 2 GiB, so the range sits inside a window the CPU can cache but the GPU's request path treats as a device aperture |
| ZCN/MP4 block `+0x020000 / +0x020004 / +0x020008` | `0x00000000 / 0x3FFFFFFF / 0x07FFFFFF` | a **base 0 / limit 0x3FFFFFFF** request aperture ending exactly at the top of the reserved range, plus a `0–0x07FFFFFF` sub-window |

Full record set, register decode and the two fences in series: `docs/S26-GPU-RESERVED-RAM.md`.

Giving it to Linux needs no loader change: `memmap=exactmap` replaces the firmware map with an explicit list, and
`scripts/console/reclaim_lowmem.sh apply` writes that list (this console's own map with the one entry turned into RAM)
into `/boot/efi/cmdline.txt`. **Expected (withdrawn — see the S26 correction above):** MemTotal about 14.7 to 15.7 GB,
a pool about 500 slots (9%) larger, which at the measured 3% of decode speed per 0.5 GiB is about +6% decode and the
same amount of extra room for long contexts. That chain assumed the pool would grow, but the pool is sized from host
RAM the GPU can actually reach (`flash_engine.hpp:155` sizes it from `/proc/meminfo`, and its buffers are host-backed
GTT/device-local allocations), and this range is not reachable by the GPU — so handing it to Linux does not enlarge the
pool. The +9% pool / +6% decode figure is **withdrawn**; the measured pool growth on the failed boot (5,466→6,059
slots) bought nothing because the engine then died on the refused accesses.

### Applied on 2026-11-10: it does not work for the pool, reverted

The console booted on the new map (`user-defined physical RAM map`, 15.78 GB available at boot instead of 14.76) and
the engine sized a 12.25 GiB pool of 6,059 slots (11.05-11.22 GiB, 5,466-5,549 before). Then:

| observation | value |
|---|---|
| machine-check records in the first three minutes | about 2,600 lines, status `dc2030000002085b` (bus error, I/O, timeout), banks 31-46 |
| addresses in those records | all inside the reclaimed range (`0x1d27....`, `0x207b....`, `0x2209....`) |
| reference run | `radv/amdgpu: Not enough memory for command submission`, `vkQueueSubmit` failed, no output |
| served engine | 29 tok/s instead of 51-55, no draft accepted (0.000), then the engine died during a request |
| filesystem, NVMe, IOMMU messages | none |

So the range is real RAM for the CPU, but accesses from the GPU side are refused by the hardware: whatever fence the
PlayStation firmware put around its kernel's memory is still up, and it is not in the memory map. The earlier tests
(read, write, static content) were all CPU accesses and could not show this. The change was removed, the console
soft-rebooted onto the firmware map, the reference run prints PASS again, served speed is back to 51-55 tok/s, and
the one stored conversation and two turn files written during those minutes were deleted. Boot log of the attempt:
`recon/s25/dmesg-reclaim-attempt.txt`.

What is left of this lever: memory that only the CPU touches. The kernel has no pmem or RAM-disk driver built in, so
that means a small block driver over the reserved range (the range stays reserved; the driver maps it) used as swap.
The gain is indirect: the front end and the OS could be pushed there under pressure instead of being OOM-killed, and
the pool reserve could come back down from 512 MiB (about +130 slots). Not built. Lifting the fence itself would mean
finding which block enforces it (the records point at the GPU's side of the memory controller); nothing in the
running kernel's sources touches it.

## 4. Storage: why the internal drive and a new driver do not pay [M, R]

- Uplink capacity: 16 GT/s x 4 lanes x 128/130 = 7.88 GB/s on the wire. Read data travels as memory-write packets of
  at most 256 payload bytes with 24 bytes of framing, header and CRC: 91.4%, less about 1.5% for acknowledgements and
  clock compensation = **7.1 GB/s of payload**. Raising MaxPayload to 512 (root port, bridge function 0 and the drive
  all report it as supported) would make that 7.4 GB/s.
- The SN810 is rated 6,600 MB/s and measures 6,574 [R6] / 6.4 GB/s at expert-sized reads [R76]: **the drive runs at
  its own rating, 7% below the link.** A larger payload would therefore change nothing today; a 7.3 GB/s-class drive
  would gain 8% and then sit on the link.
- The internal flash controller is on the same uplink. Even with a perfect driver the two drives together cannot
  exceed 7.1 GB/s: at most +0.5 GB/s over the SN810 alone. The controller is proprietary (class 0180, 18 MSI-X vectors,
  three BARs, no public protocol), its contents are the PlayStation system software, and its memory decoding was left
  off: this session read its configuration space only and did not enable it.
- Bandwidth that IS additive: the USB controllers are functions of the main chip on the other root port. A USB 10 Gbps
  NVMe enclosure added 881 MB/s on top of the SN810 [R6]; `--pack-mirror` (S23) uses it. The drive that is attached
  there now is PS5-formatted storage and is never used.
- Neither decode nor prefill is waiting for the drive at the moment: with cache-aware routing the SSD is 30-58% busy
  during prefill (this session) and decode is GPU-bound (S24).

## 5. CPU: arithmetic and power [R, M]

Eight Zen 2 cores, two 256-bit FMA units each: 8 x 2 x 8 lanes x 2 x 3.5 GHz = 0.9 TFLOPS on paper, about 0.45 in a
real kernel, against 10.7 for the GPU at its sustained clock: 4%. The CPU and the GPU also share one power envelope
managed by the SMU (`drivers/ps5/mp1.c`: CPU 3.2-3.5 GHz, GPU 2.0-2.23 GHz [S]), and they share the 448 GB/s memory
bus. Measured in S15b [R48]: the GPU holds 2,099-2,125 MHz under full load instead of 2,230 because of that envelope;
the SMU refuses any target above 2,230; parking seven cores at 800 MHz returned 0.45% of GPU clock. So CPU matrix
kernels would add at most 4% and take part of it back from the GPU. The CPU's job stays what it is: routing tables,
io_uring, sampling, drafts from the prompt (PLD), and now the session store.

## 6. The three undriven functions of the main chip [M]

Read-only register survey (`scripts/console/hw_probe.py`: BARs mapped read-only through sysfs, first 0x200 bytes plus
the head of every page; no write of any kind). Output in `recon/s25/mmio-*.txt`.

| function | what answers | reading |
|---|---|---|
| `20:00.2` Cryptographic Coprocessor, 1 MiB BAR | offset 0x0C returns a different value on every read (also 0x98); every other register reads `ffffffff` | AMD CCP version 5 layout: 0x0C is the random-number output. The command-queue registers are not exposed to this side. Its engines are AES, SHA, RSA, ECC and zlib inflate: none multiplies matrices, and nothing here is encrypted or compressible |
| `20:00.3` ZCN/MP4, 1 MiB BAR, bus mastering left on, PASID capability | two pages: 0x20000 (`3fffffff 07ffffff ... 1`), 0x70000 (`00800020`) | a mailbox-style device that can write into process address spaces (PASID) and was left active by the PlayStation OS: consistent with the I/O coprocessor / decompressor path [I]. Unknown protocol, no driver |
| `20:00.6` HD Audio Coprocessor, 256 KiB BAR | nothing: all 64 pages read 0 | powered down |

Why none of them is worth a driver for this engine, at the hardware level:

- **Decompression has no input.** The expert pack measures 7.9993 bits of entropy per byte and grows under zstd and
  xz (measured in the session before this one). EXL3 trellis codes are packed to their bit rate by construction; the
  older scalar packs had 5.1% of order-0 slack (S15). A decompressor can only expand what an encoder removed.
- **The decompressor's data path does not pass our drive.** It sits between the console's own flash controller and
  memory. The M.2 is a plain NVMe function whose reads already land in GPU-visible pages with no copy (io_uring fixed
  buffers into the udmabuf slab): there is no copy or CPU cost left for a DMA engine to remove (host time is 0.07 ms
  per decoded token).
- **The audio coprocessor is small and off.** Sony describes the Tempest unit as one GPU-class compute unit without
  caches, about the arithmetic of eight Jaguar cores: roughly 0.1 TFLOPS, 1% of the GPU, behind the same memory bus,
  with no firmware loader and its registers powered down.
- The crypto coprocessor is a fixed-function block. Its only multiplier is a big-integer modular-exponentiation
  engine.

## 7. GPU: what a driver can and cannot change [R, M]

- Compute units: 2 shader engines x 2 arrays x 10 CUs = 40, all active (`amdgpu.ps5_cu_unlock=3`, S15b: +3%).
- Clock: SMU-owned. Target 2,230 MHz, 2,099-2,125 under load; requests above 2,230 are refused by the firmware; no
  thermal headroom is involved (the limit is a power loop) [R48]. `amdgpu` loads no power-management block for this
  chip, so there is no table to edit.
- Other engines in the chip (`dmesg`): VCN video encode/decode, two SDMA copy engines, the display core. Fixed
  function; SDMA only moves bytes.
- Where the arithmetic goes (this session, prefill, per token): MoE 0.92 ms, GatedDeltaNet layers 1.05 ms, attention
  layers 0.66 ms. The dense GEMMs run at 3.3 TFMA/s (61% of the sustained peak), the expert kernels at 55%. The
  attention kernel is the outlier at 0.83 TFMA/s, and it is bound by gathering K/V rows (about 78 GB/s of scattered
  reads), not by arithmetic or LDS: cutting its LDS reads 4x (S25, `w_attn.comp`) gave 5%.

## 8. What was not done

- The reserved gigabyte was applied and reverted (section 3). Which hardware block refuses the GPU's accesses was not
  identified.
- MaxPayload 512 was not tried: the drive is below the link, so there is nothing to gain today, and a wrong setting
  on the path of the root filesystem costs a cold boot.
- The internal flash controller was not enabled or mapped. `ZCN/MP4` and the CCP were read, never written.
- The Tempest/ACP figures are Sony's public description, not a measurement.
