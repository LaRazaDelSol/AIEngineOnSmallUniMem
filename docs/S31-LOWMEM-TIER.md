# S31: the reserved low gigabyte as CPU-only swap + a second expert tier

The 1,008 MiB at `0x01000000-0x3FFFFFFF` (E820 `device reserved`) cannot be read by the GPU (S25: ~2,600 machine checks
when pool pages landed there). The CPU can read it at full speed. S31 uses it without ever giving it to the page
allocator. So no TTM, GTT, udmabuf or pool page can land there.

## What was built

| piece | file | what it does |
|---|---|---|
| kernel module | `scripts/console/s31/ps5lowmem.c` | `memremap(WB)` of the range; refuses to load if any of it is System RAM. `/dev/ps5lowswap` = block device over the first `swap_mib`, `/dev/ps5lowmem` = mmap of the rest |
| boot unit | `ps5/appliance/ps5lowmem.service` | insmod `swap_mib=256`, zswap off (it would compress into normal RAM), `swapon -p 100` |
| expert tier | `runtime/include/ps5lm/lowmem_tier.hpp` + `expert_streamer.hpp` | pool slots next in line for eviction (LFU) are copied into the tier while still resident (pinned for the copy). A pool miss whose expert is in the tier is filled by CPU memcpy instead of an SSD read. The tier stays exclusive of the pool: a filled entry becomes the next to overwrite |
| routing | `flash_engine.hpp` `route_select` | a tier expert earns the cache-aware routing bonus (`--tier-bias`, fraction of `--route-bias`, default 1) |

Flags: `--tier /dev/ps5lowmem[:MiB]`, `--tier-threads 4`, `--tier-window 64`, `--tier-min-score 0`,
`--tier-min-ctx 0`, `--tier-bias 1`. The tier only admits while decoding, never during a prefill.
Stats: CLI line `lowmem tier: ...`; serve `D` line `tier_per_tok`.

## Measurements (console, 2026-11-10)

Copy rate, 512 MiB, 2 MiB chunks (`lowmem_bench`). The tier is exactly as fast as normal RAM:

| threads | tier -> RAM | RAM -> tier | RAM -> RAM |
|---|---|---|---|
| 1 | 7.16 GB/s | 7.17 | 6.99 |
| 4 | 23.43 | 23.37 | 23.17 |
| 8 | 27.20 | 27.21 | 27.21 |

One 2 MiB chunk on one thread: 283 us. PAT reports the range `write-back`. No machine check in any run.

Engine (md5 = generated tokens; exact routing keeps them identical):

| run | tok/s off | tok/s on | notes |
|---|---|---|---|
| smoke, exact, 300 tokens, `--pool-gib 10.9` (6,279 slots both) | 45.25 | **45.97 (+1.6%)** | md5 `18179c0e1f51` PASS both; 4.1 fills/tok, 6.8% of misses, fill 129 us |
| 123k stored conversation, exact routing | 15.17 | **15.88 (+4.7%)** | md5 `af674ccc9682` both |
| 123k stored conversation, served flags | 25.23 | **30.32** | text differs (biased routing); SSD reads 5.5 -> 4.7/tok |

Bugs found on the way:
- The copy threads inherited the engine main thread's CPU 2 pin. Fills took 1.1 ms and decode lost 25%. The threads are
  now pinned to cores 4, 6, 8, 10.
- An admission filter on LFU score only removes hits. Fills per admit stayed at 0.11 for thresholds 0, 0.3 and 0.5, so
  the default admits everything.

Noise floor: under biased routing, two runs with the tier making **zero** copies differed by 3% (57.38 vs 55.77).
The text changes. So served-flag short-context A/Bs need many runs or the exact-routing instrument.

Side finding: the first engine after a configuration change can get fewer hugepages. The early reservation is sized from
the last run's footprint file, which is shared across configurations. One run got 4,979 of 5,324 pages (-6% slots).

## Corrections to S26-S29 (measured read-only this session)

- **TOM2 exists**: MSR `0xC001001D` = `0x0000000880000000`. S28 read `0xC00101EA`, which is not TOM2.
- **The faulting banks are in MSR space**: `MCG_CAP = 0x130` (48 banks). The SMCA registers sit at
  `0xC0002000 + 0x10*n`. Banks 31-46 have IPID `0x0002002e0000000{0..f}` (CS_V2, 16 instances). S28 used the legacy
  `0x401 + 4n` numbering, which ends at bank 31.
- **The IOMMU is in pass-through**: all IOMMU groups are `identity`, including the GPU. GPU DMA is not translated. S29's
  "fully IOMMU-translated" is wrong; the conclusion that the fabric sees the GPU's address unchanged holds.
- **BAR0 is the VRAM aperture** (256M). The registers are on BAR5 (512K). S29's "per-VMID page-table base array" was
  memory contents, probably page-table entries.
- **Data Fabric registers are in config space** (`setpci -s 00:18.x` reads them). They are not "unreachable from
  userspace". Writing them is still off limits.
- **The error class is Security Violation**: `MCi_STATUS 0xdc2030000002085b` gives XEC (bits 21:16) = 2, which the
  kernel's CS_V2 table calls "Security Violation". It also has UC=0, Deferred=1, UECC=1: poison was returned, which is
  why the S25 output was garbage. S26's "XEC 0" came from MISC. MISC holds threshold-counter settings, not a requester
  ID. Fault addresses are 256 B apart, not 4 KiB, consistent with channel interleaving across the 16 CS instances.

Verdict unchanged: the fence is a firmware/PSP security policy in every Coherent Station. It cannot be removed from
Linux.
