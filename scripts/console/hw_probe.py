"""Console-side hardware probe (S25): READ-ONLY register survey of the PCI functions that have no Linux driver.
    sudo python3 hw_probe.py [outdir]
For every function listed in TARGETS whose memory decoding is already enabled, the BAR is mapped read-only through
sysfs and sampled: the first 0x200 bytes in full, then the first 0x40 bytes of every 4 KiB page (register blocks start
on page boundaries). Nothing is written: no config-space change, no bus-master enable, no BAR write. Functions with
decoding disabled (the console's internal flash controller 40:00.0 and its empty slot functions) are NOT enabled:
only their configuration space is recorded (lspci does that).
"""
import mmap, os, struct, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
TARGETS = [  # (bdf, resource index, label)
    ("0000:20:00.2", 2, "ccp-13ec"),
    ("0000:20:00.3", 2, "zcn-mp4-13ef"),
    ("0000:20:00.6", 0, "acp-13eb"),
    ("0000:20:00.1", 0, "hda-13ea"),
]


def cfg16(bdf, off):
    with open(f"/sys/bus/pci/devices/{bdf}/config", "rb") as f:
        f.seek(off)
        return struct.unpack("<H", f.read(2))[0]


for bdf, idx, label in TARGETS:
    path = f"/sys/bus/pci/devices/{bdf}/resource{idx}"
    cmd = cfg16(bdf, 4)
    size = os.path.getsize(path)
    print(f"== {bdf} {label}: command 0x{cmd:04x} (mem decode {'on' if cmd & 2 else 'OFF'}, bus master {'on' if cmd & 4 else 'off'}), BAR{idx} {size >> 10} KiB")
    if not cmd & 2:
        print("   memory decoding is off: not touched")
        continue
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
    lines = []

    def rd(off):
        return struct.unpack_from("<I", mm, off)[0]

    live_pages = 0
    for page in range(0, size, 0x1000):
        span = 0x200 if page == 0 else 0x40
        vals = [rd(page + o) for o in range(0, span, 4)]
        if all(v in (0, 0xFFFFFFFF) for v in vals):
            continue
        live_pages += 1
        for o in range(0, span, 16):
            row = vals[o // 4: o // 4 + 4]
            if any(v not in (0, 0xFFFFFFFF) for v in row):
                lines.append(f"  {page + o:06x}: " + " ".join(f"{v:08x}" for v in row))
    print(f"   {live_pages} of {size >> 12} pages hold something other than 0 / ffffffff")
    with open(os.path.join(OUT, f"mmio-{label}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for l in lines[:24]:
        print(l)
    if len(lines) > 24:
        print(f"   ... {len(lines) - 24} more rows in mmio-{label}.txt")
    mm.close()
    os.close(fd)
