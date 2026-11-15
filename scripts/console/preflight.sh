#!/bin/bash
# Console-side preflight (read-only unless --fix). Run before an engine start, and as ExecStartPre of ps5lm-server.
#   bash preflight.sh [--fix]
# Checks the conditions that have silently cost pool slots, corrupted a measurement, or hung the console:
#   boot parameters, leaked hugepages, tmpfs junk, a second engine, disk space, NVMe queue settings, the kernel config.
FIX=0; [ "${1:-}" = "--fix" ] && FIX=1
rc=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; rc=1; }

echo "== boot parameters"
CL=$(cat /proc/cmdline)
for p in kho=off udmabuf.size_limit_mb=16384 ttm.pages_limit=4194304 amdgpu.gttsize=14336 swiotlb=4096 init_on_alloc=0; do
  case " $CL " in *" $p "*) ok "$p";; *) bad "$p missing (sudo ps5lm-min cmdline; soft reboot)";; esac
done
case " $CL " in *" ps5lm.ai=1 "*) ok "AI mode";; *) warn "game mode (ps5lm.ai=1 absent): ~1.1 GB goes to gamescope + Steam";; esac

echo "== memory"
HP=$(awk '/HugePages_Total/{print $2}' /proc/meminfo)
if pgrep -x flash_decode >/dev/null; then
  ok "engine running (pid $(pgrep -x flash_decode | head -1)), $HP hugepages in use"
elif [ "$HP" -gt 0 ]; then
  if [ $FIX -eq 1 ]; then echo 0 > /proc/sys/vm/nr_hugepages && ok "released $HP leaked hugepages"
  else bad "$HP hugepages reserved with no engine (a killed engine leaked them): echo 0 > /proc/sys/vm/nr_hugepages"; fi
else ok "no leaked hugepages"; fi
TMP=$(du -sm /tmp 2>/dev/null | cut -f1)
[ "${TMP:-0}" -gt 64 ] && warn "/tmp holds ${TMP} MiB (tmpfs = pool memory, S17)" || ok "/tmp ${TMP:-0} MiB"
AV=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
pgrep -x flash_decode >/dev/null || { [ "$AV" -lt 12000 ] && warn "MemAvailable ${AV} MiB < 12000 with no engine (page cache counts until dropped): check what holds memory" || ok "MemAvailable ${AV} MiB"; }

echo "== storage"
ROOTDEV=$(findmnt -no SOURCE / | sed 's/p[0-9]*$//;s#/dev/##')
[ "$(cat /sys/block/$ROOTDEV/queue/scheduler 2>/dev/null | grep -o '\[.*\]')" = "[none]" ] && ok "$ROOTDEV scheduler none" || warn "$ROOTDEV scheduler is not none"
FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$FREE" -lt 20 ] && warn "only ${FREE} GiB free on /" || ok "${FREE} GiB free on /"
[ -c /dev/udmabuf ] && ok "/dev/udmabuf" || bad "/dev/udmabuf missing (CONFIG_UDMABUF)"

echo "== GPU"
CU=$(grep -o 'amdgpu.ps5_cu_unlock=[0-9]' /proc/cmdline | cut -d= -f2)
ok "harvested-CU unlock mode ${CU:-0} (3 = 40 CUs, AI mode only)"
exit $rc
