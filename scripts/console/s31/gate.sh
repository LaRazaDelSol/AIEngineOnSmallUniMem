#!/bin/bash
# S31 gate run, one server stop: build ps5lowmem + the engine (build-tier), load the module, copy-rate bench,
# smoke without / with the tier (md5 must match), restart ps5lm-server AND ps5lm-dash. Output: ~/ps5lm/s31/gate.log
#   bash gate.sh [SWAP_MIB=256]
SWAP=${1:-256}
S=/home/steam/ps5lm/src-tier
B=/home/steam/ps5lm/build-tier
L=/home/steam/ps5lm/s31/gate.log
mkdir -p /home/steam/ps5lm/s31
exec > >(tee $L) 2>&1
restart() { sudo systemctl reset-failed ps5lm-server ps5lm-dash 2>/dev/null; sudo systemctl start ps5lm-server; sudo systemctl start ps5lm-dash; echo "server $(systemctl is-active ps5lm-server) dash $(systemctl is-active ps5lm-dash)"; }
trap restart EXIT
echo "== stop"; sudo systemctl stop ps5lm-server ps5lm-dash
for i in $(seq 60); do pgrep -x flash_decode >/dev/null || break; sleep 1; done
echo "== module"; cd $S/scripts/console/s31 && make -s 2>&1 | grep -v "^make" | tail -5; cp ps5lowmem.ko lowmem_bench /home/steam/ps5lm/s31/
echo "== engine"; mkdir -p $B && cd $B
[ -f build.ninja ] || cmake -S $S/runtime -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DPS5LM_BUILD_BENCH=ON -DPS5LM_GPU_ARCH=gfx1030 \
  -DPS5LM_HIP=OFF -DPS5LM_NATIVE=ON -DPS5LM_SOFT_DOT4=OFF >/dev/null
ninja flash_decode ps5lm_shaders 2>&1 | grep -E ' error|FAILED|error:' | head -20
ninja flash_decode ps5lm_shaders >/dev/null 2>&1 && echo BUILD-OK || { echo BUILD-FAILED; exit 1; }
echo "== insmod"; lsmod | grep -q "^ps5lowmem " || sudo insmod /home/steam/ps5lm/s31/ps5lowmem.ko swap_mib=$SWAP
sudo dmesg | grep ps5lowmem | tail -2
if [ "$SWAP" -gt 0 ]; then echo N | sudo tee /sys/module/zswap/parameters/enabled >/dev/null; sudo udevadm settle; sudo mkswap -q /dev/ps5lowswap && sudo swapon -p 100 /dev/ps5lowswap; swapon --show; fi
sudo grep -i "ps5\|0x0000000001000000" /sys/kernel/debug/x86/pat_memtype_list 2>/dev/null | head -3
echo "== bench"; sudo /home/steam/ps5lm/s31/lowmem_bench 512 2048
sudo dmesg | grep -ciE "mce|machine check|hardware error" | sed 's/^/MCE lines in dmesg: /'
echo "== smoke"; cd $S/scripts/console
B=$B bash smoke.sh tier0
B=$B bash smoke.sh tier1 --tier /dev/ps5lowmem
grep -a "lowmem tier" ~/ps5lm/bench/sm_tier1.err
sudo dmesg | grep -ciE "mce|machine check|hardware error" | sed 's/^/MCE lines in dmesg: /'
echo "== done"
