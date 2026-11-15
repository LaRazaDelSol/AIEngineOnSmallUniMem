#!/bin/bash
# S31 console side: build ps5lowmem.ko + lowmem_bench, install the unit. Run from ~/ps5lm/src-tier/scripts/console/s31.
#   bash install.sh            build + copy the module to ~/ps5lm/s31 + install ps5lowmem.service (not started)
#   bash install.sh --start    ... and start it now (insmod + swap)
set -euo pipefail
cd "$(dirname "$0")"
make -s
mkdir -p /home/steam/ps5lm/s31
cp ps5lowmem.ko lowmem_bench /home/steam/ps5lm/s31/
sudo install -m 0644 ../../../ps5/appliance/ps5lowmem.service /etc/systemd/system/ps5lowmem.service
sudo systemctl daemon-reload
sudo systemctl enable ps5lowmem.service >/dev/null
echo "installed: /home/steam/ps5lm/s31/ps5lowmem.ko, ps5lowmem.service (enabled)"
if [ "${1:-}" = "--start" ]; then
  sudo systemctl start ps5lowmem.service
  dmesg | grep ps5lowmem | tail -2
  swapon --show
  cat /sys/module/ps5lowmem/parameters/tier_bytes
fi
