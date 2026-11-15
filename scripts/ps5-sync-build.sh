#!/bin/bash
# Sync runtime/ + the appliance files to the console and build there. Replaces runtime/tools/s17/sync.sh.
#   scripts/ps5-sync-build.sh [--stop-server] [--no-build] [ninja targets...]     (default targets: flash_decode ps5lm_shaders)
# Safety rules this encodes (each cost a session once):
#   - never build while an engine runs (it is OOM-killed): refuses unless --stop-server, which stops the unit first
#   - the console clock is not NTP-synced in AI mode: set it from the PC, and extract with tar -m, or ninja sees
#     future mtimes and loops on "manifest still dirty"
#   - the tar overwrites console-side edits of synced files: edit in the repo, never on the console
set -euo pipefail
. "$(dirname "$0")/lib.sh"
STOP=0; BUILD=1; TARGETS=()
for a in "$@"; do case "$a" in --stop-server) STOP=1;; --no-build) BUILD=0;; *) TARGETS+=("$a");; esac; done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(flash_decode ps5lm_shaders)

if engine_running; then
  if [ $STOP -eq 1 ]; then
    echo "stopping ps5lm-server (the engine saves its residency and releases the hugetlb slab)"
    ps5 'sudo systemctl stop ps5lm-server; for i in $(seq 30); do pgrep -x flash_decode >/dev/null || break; sleep 1; done; ! pgrep -x flash_decode >/dev/null'
  elif [ $BUILD -eq 1 ]; then
    echo "an engine is running on the console; pass --stop-server (or --no-build to sync only)" >&2; exit 1
  fi
fi
ps5 "sudo date -s @$(date +%s) >/dev/null"
cd "$REPO"
tar -cf - --exclude='__pycache__' --exclude='build*' runtime ps5/appliance scripts/console \
  | ps5 "mkdir -p $PS5_ROOT/src && cd $PS5_ROOT/src && tar -xmf -"
echo "synced"
[ $BUILD -eq 0 ] && exit 0
ps5 "cd $PS5_BUILD && cmake . >/dev/null && ninja ${TARGETS[*]} 2>&1 | grep -E ' error|FAILED|error:|warning: unused' | head -30; ninja ${TARGETS[*]} >/dev/null 2>&1 && echo BUILD-OK || { echo BUILD-FAILED; exit 1; }"
