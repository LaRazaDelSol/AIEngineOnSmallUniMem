#!/bin/bash
# Shared by scripts/ps5-*.sh (PC side). Sources scripts/console.env if present, else the defaults below.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PS5_HOST=steam@ps5.local   # set your console in scripts/console.env
PS5_KEY=.ssh/ps5_ed25519
PS5_ROOT='~/ps5lm'
PS5_BUILD='~/ps5lm/build14'
[ -f "$REPO/scripts/console.env" ] && . "$REPO/scripts/console.env"
ps5() { ssh -o BatchMode=yes -o ConnectTimeout=8 -i "$REPO/$PS5_KEY" "$PS5_HOST" "$@"; }
# The engine pins ~12 GiB with oom_score_adj=1000: a compiler or a second engine next to it gets it OOM-killed (S17).
# Match by exact process name (pgrep -x), never `pkill -f`: -f also matches the SSH command line itself (S14).
engine_running() { ps5 'pgrep -x flash_decode >/dev/null'; }
