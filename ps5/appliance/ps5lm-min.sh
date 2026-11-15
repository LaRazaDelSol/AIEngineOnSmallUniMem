#!/usr/bin/env bash
# ps5lm-min — put the PS5 Linux console into (and back out of) maximum-memory inference mode.
#
#   sudo ps5lm-min on        free everything the engine can use (runtime only, reversible)
#   sudo ps5lm-min off       restore the normal console (gaming session, desktop services, sysctls, modules)
#   sudo ps5lm-min status    memory budget + what is still running
#   sudo ps5lm-min cmdline   add the boot parameters that runtime cannot change, to /boot/efi/cmdline.txt
#                            (backup kept; takes effect on the next soft reboot = kexec)
#   sudo ps5lm-min install   install this script + the AI-mode units (ps5lm-min.service, ps5lm-server.service,
#                            getty@tty1 drop-in, /etc/ps5lm/server.env) from the directory this script lives in
#   sudo ps5lm-min boot ai   next soft reboot comes up in AI mode: ps5lm.ai=1 on the cmdline -> no tty1 gaming
#                            session, `ps5lm-min on` at boot, the server on :8081 (the units are inert without it)
#   sudo ps5lm-min boot game next soft reboot is the normal console again
#   NOTE: a COLD boot reads the cmdline.txt on the loader USB, not /boot/efi: it comes up in whatever mode that file
#         says (game mode unless ps5lm.ai=1 is added there too). Soft reboots (kexec) use /boot/efi/cmdline.txt.
#         In AI mode `ps5lm-min off` restores the services but not the tty1 gaming session (its drop-in condition
#         keeps it off for the whole boot): use `boot game` + a soft reboot to get the console back.
#
# Replaces ai-min (which killed the session once; tty1 autologin restarted gamescope + Steam seconds later).
# Every number below was measured on this console (S14, 2026-11-06):
#   * KHO (CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y) keeps 1,512 MiB as MIGRATE_CMA scratch: only movable pages may
#     live there, so the long-term-pinned udmabuf/io_uring expert slab cannot -> `kho=off` (cmdline).
#   * getty@tty1 autologin -> /etc/profile.d/ps5-tty-session.sh -> gamescope + Steam (~1.1 GB).
#   * vm.min_free_kbytes 66 MiB + watermark_boost_factor 15000 (boost 142 MB seen in the OOM report).
#   * NetworkManager runs both links; systemd-networkd manages nothing.
# Never touched: sshd, NetworkManager, wpa_supplicant, wifi-ensure, journald, udevd, logind, dbus, ps5fan, ps5boost,
# the serial getty (rescue console).
set -uo pipefail

STATE=/run/ps5lm-min            # tmpfs: runtime state dies with the boot, exactly like the changes it undoes
CMDLINE=/boot/efi/cmdline.txt
USER_NAME=steam

SYSTEM_UNITS=(sddm.service getty@tty1.service bluetooth.service udisks2.service upower.service polkit.service
              systemd-homed.service systemd-userdbd.service systemd-userdbd.socket systemd-nsresourced.service
              systemd-nsresourced.socket systemd-networkd.service systemd-networkd.socket systemd-networkd-varlink.socket
              systemd-networkd-resolve-hook.socket systemd-networkd-varlink-metrics.socket systemd-timesyncd.service
              systemd-coredumpd.service systemd-coredump.socket ModemManager.service cups.service avahi-daemon.service
              power-profiles-daemon.service ananicy-cpp.service systemd-oomd.service)
USER_UNITS=(pipewire.socket pipewire-pulse.socket pipewire.service pipewire-pulse.service wireplumber.service
            xdg-desktop-portal.service xdg-document-portal.service xdg-permission-store.service at-spi-dbus-bus.service
            ydotool.service)
SESSION_PROCS=(gamescope gamescope-wl gamescopereaper steam steamwebhelper Xwayland plasmashell kwin_wayland
               baloo_file kded6 ksmserver)
MODULES=(rfcomm bnep hidp btusb btrtl btintel btbcm btmtk bluetooth
         snd_hda_codec_atihdmi snd_hda_codec_hdmi snd_hda_intel snd_hda_codec snd_hda_core snd_intel_dspcfg
         snd_hwdep snd_seq snd_seq_device snd_hrtimer snd_pcm snd_timer snd soundcore kvm_amd kvm irqbypass joydev)
# key=value: the appliance values. The originals are saved on `on` and restored on `off`.
SYSCTLS=(vm.min_free_kbytes=16384 vm.watermark_boost_factor=0 vm.watermark_scale_factor=10 vm.compaction_proactiveness=0
         vm.stat_interval=10 vm.swappiness=10 vm.overcommit_memory=1 vm.zone_reclaim_mode=0 kernel.nmi_watchdog=0
         kernel.timer_migration=0 kernel.sched_autogroup_enabled=0 kernel.numa_balancing=0 kernel.printk="3 3 3 3"
         vm.hugetlb_optimize_vmemmap=1)
# swiotlb: the 64 MiB default bounce pool peaked at 277 of 32,768 slabs (~0.5 MiB) on this console -> 4096 slabs = 8 MiB
BOOT_PARAMS=(kho=off udmabuf.size_limit_mb=16384 nowatchdog nmi_watchdog=0 init_on_alloc=0 init_on_free=0 swiotlb=4096)
# units that socket/dbus activation brings straight back: masked for this boot only (/run), unmasked by `off`
RUNTIME_MASK=(systemd-homed.service systemd-coredumpd.service systemd-networkd.service systemd-userdbd.service)

need_root() { [ "$(id -u)" = 0 ] || { echo "run as root (sudo $0 $*)"; exit 1; }; }
log() { printf '%s\n' "$*"; }
uctl() { systemctl --machine="${USER_NAME}@" --user "$@" 2>/dev/null; }

kib() { awk -v k="$1:" '$1==k{print $2}' /proc/meminfo; }
zone_cma_mib() { awk '/^Node/{z=$4} / cma /{s+=$2} END{printf "%d", s*4/1024}' /proc/zoneinfo; }
vram() { local d=/sys/class/drm/card0/device; [ -r $d/mem_info_vram_total ] && echo "$(($(cat $d/mem_info_vram_used)>>20))/$(($(cat $d/mem_info_vram_total)>>20)) MiB" || echo n/a; }

budget() {
    local avail cma
    avail=$(kib MemAvailable); cma=$(kib CmaFree)
    log "memory: MemTotal $(( $(kib MemTotal)>>10 )) MiB  MemAvailable $((avail>>10)) MiB  CmaFree $((cma>>10)) MiB (zone CMA $(zone_cma_mib) MiB)"
    log "        pinnable now ~ $(( (avail-cma)>>10 )) MiB  (engine: ~2.6 GiB dense weights + the rest = expert pool)"
    log "        VRAM carve $(vram)   hugepages 2M: $(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)"
    log "        udmabuf limit $(cat /sys/module/udmabuf/parameters/size_limit_mb 2>/dev/null) MiB   kho: $(tr ' ' '\n' </proc/cmdline | grep -c '^kho=off' | sed 's/1/off (good)/;s/^0$/ON (1.5 GiB CMA scratch!)/')"
}

cmd_status() {
    budget
    local running=()
    for u in "${SYSTEM_UNITS[@]}"; do systemctl is-active -q "$u" 2>/dev/null && running+=("$u"); done
    log "system units still active: ${running[*]:-none}"
    running=()
    for u in "${USER_UNITS[@]}"; do uctl is-active -q "$u" && running+=("$u"); done
    log "user units still active:   ${running[*]:-none}"
    running=()
    for p in "${SESSION_PROCS[@]}"; do pgrep -x "$p" >/dev/null && running+=("$p"); done
    log "session processes:         ${running[*]:-none}"
    running=()
    for m in "${MODULES[@]}"; do [ -d /sys/module/$m ] && running+=("$m"); done
    log "modules loaded (of list):  ${running[*]:-none}"
    log "top userspace RSS (MiB):"
    ps -eo rss=,comm= --sort=-rss | head -6 | awk '{printf "        %6.1f %s\n", $1/1024, $2}'
    log "kernel: Slab $(( $(kib Slab)>>10 )) MiB  PageTables $(( $(kib PageTables)>>10 )) MiB  Shmem $(( $(kib Shmem)>>10 )) MiB  Cached $(( $(kib Cached)>>10 )) MiB"
    [ -d "$STATE" ] && log "ps5lm-min: ON since $(cat $STATE/since 2>/dev/null)" || log "ps5lm-min: off"
    log "boot mode: this boot $(grep -qw 'ps5lm.ai=1' /proc/cmdline && echo AI || echo game), next soft reboot $(grep -qw 'ps5lm.ai=1' "$CMDLINE" 2>/dev/null && echo AI || echo game); server $(systemctl is-active ps5lm-server.service 2>/dev/null)"
}

cmd_on() {
    need_root on
    mkdir -p "$STATE"
    [ -f "$STATE/since" ] || date -Is > "$STATE/since"
    local before; before=$(kib MemAvailable)

    # 1. the display session first (tty1 autologin respawns gamescope + Steam unless getty@tty1 is stopped)
    : > "$STATE/units"
    # two passes, sockets first: socket activation (networkd, coredump, homed via userdbd) restarts a stopped service
    for pass in 1 2; do
        for u in $(printf '%s\n' "${SYSTEM_UNITS[@]}" | grep '\.socket$') $(printf '%s\n' "${SYSTEM_UNITS[@]}" | grep -v '\.socket$'); do
            if systemctl is-active -q "$u" 2>/dev/null; then
                grep -qx "$u" "$STATE/units" || echo "$u" >> "$STATE/units"
                systemctl stop "$u" 2>/dev/null
            fi
        done
    done
    for p in "${SESSION_PROCS[@]}"; do pkill -x "$p" 2>/dev/null; done
    sleep 1
    for p in "${SESSION_PROCS[@]}"; do pkill -9 -x "$p" 2>/dev/null; done

    : > "$STATE/masked"
    for u in "${RUNTIME_MASK[@]}"; do
        systemctl mask --runtime "$u" >/dev/null 2>&1 && echo "$u" >> "$STATE/masked"
        systemctl stop "$u" 2>/dev/null
    done

    # 2. user-session services of the console user (audio, portals, ydotool)
    : > "$STATE/user_units"
    for u in "${USER_UNITS[@]}"; do
        if uctl is-active -q "$u"; then echo "$u" >> "$STATE/user_units"; uctl stop "$u"; fi
    done

    # 3. kernel modules nobody uses during inference (bluetooth, audio, KVM, joystick)
    : > "$STATE/modules"
    # repeat until nothing more unloads (a codec module only reaches refcnt 0 after the driver that holds it)
    local changed=1
    while [ $changed = 1 ]; do
        changed=0
        for m in "${MODULES[@]}"; do
            if [ -d /sys/module/$m ] && [ -f /sys/module/$m/refcnt ] && [ "$(cat /sys/module/$m/refcnt)" = 0 ]; then
                modprobe -r "$m" 2>/dev/null && { echo "$m" >> "$STATE/modules"; changed=1; }
            fi
        done
    done

    # 4. sysctls (originals saved once)
    if [ ! -f "$STATE/sysctl.orig" ]; then
        for kv in "${SYSCTLS[@]}"; do k=${kv%%=*}; echo "$k=$(sysctl -n "$k" 2>/dev/null)" >> "$STATE/sysctl.orig"; done
    fi
    for kv in "${SYSCTLS[@]}"; do sysctl -q -w "$kv" >/dev/null 2>&1; done

    # 5. engine prerequisites: udmabuf may export a 16 GiB slab; boost + fan on
    [ -w /sys/module/udmabuf/parameters/size_limit_mb ] && echo 16384 > /sys/module/udmabuf/parameters/size_limit_mb
    systemctl start ps5boost.service ps5fan.service 2>/dev/null

    # 6. give memory back: flush + drop page cache + compact (contiguous memory for the slab / hugepages)
    sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
    journalctl --vacuum-size=16M >/dev/null 2>&1

    local after; after=$(kib MemAvailable)
    log "ps5lm-min on: MemAvailable $((before>>10)) -> $((after>>10)) MiB (+$(((after-before)>>10)) MiB)"
    budget
    grep -q '^kho=off$' <(tr ' ' '\n' </proc/cmdline) || log "NOTE: KHO scratch still holds $(zone_cma_mib) MiB the pinned pool cannot use: run 'ps5lm-min cmdline' + soft reboot"
}

cmd_off() {
    need_root off
    [ -d "$STATE" ] || { log "ps5lm-min is not on"; exit 0; }
    if [ -f "$STATE/sysctl.orig" ]; then while IFS= read -r kv; do sysctl -q -w "$kv" >/dev/null 2>&1; done < "$STATE/sysctl.orig"; fi
    echo 0 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null
    [ -f "$STATE/modules" ] && tac "$STATE/modules" | while read -r m; do modprobe "$m" 2>/dev/null; done
    [ -f "$STATE/masked" ] && while read -r u; do systemctl unmask --runtime "$u" >/dev/null 2>&1; done < "$STATE/masked"
    [ -f "$STATE/user_units" ] && while read -r u; do uctl start "$u"; done < "$STATE/user_units"
    # restart system units in reverse stop order; getty@tty1 last so the gaming session comes back on top
    [ -f "$STATE/units" ] && tac "$STATE/units" | grep -v '^getty@tty1' | while read -r u; do systemctl start "$u" 2>/dev/null; done
    grep -q '^getty@tty1' "$STATE/units" 2>/dev/null && systemctl start getty@tty1.service
    rm -rf "$STATE"
    log "ps5lm-min off: console restored"
    budget
}

write_cmdline() {  # $1 = new command line: backup + atomic replace + kexec re-stage
    local cur new="$1"
    cur=$(tr -d '\r\n' < "$CMDLINE")
    if [ "$new" = "$cur" ]; then log "cmdline unchanged: $cur"; return 0; fi
    local bak="$CMDLINE.pre-ps5lm-min.$(date +%Y%m%d-%H%M%S)"
    cp -p "$CMDLINE" "$bak"
    printf '%s\n' "$new" > "$CMDLINE.tmp" && sync && mv "$CMDLINE.tmp" "$CMDLINE" && sync
    log "backup: $bak"
    log "old: $cur"
    log "new: $(cat "$CMDLINE")"
    # ps5-kexec-load staged the kexec image (with the OLD command line) at boot: stage it again
    if systemctl cat ps5-kexec-load.service >/dev/null 2>&1; then
        systemctl restart ps5-kexec-load.service && log "kexec image re-staged with the new command line"
    fi
    log "takes effect on the next soft reboot (kexec): sudo systemctl reboot"
}

cmd_boot() {
    need_root boot
    [ -f "$CMDLINE" ] || { log "$CMDLINE not found"; exit 1; }
    local cur new u
    cur=$(tr -d '\r\n' < "$CMDLINE")
    new=$(printf '%s' "$cur" | sed -E 's/(^| )ps5lm\.ai=[^ ]*//g; s/(^| )amdgpu\.ps5_cu_unlock=[^ ]*//g')
    case "${1:-}" in
        ai)
            for u in ps5lm-min.service ps5lm-server.service; do
                systemctl is-enabled -q "$u" 2>/dev/null || { log "$u not installed/enabled: run 'sudo ps5lm-min install' first"; exit 1; }
            done
            new="$new ps5lm.ai=1"
            # S15b: harvested-CU unlock in AI mode only (no display needed there; a black screen is possible).
            # PS5LM_CU_UNLOCK in /etc/ps5lm/server.env: 0 off, 1 probe, 2 SE0/SH0, 3 all, 4 probe-all (amdgpu param).
            # Recovery: a cold boot reads the loader stick's own cmdline, which never carries this parameter.
            local cu
            cu=$(sed -n 's/^PS5LM_CU_UNLOCK=\([0-9]\).*/\1/p' /etc/ps5lm/server.env 2>/dev/null | tail -1)
            [ -n "$cu" ] && [ "$cu" != 0 ] && new="$new amdgpu.ps5_cu_unlock=$cu" ;;
        game) ;;
        *) log "usage: ps5lm-min boot ai|game  (next soft reboot: $(grep -qw 'ps5lm.ai=1' <<<"$cur" && echo ai || echo game))"; exit 2 ;;
    esac
    write_cmdline "$new"
}

cmd_install() {
    need_root install
    local src f
    src=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
    for f in ps5lm-min.sh ps5lm-min.service ps5lm-server.service ps5lm-dash.service getty-tty1-ps5lm-ai.conf etc-ps5lm/server.env; do
        [ -f "$src/$f" ] || { log "missing $src/$f (run install from the repo copy: sudo bash ps5/appliance/ps5lm-min.sh install)"; exit 1; }
    done
    install -m 0755 "$src/ps5lm-min.sh" /usr/local/sbin/ps5lm-min
    install -m 0644 "$src/ps5lm-min.service" "$src/ps5lm-server.service" "$src/ps5lm-dash.service" /etc/systemd/system/
    install -d /etc/systemd/system/getty@tty1.service.d /etc/ps5lm
    install -m 0644 "$src/getty-tty1-ps5lm-ai.conf" /etc/systemd/system/getty@tty1.service.d/zz-ps5lm-ai.conf
    [ -f /etc/ps5lm/server.env ] || install -m 0644 "$src/etc-ps5lm/server.env" /etc/ps5lm/server.env
    systemctl daemon-reload
    systemctl enable ps5lm-min.service ps5lm-server.service ps5lm-dash.service
    log "installed. Both units are inert until 'sudo ps5lm-min boot ai' + a soft reboot."
}

cmd_cmdline() {
    need_root cmdline
    [ -f "$CMDLINE" ] || { log "$CMDLINE not found (is the boot FAT mounted at /boot/efi?)"; exit 1; }
    local cur new p
    cur=$(tr -d '\r\n' < "$CMDLINE")
    new="$cur"
    for p in "${BOOT_PARAMS[@]}"; do
        local key=${p%%=*}
        if printf '%s\n' $new | grep -q "^${key}="; then new=$(printf '%s' "$new" | sed -E "s/(^| )${key}=[^ ]*/\\1${p}/");
        elif ! printf '%s\n' $new | grep -qx "$p"; then new="$new $p"; fi
    done
    if [ "$new" = "$cur" ]; then log "cmdline already has: ${BOOT_PARAMS[*]}"; return 0; fi
    write_cmdline "$new"
}

case "${1:-status}" in
    on) cmd_on ;;
    off) cmd_off ;;
    status) cmd_status ;;
    cmdline) cmd_cmdline ;;
    boot) cmd_boot "${2:-}" ;;
    install) cmd_install ;;
    *) echo "usage: $0 on|off|status|cmdline|install|boot ai|game"; exit 2 ;;
esac
