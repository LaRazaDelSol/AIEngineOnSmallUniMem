#!/bin/bash
# S28: read-only. Get the AUTHORITATIVE decode so a probe module can name the
# faulting IP block and requester exactly:
#  (1) does the tree carry the kernel's SMCA MCE decode tables?
#  (2) what files does /sys/firmware/memmap actually expose?
#  (3) confirm the engine is still running untouched.

echo "=== kernel tree: x86 MCE decode sources ==="
find ~/kbuild/linux/arch/x86 -path "*mce*" -type f 2>/dev/null | sed -n "1,12p"
ls ~/kbuild/linux/arch/x86/kernel/cpu/mce/ 2>/dev/null | tr "\n" " "; echo
ls ~/kbuild/linux/arch/x86/include/asm/ 2>/dev/null | grep -iE "mce|scan|mtrr" | tr "\n" " "; echo

echo; echo "=== SMCA bank-type table: global bank number -> IP block ==="
grep -rn "SMCA_CS_V2\|SMCA_CS_\|smca_bank_types\|smca_get_bank_type\|MCE_GENDEADDR\|SMCA_UMC_V2\|SMCA_MPAM\|hwid_to_mca_type\|MCA_HWID_CS" ~/kbuild/linux/arch/x86 2>/dev/null | sed -n "1,20p"

echo; echo "=== MSR numbering for the MCE banks (so a probe reads the right MSRs) ==="
grep -rn "MSR_MCG_CAP\|MSR_MCG_STATUS\|MCG_CAP_COUNT\|MCE_MSR_\|MSR_IA32_MC0_\|SMCA_MSR_\|MCG_EXT_CTL" ~/kbuild/linux/arch/x86 2>/dev/null | sed -n "1,14p"

echo; echo "=== what /sys/firmware/memmap exposes ==="
sudo -n ls /sys/firmware/memmap/ 2>/dev/null | tr "\n" " "; echo
sudo -n sh -c 'ls /sys/firmware/memmap/0/ 2>/dev/null | tr "\n" " "; echo; for i in 0 3 4; do echo "-- entry $i:"; for f in base top type; do printf "%s=%s " "$f" "$(cat /sys/firmware/memmap/$i/$f 2>/dev/null)"; done; echo; done'

echo; echo "=== e820 extended map the kernel kept ==="
sudo -n cat /sys/firmware/firmware/e820 2>/dev/null | sed -n "1,20p"
sudo -n cat /sys/firmware/e820 2>/dev/null | sed -n "1,20p"

echo; echo "=== engine untouched (read-only ps) ==="
ps -o pid,etime= -p 7776,7773,1366 2>/dev/null | tr -s " "
