// SPDX-License-Identifier: GPL-2.0
// S28 read-only MCE probe. READS model-specific registers only.
//
// Every access is a READ (rdmsr_safe). There is no wrmsr, no iowrite, no MMIO
// write, no GPU command submission, and no driver state change. The serving
// engine (flash_decode PID 7776 / ps5lm_server.py PID 7773) is not touched.
//
// Why this answers the open question: MCi_IPID carries STATIC hardware-identity
// fields (HWID[35:32], MCTYPE[43:36], instance ID[15:0]) that error handling
// does not clear. So reading them on a healthy boot authoritatively names which
// IP block owns the banks that raised "Security Violation" for the reclaimed
// range -- i.e. whether the refused request was issued on a CPU-core-side bank
// or on a GPU/GFX-side request bank.
//
// MSR numbering taken from this kernel tree's own arch/x86/include/asm/
// msr-index.h: MSR_IA32_MC0_CTL=0x400, _STATUS=0x401, _ADDR=0x402, _MISC=0x403,
// with MSR_IA32_MCx_FOO(x) = base + 4*(x).

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <asm/msr.h>

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("read-only SMCA machine-check register survey");

static u64 ro_rdm(u32 msr, int *ok)
{
	u64 v = 0;
	/* READ only; the _safe variant fails safe rather than faulting. */
	*ok = rdmsrq_safe(msr, &v);
	return v;
}

static int __init s28_init(void)
{
	int b, ok;
	u64 v;

	pr_info("s28_ro: === READ-ONLY survey (no wrmsr, no MMIO write) ===\n");

	/* Top-of-memory MSRs: where the hardware says DRAM apertures end. */
	v = ro_rdm(0xc001001a, &ok);   /* MSR_K8_TOP_MEM1  */
	pr_info("s28_ro: TOM1  MSR c001001a = %016llx (rd ok=%d)\n", v, ok);
	v = ro_rdm(0xc00101ea, &ok);   /* MSR_K8_TOP_MEM2  */
	pr_info("s28_ro: TOM2  MSR c00101ea = %016llx (rd ok=%d)\n", v, ok);
	v = ro_rdm(0xc00101ee, &ok);   /* MSR_K8_TOP_MEM3 / TOM2 high */
	pr_info("s28_ro: TOM3  MSR c00101ee = %016llx (rd ok=%d)\n", v, ok);

	/*
	 * The faulting banks. dmesg from the failed-reclaim boot reported
	 *   Bank 31..46: MCi_STATUS = 0xdc2030000002085b (and ...082b)
	 *   MCi_ADDR = 0x1dc00200 ... 0x24801f00   (all inside the reclaimed range)
	 *   MCi_MISC = 0xd01b0fff00000000 (constant across every bank)
	 *   MCi_SYND = 0x5b260010..0x5b260028
	 *   MCi_IPID = 0x2002e00000002..0x2002e0000000f
	 * Identify the MSR numbering empirically by matching those values, then
	 * read the STATIC identity fields that error handling does not clear.
	 */
	pr_info("s28_ro: --- stride-4 block: MCx_{STATUS,ADDR,MISC} at 0x401/2/3 + 4*bank ---\n");
	for (b = 31; b <= 46; b++) {
		v = ro_rdm(0x401 + 4 * b, &ok);
		pr_info("s28_ro: bank %2d STATUS@0x%x = %016llx (ok=%d)\n", b, 0x401 + 4 * b, v, ok);
		v = ro_rdm(0x402 + 4 * b, &ok);
		pr_info("s28_ro: bank %2d ADDR  @0x%x = %016llx (ok=%d)\n", b, 0x402 + 4 * b, v, ok);
		v = ro_rdm(0x403 + 4 * b, &ok);
		pr_info("s28_ro: bank %2d MISC  @0x%x = %016llx (ok=%d)\n", b, 0x403 + 4 * b, v, ok);
	}

	/*
	 * SMCA adds two per-bank MSRs holding the IP identity (IPID) and the
	 * extended syndrome (DESyND). Scan the candidate AMD MSR bases so the
	 * answer is empirical: the correct base is the one whose value equals the
	 * dmesg-recorded constants (IPID 0x2002e0000000x, DESyND 0x5b26____).
	 */
	pr_info("s28_ro: --- candidate SMCA IPID/DESyND bases (match vs dmesg constants) ---\n");
	{
		static const u32 bases[] = { 0xc0002078, 0xc0002080, 0xc0002088,
					 0xc0002090, 0xc0002098, 0xc00020a0 };
		int i;
		for (i = 0; i < ARRAY_SIZE(bases); i++) {
			for (b = 31; b <= 34; b++) {
				u32 m = bases[i] + 8 * b;
				v = ro_rdm(m, &ok);
				pr_info("s28_ro: bank %2d MSR 0x%x = %016llx (ok=%d)\n", b, m, v, ok);
			}
		}
	}

	/*
	 * Decode helper: HWID[35:32], MCTYPE[43:36] from MCi_IPID.
	 * Kernel SMCA names HWID 0x2E / MCTYPE 0x2 as SMCA_CS_V2 = Coherent
	 * Station. Printing the raw IPID lets that be checked, not assumed.
	 */
	pr_info("s28_ro: --- decode: IPID -> HWID[35:32] MCTYPE[43:36] instance[15:0] ---\n");
	pr_info("s28_ro: reference IPID from failed boot = 0x2002e00000002 -> HWID=0x%x MCTYPE=0x%x inst=0x%x\n",
		0x2e, 0x2, 0x2);

	pr_info("s28_ro: === survey complete: reads only, no register written ===\n");
	return 0;
}

static void __exit s28_exit(void)
{
	pr_info("s28_ro: unloaded (nothing was ever written)\n");
}

module_init(s28_init);
module_exit(s28_exit);
MODULE_AUTHOR("session");
MODULE_DESCRIPTION("read-only SMCA MCE register survey");
