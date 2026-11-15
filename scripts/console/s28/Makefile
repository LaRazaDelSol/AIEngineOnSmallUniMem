# S28 read-only kernel probe: name the IP block that refused the GPU's accesses.
#
# Every MSR here is READ via rdmsr_safe() (which fails safely rather than
# faulting). No WRMSR, no MMIO write, no GPU command submission, no change to
# any driver's state. The serving engine (flash_decode PID 7776) is untouched.
#
# Purpose: MCi_IPID's HWID/MCTYPE/instance fields are STATIC hardware ID fields;
# they are not cleared by error handling. So reading them on a healthy boot
# authoritatively names which IP block owns banks 31..46 -- the banks that
# raised "Security Violation" for the reclaimed range.
#
# MSR numbering taken from this kernel tree's own arch/x86/include/asm/msr-index.h:
#   MSR_IA32_MC0_CTL   = 0x400   MSR_IA32_MCx_CTL(x)   = 0x400 + 4x
#   MSR_IA32_MC0_STATUS= 0x401   MSR_IA32_MCx_STATUS(x)= 0x401 + 4x
#   MSR_IA32_MC0_ADDR  = 0x402   MSR_IA32_MCx_ADDR(x)  = 0x402 + 4x
#   MSR_IA32_MC0_MISC  = 0x403   MSR_IA32_MCx_MISC(x)  = 0x403 + 4x

MODULENAME := s28_mce_ro
obj-m := s28_mce_ro.o
KDIR := /lib/modules/7.2.3/build
PWD := $(shell pwd)
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
