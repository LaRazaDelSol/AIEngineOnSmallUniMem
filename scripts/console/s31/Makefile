# S31 ps5lowmem: the firmware-reserved low gigabyte as CPU-only swap (/dev/ps5lowswap) + expert tier (/dev/ps5lowmem).
# Build: make   Load: sudo insmod ps5lowmem.ko swap_mib=N   Bench: ./lowmem_bench
obj-m := ps5lowmem.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
all: lowmem_bench
	$(MAKE) -C $(KDIR) M=$(PWD) modules
lowmem_bench: lowmem_bench.cpp
	g++ -O2 -std=c++20 -pthread -o $@ $<
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f lowmem_bench
