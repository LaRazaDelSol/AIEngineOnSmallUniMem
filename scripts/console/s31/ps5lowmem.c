// SPDX-License-Identifier: GPL-2.0
// S31 ps5lowmem: CPU-only use of the firmware-reserved low gigabyte (0x01000000-0x3FFFFFFF, E820 "device reserved").
//
// The GPU cannot touch this range (S25: ~2,600 Coherent-Station "Security Violation" machine checks when pool pages
// landed there; S26-S29 + S31 correction: firmware/PSP-owned fabric policy). The CPU can. So this module NEVER hands
// the range to the page allocator (no memmap=, no hotplug, no pages for TTM/udmabuf/GTT). It maps it write-back
// for the CPU only and exposes two things:
//
//   /dev/ps5lowswap   block device over [base, base + swap_mib): swap / any CPU-side block use (bio memcpy).
//   /dev/ps5lowmem    char device, mmap() of [base + swap_mib, base + size) write-back: the engine's second
//                     expert tier (CPU memcpy only; a tier byte reaches the GPU only after a CPU copy into a pool slot).
//
// Params: base, size (bytes), swap_mib. tier_bytes (read-only) = what /dev/ps5lowmem maps.
#include <linux/module.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/blkdev.h>
#include <linux/bio.h>
#include <linux/highmem.h>

static unsigned long long base = 0x01000000ULL;
static unsigned long long size = 0x3F000000ULL;  // through 0x3FFFFFFF
static unsigned int swap_mib;                     // 0 = no block device
static unsigned long long tier_bytes;
module_param(base, ullong, 0444);
module_param(size, ullong, 0444);
module_param(swap_mib, uint, 0444);
module_param(tier_bytes, ullong, 0444);
MODULE_PARM_DESC(base, "physical base (must be inside the firmware-reserved range)");
MODULE_PARM_DESC(size, "bytes");
MODULE_PARM_DESC(swap_mib, "MiB at the start of the range for /dev/ps5lowswap (0 = none)");
MODULE_PARM_DESC(tier_bytes, "read-only: bytes /dev/ps5lowmem maps");

static void *va;               // write-back kernel mapping of the whole range
static u64 swap_bytes, tier_off;
static int blk_major;
static struct gendisk *disk;

// ---- block device (swap) ----
static void lm_submit_bio(struct bio *bio)
{
	struct bio_vec bv;
	struct bvec_iter it;
	u64 off = (u64)bio->bi_iter.bi_sector << SECTOR_SHIFT;

	switch (bio_op(bio)) {
	case REQ_OP_READ:
	case REQ_OP_WRITE:
		break;
	case REQ_OP_FLUSH:
	case REQ_OP_DISCARD:
	case REQ_OP_WRITE_ZEROES:
		bio_endio(bio);
		return;
	default:
		bio_io_error(bio);
		return;
	}
	if (off + bio->bi_iter.bi_size > swap_bytes) {
		bio_io_error(bio);
		return;
	}
	bio_for_each_segment(bv, bio, it) {
		if (bio_op(bio) == REQ_OP_READ)
			memcpy_to_page(bv.bv_page, bv.bv_offset, (char *)va + off, bv.bv_len);
		else
			memcpy_from_page((char *)va + off, bv.bv_page, bv.bv_offset, bv.bv_len);
		off += bv.bv_len;
	}
	bio_endio(bio);
}

static const struct block_device_operations lm_bops = {
	.owner = THIS_MODULE,
	.submit_bio = lm_submit_bio,
};

// ---- char device (expert tier) ----
static int lm_mmap(struct file *f, struct vm_area_struct *vma)
{
	const u64 off = (u64)vma->vm_pgoff << PAGE_SHIFT, len = vma->vm_end - vma->vm_start;

	if (off >= tier_bytes || len > tier_bytes - off)
		return -EINVAL;
	vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);
	// vm_page_prot is the default write-back protection; the memtype was reserved WB by memremap() below
	return remap_pfn_range(vma, vma->vm_start, (base + tier_off + off) >> PAGE_SHIFT, len, vma->vm_page_prot);
}

static const struct file_operations lm_fops = {
	.owner = THIS_MODULE,
	.mmap = lm_mmap,
};

static struct miscdevice lm_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "ps5lowmem",
	.fops = &lm_fops,
	.mode = 0600,
};

static int __init lm_init(void)
{
	int r;

	if (base & ~PAGE_MASK || size & ~PAGE_MASK || !size)
		return -EINVAL;
	// refuse anything the kernel treats as System RAM: this module is only for the reserved carve-out
	if (region_intersects(base, size, IORESOURCE_SYSTEM_RAM, IORES_DESC_NONE) != REGION_DISJOINT) {
		pr_err("ps5lowmem: [%llx, %llx) overlaps System RAM, refusing\n", base, base + size);
		return -EINVAL;
	}
	swap_bytes = (u64)swap_mib << 20;
	if (swap_bytes > size)
		return -EINVAL;
	tier_off = swap_bytes;
	tier_bytes = size - swap_bytes;

	va = memremap(base, size, MEMREMAP_WB);
	if (!va) {
		pr_err("ps5lowmem: memremap WB failed\n");
		return -ENOMEM;
	}
	if (tier_bytes) {
		r = misc_register(&lm_misc);
		if (r)
			goto err_unmap;
	}
	if (swap_bytes) {
		struct queue_limits lim = {
			.logical_block_size = PAGE_SIZE,
			.physical_block_size = PAGE_SIZE,
			.io_min = PAGE_SIZE,
			.max_hw_sectors = 1024,  // 512 KiB per bio
			.features = BLK_FEAT_SYNCHRONOUS,
		};

		blk_major = register_blkdev(0, "ps5lowswap");
		if (blk_major < 0) {
			r = blk_major;
			goto err_misc;
		}
		disk = blk_alloc_disk(&lim, NUMA_NO_NODE);
		if (IS_ERR(disk)) {
			r = PTR_ERR(disk);
			disk = NULL;
			goto err_blk;
		}
		disk->major = blk_major;
		disk->first_minor = 0;
		disk->minors = 1;
		disk->fops = &lm_bops;
		strscpy(disk->disk_name, "ps5lowswap", DISK_NAME_LEN);
		set_capacity(disk, swap_bytes >> SECTOR_SHIFT);
		r = add_disk(disk);
		if (r)
			goto err_disk;
	}
	pr_info("ps5lowmem: [%llx, %llx) mapped WB, swap %llu MiB, tier %llu MiB (CPU only; never GPU-visible)\n",
		base, base + size, swap_bytes >> 20, tier_bytes >> 20);
	return 0;

err_disk:
	put_disk(disk);
	disk = NULL;
err_blk:
	unregister_blkdev(blk_major, "ps5lowswap");
err_misc:
	if (tier_bytes)
		misc_deregister(&lm_misc);
err_unmap:
	memunmap(va);
	return r;
}

static void __exit lm_exit(void)
{
	if (disk) {
		del_gendisk(disk);
		put_disk(disk);
		unregister_blkdev(blk_major, "ps5lowswap");
	}
	if (tier_bytes)
		misc_deregister(&lm_misc);
	memunmap(va);
}

module_init(lm_init);
module_exit(lm_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("PS5 reserved low gigabyte as CPU-only swap + expert tier");
