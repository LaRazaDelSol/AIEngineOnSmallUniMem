// platform.hpp — Linux appliance primitives: core pinning, RT scheduling,
// memory locking, HugeTLB (1 GiB -> 2 MiB -> THP fallback) mappings.
//
// Core map (research/design-history/RUNTIME-SPEC.md §1.3, design spec):
//   CPU 0      io_uring SQPOLL kernel thread           (I/O)
//   CPU 1      streamer: request / submit / reap        (I/O)
//   CPU 2      GPU queue submission (per-token graph)   (GPU host)
//   CPU 3      router readback -> slot binding          (GPU host)
//   CPU 4-7    predictor, sampler, tokenizer, prefill    (GPU host)
//   SMT sibs   15 = housekeeping + IRQs; 8-14 offline
#pragma once
#if !defined(__linux__)
#error "platform.hpp is Linux-only"
#endif

#include <fcntl.h>
#include <linux/udmabuf.h>
#include <pthread.h>
#include <sys/ioctl.h>

#include <algorithm>
#include <vector>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

namespace ps5lm::platform {

namespace cores {
inline constexpr int kSqpoll = 0;
inline constexpr int kStreamer = 1;
inline constexpr int kGpuSubmit = 2;
inline constexpr int kRouter = 3;
inline constexpr int kWorkerFirst = 4;
inline constexpr int kWorkerLast = 7;
}  // namespace cores

[[noreturn]] inline void fail(const std::string& what, int err) {
    throw std::runtime_error(what + ": " + std::strerror(err));
}

// Pin the calling thread to one CPU.
inline void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (int r = pthread_setaffinity_np(pthread_self(), sizeof(set), &set); r) fail("pthread_setaffinity_np", r);
}

// SCHED_FIFO on an isolated core. The RT throttle (sched_rt_runtime_us) stays
// on: a runaway FIFO thread then costs 5% instead of a hard lockup.
inline void set_fifo(int prio) {
    sched_param sp{};
    sp.sched_priority = prio;
    if (int r = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp); r) fail("pthread_setschedparam(FIFO)", r);
}

inline void name_thread(const char* name) { pthread_setname_np(pthread_self(), name); }

// Whole-process lock. MCL_ONFAULT: lock pages as they are touched, so huge
// reserve-only GPU-driver mappings are not populated eagerly.
inline void lock_all_memory() {
    rlimit rl{RLIM_INFINITY, RLIM_INFINITY};
    setrlimit(RLIMIT_MEMLOCK, &rl);  // succeeds with CAP_SYS_RESOURCE or LimitMEMLOCK=infinity
    int flags = MCL_CURRENT | MCL_FUTURE;
#ifdef MCL_ONFAULT
    flags |= MCL_ONFAULT;
#endif
    if (mlockall(flags) != 0) fail("mlockall (set LimitMEMLOCK=infinity in the unit)", errno);
}

enum class PageKind : uint8_t { Huge1G, Huge2M, THP };

struct Mapping {
    std::byte* ptr = nullptr;
    size_t bytes = 0;  // mapped length (rounded to the page size)
    PageKind kind = PageKind::THP;
    const char* kind_name() const {
        return kind == PageKind::Huge1G ? "hugetlb-1G" : kind == PageKind::Huge2M ? "hugetlb-2M" : "thp-4K";
    }
};

inline size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Map `bytes` of pinned, pre-faulted memory, preferring the largest page size
// the boot reservation provides. Huge pages are physically contiguous per page
// (GPU fragment PTEs, fewer IOMMU/GART entries, no CPU TLB misses on the slab).
inline Mapping map_pinned(size_t bytes, bool allow_1g = true, bool allow_thp_fallback = true) {
    const int base = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
    if (allow_1g) {
        size_t len = round_up(bytes, size_t(1) << 30);
        void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, base | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
        if (p != MAP_FAILED) return {static_cast<std::byte*>(p), len, PageKind::Huge1G};
    }
    {
        size_t len = round_up(bytes, size_t(2) << 20);
        void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, base | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) return {static_cast<std::byte*>(p), len, PageKind::Huge2M};
    }
    if (!allow_thp_fallback) fail("no hugetlb pages reserved (hugepagesz=1G/2M hugepages=N)", ENOMEM);
    size_t len = round_up(bytes, size_t(2) << 20);
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) fail("mmap(anonymous)", errno);
    madvise(p, len, MADV_HUGEPAGE);
    if (mlock(p, len) != 0) fail("mlock slab", errno);  // faults in + pins
    return {static_cast<std::byte*>(p), len, PageKind::THP};
}

// Anonymous, locked, pre-faulted mapping WITHOUT a hugepage hint. This is the slab type the GPU can
// import: amdgpu userptr (VK_EXT_external_memory_host) is anonymous-only, so hugetlbfs pages are
// rejected (VK_ERROR_INVALID_EXTERNAL_HANDLE, measured S13), and MADV_HUGEPAGE on a 10 GiB mlock
// stalls in direct compaction under memory pressure (measured S13: minutes in D state).
// GPU cost of 4 KiB pages: 411 vs 423 GB/s streaming read (K1b).
inline Mapping map_anon_locked(size_t bytes) {
    size_t len = round_up(bytes, size_t(2) << 20);
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) fail("mmap(anonymous)", errno);
    madvise(p, len, MADV_NOHUGEPAGE);
    if (mlock(p, len) != 0) fail("mlock slab", errno);
    return {static_cast<std::byte*>(p), len, PageKind::THP};
}

// THE expert-slab allocator (S13): a sealed memfd mapped MAP_SHARED (normal shmem pages, so
// O_DIRECT / io_uring READ_FIXED can DMA into them) and exported in <=1 GiB chunks as dma-bufs via
// /dev/udmabuf, which the GPU imports pinned-once (no per-submit userptr validation). The same
// physical pages are seen by the NVMe DMA, the CPU and the GPU: zero copies.
// Needs: CONFIG_UDMABUF, /sys/module/udmabuf/parameters/size_limit_mb >= chunk MiB, root or the
// /dev/udmabuf group (kvm on the console).
struct UdmabufSlab {
    std::byte* ptr = nullptr;
    size_t bytes = 0;
    int memfd = -1;
    std::vector<int> dmabufs;   // one per chunk
    size_t chunk_bytes = 0;
};
inline UdmabufSlab map_udmabuf(size_t bytes, size_t chunk_bytes, bool hugetlb = false, const std::vector<size_t>* chunk_lens = nullptr) {
    UdmabufSlab s;
    const size_t page = hugetlb ? (size_t(2) << 20) : 4096;
    s.bytes = round_up(bytes, page);
    s.chunk_bytes = chunk_bytes;
    unsigned flags = MFD_ALLOW_SEALING | MFD_CLOEXEC;
    if (hugetlb) flags |= MFD_HUGETLB | (21u << 26);  // MFD_HUGE_2MB
    s.memfd = memfd_create("ps5lm-slab", flags);
    if (s.memfd < 0) fail("memfd_create", errno);
    if (ftruncate(s.memfd, off_t(s.bytes)) != 0) fail("ftruncate memfd", errno);
    if (fcntl(s.memfd, F_ADD_SEALS, F_SEAL_SHRINK) != 0) fail("F_ADD_SEALS", errno);
    void* p = mmap(nullptr, s.bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, s.memfd, 0);
    if (p == MAP_FAILED) fail("mmap memfd", errno);
    s.ptr = static_cast<std::byte*>(p);
    if (mlock(p, s.bytes) != 0) fail("mlock memfd slab", errno);
    int dev = ::open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    if (dev < 0) fail("open /dev/udmabuf", errno);
    // S15b: an explicit chunk list (size classes: no slot may straddle a chunk); the last chunk absorbs the rounding
    for (size_t off = 0, ci = 0; off < s.bytes; off += (chunk_lens ? (*chunk_lens)[ci++] : chunk_bytes)) {
        udmabuf_create c{};
        c.memfd = uint32_t(s.memfd);
        c.flags = UDMABUF_FLAGS_CLOEXEC;
        c.offset = off;
        const bool last = chunk_lens && ci + 1 == chunk_lens->size();
        c.size = chunk_lens ? (last ? s.bytes - off : (*chunk_lens)[ci]) : std::min(chunk_bytes, s.bytes - off);
        int fd = ioctl(dev, UDMABUF_CREATE, &c);
        if (fd < 0) { ::close(dev); fail("UDMABUF_CREATE (raise /sys/module/udmabuf/parameters/size_limit_mb)", errno); }
        s.dmabufs.push_back(fd);
        if (last) break;  // the last listed chunk already runs to the (page-rounded) end
    }
    ::close(dev);
    return s;
}
inline void unmap(UdmabufSlab& s) {
    for (int fd : s.dmabufs) ::close(fd);
    if (s.ptr) munmap(s.ptr, s.bytes);
    if (s.memfd >= 0) ::close(s.memfd);
    s = {};
}

inline void unmap(Mapping& m) {
    if (m.ptr) munmap(m.ptr, m.bytes);
    m = {};
}

}  // namespace ps5lm::platform
