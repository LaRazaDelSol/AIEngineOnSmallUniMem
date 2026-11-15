// S31 gate 1: CPU copy rate between the reserved low gigabyte (/dev/ps5lowmem, write-back) and normal RAM.
// ./lowmem_bench [MiB=512] [chunk_kib=2048]
// Prints, per thread count: tier->RAM and RAM->tier GB/s, and the latency of one chunk copy (the size of an expert).
// Baseline: RAM->RAM. Verifies every byte it copied (a pattern written through the mapping and read back).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

static double run(uint8_t* dst, const uint8_t* src, size_t bytes, size_t chunk, int nt, int reps) {
    double best = 1e9;
    for (int r = 0; r < reps; ++r) {
        std::atomic<size_t> next{0};
        auto t = Clock::now();
        std::vector<std::thread> th;
        for (int i = 0; i < nt; ++i) th.emplace_back([&] {
            for (size_t o; (o = next.fetch_add(chunk)) < bytes;) std::memcpy(dst + o, src + o, std::min(chunk, bytes - o));
        });
        for (auto& x : th) x.join();
        best = std::min(best, secs(t));
    }
    return double(bytes) / best / 1e9;
}
static double lat_us(uint8_t* dst, const uint8_t* src, size_t bytes, size_t chunk, int n) {  // one chunk, one thread, median-ish (min of n)
    double best = 1e9;
    for (int i = 0; i < n; ++i) {
        const size_t o = (size_t(i) * 7919 * chunk) % (bytes - chunk + 1) & ~size_t(4095);
        auto t = Clock::now();
        std::memcpy(dst + o, src + o, chunk);
        best = std::min(best, secs(t));
    }
    return best * 1e6;
}

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 512, chunk = (argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 2048) << 10;
    const size_t bytes = mib << 20;
    int fd = open("/dev/ps5lowmem", O_RDWR);
    if (fd < 0) { perror("/dev/ps5lowmem"); return 1; }
    auto* tier = static_cast<uint8_t*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (tier == MAP_FAILED) { perror("mmap"); return 1; }
    auto* ram = static_cast<uint8_t*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0));
    auto* ram2 = static_cast<uint8_t*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0));
    if (ram == MAP_FAILED || ram2 == MAP_FAILED) { perror("mmap ram"); return 1; }
    for (size_t i = 0; i < bytes / 8; ++i) reinterpret_cast<uint64_t*>(ram)[i] = i * 0x9E3779B97F4A7C15ull;
    // correctness: RAM -> tier -> RAM2 must round-trip
    std::memcpy(tier, ram, bytes);
    std::memcpy(ram2, tier, bytes);
    if (std::memcmp(ram, ram2, bytes)) { std::puts("VERIFY FAILED: tier round-trip differs"); return 2; }
    std::printf("verify: %zu MiB round-trip RAM -> tier -> RAM identical\n", mib);
    std::printf("chunk %zu KiB   %8s %8s %8s   (GB/s, best of 3)\n", chunk >> 10, "tier>RAM", "RAM>tier", "RAM>RAM");
    for (int nt : {1, 2, 4, 8}) {
        const double a = run(ram2, tier, bytes, chunk, nt, 3), b = run(tier, ram, bytes, chunk, nt, 3), c = run(ram2, ram, bytes, chunk, nt, 3);
        std::printf("threads %d       %8.2f %8.2f %8.2f\n", nt, a, b, c);
    }
    std::printf("one chunk, one thread, min of 64 (us): tier>RAM %.1f  RAM>tier %.1f  RAM>RAM %.1f\n",
                lat_us(ram2, tier, bytes, chunk, 64), lat_us(tier, ram, bytes, chunk, 64), lat_us(ram2, ram, bytes, chunk, 64));
    return 0;
}
