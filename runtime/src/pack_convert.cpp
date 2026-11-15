// pack_convert — quantise one HF fused-expert tensor (BF16) into PS5Q pack slots, Q4 and/or Q2 in one pass.
//
//   pack_convert --src SHARD --offset BYTES --kind gate_up|down --layer L
//                [--q4 PACK4] [--q2 PACK2] [--experts 512] [--threads 8] [--steps 12]
//
// HF layout (modeling_qwen4_exp.Qwen4ExpTextExperts):
//   gate_up_proj [E, 2*FFN, HIDDEN]: rows 0..FFN-1 = gate, FFN..2FFN-1 = up   -> slot gate_off / up_off
//   down_proj    [E, HIDDEN, FFN]                                              -> slot down_off
// Pack = (layer * E + expert) * slot_bytes; files are created/extended as needed (sparse until written).
#include "ps5lm/quant.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace qn = ps5lm::q;
static constexpr int H = 2560, F = 640;

static inline float bf16(uint16_t v) {
    uint32_t x = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

template <int B>
static void convert_expert(const uint16_t* src, bool gate_up, uint8_t* slot_buf, int steps, std::vector<float>& row) {
    const qn::ExpertLayout<B> L;
    if (gate_up) {
        const size_t rb = qn::row_bytes<B>(H);
        for (int r = 0; r < 2 * F; ++r) {
            for (int k = 0; k < H; ++k) row[k] = bf16(src[size_t(r) * H + k]);
            uint8_t* dst = slot_buf + (r < F ? L.gate_off + size_t(r) * rb : L.up_off + size_t(r - F) * rb);
            qn::quantize_row_fast<B>(row.data(), H, dst, steps);
        }
    } else {
        const size_t rb = qn::row_bytes<B>(F);
        for (int r = 0; r < H; ++r) {
            for (int k = 0; k < F; ++k) row[k] = bf16(src[size_t(r) * F + k]);
            qn::quantize_row_fast<B>(row.data(), F, slot_buf + L.down_off + size_t(r) * rb, steps);
        }
    }
}

template <int B>
static void write_part(int fd, uint64_t slot_index, const uint8_t* slot_buf, bool gate_up) {
    const qn::ExpertLayout<B> L;
    const uint64_t base = slot_index * L.bytes;
    const size_t off = gate_up ? L.gate_off : L.down_off;
    const size_t len = gate_up ? L.down_off : L.bytes - L.down_off;
    if (::pwrite(fd, slot_buf + off, len, off_t(base + off)) != ssize_t(len)) { std::perror("pwrite pack"); std::exit(1); }
}

template <int B> static void selftest(int steps) {
    std::vector<float> x(H), y(H);
    std::vector<uint8_t> row(qn::row_bytes<B>(H));
    uint64_t s = 12345;
    auto gauss = [&]() {  // Box-Muller from xorshift
        auto u = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (double(s >> 11) + 0.5) / 9007199254740992.0; };
        return float(std::sqrt(-2 * std::log(u())) * std::cos(6.283185307179586 * u()));
    };
    const int R = 400;
    double e_ref = 0, e_fast = 0, n = 0, t_ref = 0, t_fast = 0;
    for (int r = 0; r < R; ++r) {
        for (auto& v : x) v = gauss() * (r % 7 == 0 ? 3.f : 1.f);
        auto t0 = std::chrono::steady_clock::now();
        qn::quantize_row<B>(x.data(), H, row.data());
        auto t1 = std::chrono::steady_clock::now();
        qn::dequantize_row<B>(row.data(), H, y.data());
        for (int k = 0; k < H; ++k) { e_ref += double(y[k] - x[k]) * (y[k] - x[k]); n += double(x[k]) * x[k]; }
        auto t2 = std::chrono::steady_clock::now();
        qn::quantize_row_fast<B>(x.data(), H, row.data(), steps);
        auto t3 = std::chrono::steady_clock::now();
        qn::dequantize_row<B>(row.data(), H, y.data());
        for (int k = 0; k < H; ++k) e_fast += double(y[k] - x[k]) * (y[k] - x[k]);
        t_ref += std::chrono::duration<double>(t1 - t0).count();
        t_fast += std::chrono::duration<double>(t3 - t2).count();
    }
    std::printf("Q%d rel-RMS ref %.4f fast %.4f | speedup %.1fx (%.2f vs %.2f Mweights/s/thread)\n", B, std::sqrt(e_ref / n),
                std::sqrt(e_fast / n), t_ref / t_fast, R * H / t_ref * 1e-6, R * H / t_fast * 1e-6);
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--selftest") {
        const int steps = argc >= 3 ? std::stoi(argv[2]) : 16;
        selftest<4>(steps);
        selftest<2>(steps);
        return 0;
    }
    std::string src, kind, q4, q2;
    uint64_t offset = 0;
    int layer = -1, experts = 512, threads = 8, steps = 12;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--src") src = nx();
        else if (a == "--offset") offset = std::stoull(nx());
        else if (a == "--kind") kind = nx();
        else if (a == "--layer") layer = std::stoi(nx());
        else if (a == "--q4") q4 = nx();
        else if (a == "--q2") q2 = nx();
        else if (a == "--experts") experts = std::stoi(nx());
        else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--steps") steps = std::stoi(nx());
    }
    if (src.empty() || layer < 0 || (kind != "gate_up" && kind != "down") || (q4.empty() && q2.empty())) {
        std::fprintf(stderr, "usage: pack_convert --src F --offset O --kind gate_up|down --layer L [--q4 P] [--q2 P]\n");
        return 2;
    }
    const bool gu = kind == "gate_up";
    const size_t per_expert = size_t(gu ? 2 * F * H : H * F);
    int sfd = ::open(src.c_str(), O_RDONLY);
    if (sfd < 0) { std::perror("open src"); return 1; }
    struct stat st;
    fstat(sfd, &st);
    const size_t need = offset + per_expert * 2 * size_t(experts);
    if (size_t(st.st_size) < need) { std::fprintf(stderr, "src too small: %lld < %zu\n", (long long)st.st_size, need); return 1; }
    void* m = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, sfd, 0);
    if (m == MAP_FAILED) { std::perror("mmap src"); return 1; }
    madvise(m, size_t(st.st_size), MADV_SEQUENTIAL);
    const auto* base = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(m) + offset);
    int fd4 = q4.empty() ? -1 : ::open(q4.c_str(), O_RDWR | O_CREAT, 0644);
    int fd2 = q2.empty() ? -1 : ::open(q2.c_str(), O_RDWR | O_CREAT, 0644);

    auto t0 = std::chrono::steady_clock::now();
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            std::vector<uint8_t> b4(qn::ExpertLayout<4>().bytes), b2(qn::ExpertLayout<2>().bytes);
            std::vector<float> row(H);
            for (int e; (e = next.fetch_add(1)) < experts;) {
                const uint16_t* s = base + size_t(e) * per_expert;
                const uint64_t si = uint64_t(layer) * 512 + uint64_t(e);
                if (fd4 >= 0) { convert_expert<4>(s, gu, b4.data(), steps, row); write_part<4>(fd4, si, b4.data(), gu); }
                if (fd2 >= 0) { convert_expert<2>(s, gu, b2.data(), steps, row); write_part<2>(fd2, si, b2.data(), gu); }
            }
        });
    for (auto& t : pool) t.join();
    if (fd4 >= 0) { fdatasync(fd4); ::close(fd4); }
    if (fd2 >= 0) { fdatasync(fd2); ::close(fd2); }
    munmap(m, size_t(st.st_size));
    ::close(sfd);
    std::printf("layer %d %s: %d experts in %.1f s\n", layer, kind.c_str(), experts,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
