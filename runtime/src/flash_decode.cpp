// flash_decode — Qwen3.8-Flash-Next on the PS5 GPU (Vulkan, GFX1013), experts streamed from NVMe.
//
// CLI (one prompt, greedy; S13-compatible output: comma-separated ids on stdout, stats on stderr):
//   flash_decode --model DIR --shaders DIR --ids 760,6511,... [--gen N] [--pool-gib G] [--bits 2|4]
//                [--predict D] [--dump FILE] [--trace-out FILE] [--profile] [--layers N] [--max-ctx C]
//
// Server (S14): --serve keeps weights, pipelines and the warm expert pool resident and reads requests on
// stdin (runtime/tools/ps5lm_server.py is the HTTP front end: OpenAI + Anthropic APIs, tokenizer, SSE).
//   G n=<max_new> t=<temp> p=<top_p> k=<top_k> m=<min_p> r=<rep_pen> pp=<presence> fp=<frequency>
//     s=<seed> snap=<pos,pos,...> stop=<id,id,...> ids=<id,id,...>
//       -> "T <id>" per generated token, then "D reason=<stop|length|cancel|ctx> key=value ..."
//   X                 cancel the running generation (checked between tokens)
//   S <path>          save the pool residency (most recently used first)    -> "OK <n>"
//   W <path>          preload the pool from a residency/profile file          -> "OK <n>"
//   I                 info                                                    -> "I key=value ..."
//   Q                 quit (saves residency to --residency if given)
// Prefix reuse: the engine keeps its live state (tokens fed so far) plus --snapshots slots. A request
// continues from the live state if its ids extend it, else restores the longest valid snapshot that is a
// prefix of the ids, else resets. `snap=` asks for snapshots at those prompt positions (the front end sends
// the end of the system/tools block and the end of the prompt).
#include "ps5lm/flash_engine.hpp"
#include "ps5lm/session_store.hpp"

#include <poll.h>
#include <sys/mman.h>

#include <cerrno>
#include <csignal>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ps5lm;
using namespace ps5lm::flash;

static std::vector<int64_t> parse_ids(const std::string& s) {
    std::vector<int64_t> v;
    for (size_t p = 0; p < s.size();) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) v.push_back(std::stoll(s.substr(p, q - p)));
        p = q + 1;
    }
    return v;
}

// ---------------------------------------------------------------- sampling (host, from HOST_CACHED logits)
struct SampleParams {
    float temperature = 0, top_p = 1, min_p = 0, rep_pen = 1, presence = 0, frequency = 0;
    uint32_t top_k = 0;
    uint64_t seed = 0;
    bool greedy_gpu() const { return temperature <= 0 && rep_pen == 1 && presence == 0 && frequency == 0; }
};

struct Sampler {
    std::mt19937_64 rng{0x5eed};
    std::vector<std::pair<float, uint32_t>> cand;
    // counts: GENERATED tokens of this request (presence/frequency, OpenAI/vLLM semantics, which Qwen's presence_penalty
    // 1.5 assumes). prompt: tokens of the prompt, seen only by the repetition penalty (HF/vLLM: prompt + output).
    // S34: both penalties used to count the whole prompt, so presence 1.5 hit every token of the conversation
    // (</think>, <|im_end|>, common words), not just what this answer repeated.
    uint32_t sample(const float* lg, uint32_t n, const SampleParams& sp, const std::unordered_map<int64_t, uint32_t>& counts,
                    const std::unordered_set<int64_t>& prompt) {
        // Penalised value of token i (HF semantics: repetition penalty divides positive logits, multiplies negative).
        auto val = [&](uint32_t i) {
            float v = lg[i];
            auto it = counts.find(i);
            const bool out = it != counts.end();
            if (sp.rep_pen != 1 && (out || prompt.count(i))) v = v > 0 ? v / sp.rep_pen : v * sp.rep_pen;
            if (out) v -= sp.presence + sp.frequency * float(it->second);
            return v;
        };
        if (sp.temperature <= 0) {
            uint32_t best = 0;
            float bv = -INFINITY;
            for (uint32_t i = 0; i < n; ++i) { const float v = val(i); if (v > bv) { bv = v; best = i; } }
            return best;
        }
        // Candidates: top max(top_k, 1024) by value (the nucleus at top_p <= 0.99 is far smaller in practice).
        const uint32_t keep = std::min<uint32_t>(n, sp.top_k ? sp.top_k : 1024);
        cand.resize(n);
        for (uint32_t i = 0; i < n; ++i) cand[i] = {val(i), i};
        std::nth_element(cand.begin(), cand.begin() + keep, cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        cand.resize(keep);
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const float mx = cand[0].first;
        double z = 0;
        std::vector<double> p(keep);
        for (uint32_t i = 0; i < keep; ++i) { p[i] = std::exp(double(cand[i].first - mx) / sp.temperature); z += p[i]; }
        uint32_t cut = keep;
        double acc = 0;
        for (uint32_t i = 0; i < keep; ++i) {
            p[i] /= z;
            if (sp.min_p > 0 && p[i] < sp.min_p * p[0]) { cut = std::max<uint32_t>(i, 1); break; }
            acc += p[i];
            if (acc >= sp.top_p) { cut = i + 1; break; }
        }
        double tot = 0;
        for (uint32_t i = 0; i < cut; ++i) tot += p[i];
        double u = std::uniform_real_distribution<double>(0, tot)(rng);
        for (uint32_t i = 0; i < cut; ++i) { u -= p[i]; if (u <= 0) return cand[i].second; }
        return cand[cut - 1].second;
    }
};

// ---------------------------------------------------------------- stdin line reader with non-blocking peek
struct LineIn {
    std::string buf;
    bool eof = false;
    bool fill(int timeout_ms) {
        pollfd pf{0, POLLIN, 0};
        if (::poll(&pf, 1, timeout_ms) <= 0) return false;
        char tmp[65536];
        ssize_t r = ::read(0, tmp, sizeof tmp);
        if (r <= 0) { eof = true; return false; }
        buf.append(tmp, size_t(r));
        return true;
    }
    bool next(std::string& line, bool block) {
        for (;;) {
            size_t nl = buf.find('\n');
            if (nl != std::string::npos) { line = buf.substr(0, nl); buf.erase(0, nl + 1); return true; }
            if (eof) return false;
            if (!fill(block ? -1 : 0) && !block) return false;
        }
    }
};

// S16: NVMe busy time over the generation phase (/sys/block/nvme0n1/stat: field 3 sectors read, field 10 io_ticks ms)
struct DiskStat {
    unsigned long long sectors = 0, ticks = 0;
    static DiskStat now() {
        DiskStat d;
        if (FILE* f = std::fopen("/sys/block/nvme0n1/stat", "r")) {
            unsigned long long v[11] = {};
            for (int i = 0; i < 11; ++i) if (std::fscanf(f, "%llu", &v[i]) != 1) break;
            d.sectors = v[2];
            d.ticks = v[9];
            std::fclose(f);
        }
        return d;
    }
    static void report(const DiskStat& a, const DiskStat& b, double wall_s) {
        const double busy = double(b.ticks - a.ticks) * 1e-3, mb = double(b.sectors - a.sectors) * 512.0 / 1e6;
        std::fprintf(stderr, "SSD (gen phase): busy %.1f%% of %.2f s wall, %.0f MB read = %.2f GB/s average, %.2f GB/s while busy\n",
                     wall_s > 0 ? 100.0 * busy / wall_s : 0.0, wall_s, mb, wall_s > 0 ? mb / 1e3 / wall_s : 0.0, busy > 0 ? mb / 1e3 / busy : 0.0);
    }
};

// S24 cache-aware routing: how often the biased selection substituted a candidate and how much true top-K weight it displaced
static void route_line(FlashEngine& e) {
    auto& s = e.stats();
    if (!s.rr_sel) return;
    std::fprintf(stderr, "cache-aware routing: %llu selections, %.2f substituted / selection, %.2f non-resident kept / selection, displaced weight share %.4f\n",
                 (unsigned long long)s.rr_sel, double(s.rr_swaps) / double(s.rr_sel), double(s.rr_miss_kept) / double(s.rr_sel), s.rr_mass / double(s.rr_sel));
    if (s.rr_shadow)
        std::fprintf(stderr, "cache-aware routing: %.2f shadow uses / selection\n", double(s.rr_shadow) / double(s.rr_sel));
    if (s.rr_bg || s.rr_bg_dropped)
        std::fprintf(stderr, "cache-aware routing: %.2f background reads / token (%llu dropped)\n", double(s.rr_bg) / double(std::max<uint64_t>(s.tokens, 1)),
                     (unsigned long long)s.rr_bg_dropped);
}
// S31 lowmem tier: misses served by a CPU copy from the reserved gigabyte instead of an SSD read
static void tier_line(FlashEngine& e) {
    const LowTier* t = e.streamer().tier();
    if (!t) return;
    const auto& s = t->stats();
    const uint64_t tok = std::max<uint64_t>(e.stats().tokens, 1);
    std::fprintf(stderr, "lowmem tier: %.2f fills/tok (%.1f%% of pool misses, %.0f us avg), %.2f admits/tok, %llu re-admits, %llu skips, "
                         "%.1f MB/tok copied, %.1f%% of evictions kept\n",
                 double(s.hits) / double(tok), s.lookups ? 100.0 * double(s.hits) / double(s.lookups) : 0.0, s.hits ? s.fill_s / double(s.hits) * 1e6 : 0.0,
                 double(s.admits) / double(tok), (unsigned long long)s.readmits, (unsigned long long)s.admit_skips,
                 double(s.fill_bytes + s.admit_bytes) / 1e6 / double(tok), s.evicted ? 100.0 * double(s.evicted_kept) / double(s.evicted) : 0.0);
}
static void stats_line(FlashEngine& e, uint32_t prompt, uint32_t gen, double tgen) {
    auto& s = e.stats();
    if (e.streamer().n_devices() > 1) {  // S23 mirrored packs: where the expert reads went
        std::fprintf(stderr, "\npack mirror reads:");
        for (uint32_t d = 0; d < e.streamer().n_devices(); ++d) std::fprintf(stderr, " dev%u %llu", d, (unsigned long long)e.streamer().device_reads(d));
    }
    const uint64_t tot = std::max<uint64_t>(s.tokens, 1);
    std::fprintf(stderr, "\nprompt %u tok, generated %u tok: %.2f tok/s (gen phase), GPU+sync %.1f ms/tok, I/O wait %.1f ms/tok, "
                         "hit %.3f, GPU-pure %.1f ms/tok, host %.2f ms/tok, n-gram wait %.3f ms/tok, predict precision %.3f, prefetch %.1f/tok\n",
                 prompt, gen, gen > 1 ? double(gen - 1) / tgen : 0.0, s.t_gpu / tot * 1e3, s.t_io / tot * 1e3, e.streamer().cache().stats().hit_rate(),
                 s.t_gpu_pure / tot * 1e3, s.t_host / tot * 1e3, s.t_ngram / tot * 1e3, s.pred_total ? double(s.pred_correct) / double(s.pred_total) : 0.0,
                 double(s.pref_issued) / tot);
    e.selstat_report();  // S32 tiering probe (PS5LM_SELSTAT=1)
    if (s.trt_total || s.trt_issued)
        std::fprintf(stderr, "token-route table: %.2f reads/tok issued, guess precision %.3f (%llu scored)\n", double(s.trt_issued) / tot,
                     s.trt_total ? double(s.trt_ok) / double(s.trt_total) : 0.0, (unsigned long long)s.trt_total);
    route_line(e);
    tier_line(e);
    if (e.streamer().cache().stats().saturated)
        std::fprintf(stderr, "slot pool: %llu lookups refused (every slot of the class pinned or in flight)\n",
                     (unsigned long long)e.streamer().cache().stats().saturated);
}

// S16 --mtp-measure D: sequential decoding (output identical to plain decoding) + after every trunk token a chain of D
// MTP drafts; prefill fills the MTP KV. Reports the greedy acceptance a speculative decoder would get.
static void mtp_report(const EngineStats& s, const std::vector<int64_t>& toks, const std::vector<std::vector<uint32_t>>& drafts, uint32_t depth) {
    std::vector<uint64_t> hist(depth + 1, 0), marg(depth, 0), cond_n(depth, 0), cond_ok(depth, 0);
    uint64_t n = 0;
    for (size_t t = 0; t < drafts.size(); ++t) {
        if (drafts[t].empty() || t + 1 + depth >= toks.size()) continue;
        ++n;
        uint32_t a = 0;
        while (a < depth && int64_t(drafts[t][a]) == toks[t + 2 + a]) ++a;
        ++hist[a];
        for (uint32_t j = 0; j < depth; ++j) {
            marg[j] += int64_t(drafts[t][j]) == toks[t + 2 + j];
            if (j <= a) { ++cond_n[j]; cond_ok[j] += j < a; }
        }
    }
    if (!n) return;
    double mean = 0;
    for (uint32_t a = 0; a <= depth; ++a) mean += double(a) * double(hist[a]);
    mean /= double(n);
    std::fprintf(stderr, "MTP depth %u over %llu positions: accepted-length histogram", depth, (unsigned long long)n);
    for (uint32_t a = 0; a <= depth; ++a) std::fprintf(stderr, " %u:%llu", a, (unsigned long long)hist[a]);
    std::fprintf(stderr, "\n  mean accepted drafts %.3f -> %.3f tokens per verify; per depth (chain-conditional / marginal):", mean, 1.0 + mean);
    for (uint32_t j = 0; j < depth; ++j)
        std::fprintf(stderr, " d%u %.3f/%.3f", j + 1, cond_n[j] ? double(cond_ok[j]) / double(cond_n[j]) : 0.0, double(marg[j]) / double(n));
    std::fprintf(stderr, "\n  MTP cost: %llu drafts, %.2f ms/draft wall (of which expert I/O wait %.2f), %.2f expert lookups/draft; KV fill %.2f ms total\n",
                 (unsigned long long)s.mtp_drafts, s.mtp_drafts ? s.t_mtp / double(s.mtp_drafts) * 1e3 : 0.0,
                 s.mtp_drafts ? s.t_mtp_io / double(s.mtp_drafts) * 1e3 : 0.0, s.mtp_drafts ? double(s.mtp_demand) / double(s.mtp_drafts) : 0.0,
                 s.t_mtp_fill * 1e3);
}

// S20 batched prefill of ids[start, end): chunks of up to prefill_max() tokens through the one-cb verify path (no
// checkpoints, no lm_head), cut at the `cuts` positions (snapshots are taken there by after()). With mtp, each position's
// MTP KV entry is written right after its chunk (it needs the next token, so position end-1 only if end < ids.size()).
// after(pos) runs after every chunk with pos = tokens fed so far; returning false stops the prefill (cancel).
static uint32_t g_prefill_tb = 4;
static std::function<void(size_t)> g_pf_progress;  // S20: progress inside a layer-major chunk (server P lines)
static size_t g_lm_min = 64;  // S20: layer-major prefill for runs of >= this many tokens (--prefill-lm-min; huge = off)
static double g_prefill_s = 0;  // CLI report
template <class After>
static bool prefill_range(FlashEngine& e, const std::vector<int64_t>& ids, size_t start, size_t end, bool mtp,
                          const std::vector<int64_t>& cuts, After after) {
    const uint32_t P = std::max<uint32_t>(1, std::min<uint32_t>(e.prefill_max(), g_prefill_tb));
    size_t i = start;
    while (i < end) {
        size_t stop = end;  // the next cut
        for (int64_t c : cuts) if (c > int64_t(i) && size_t(c) < stop) stop = size_t(c);
        if (e.lm_max() && g_prefill_tb > 1 && stop - i >= g_lm_min) {  // S20 layer-major chunk
            const size_t N = std::min<size_t>(stop - i, e.lm_max()) & ~size_t(3);
            e.prefill_lm(ids.data() + i, uint32_t(N), mtp, i + N < ids.size() ? ids[i + N] : -1, [&](uint32_t s) {
                if (g_pf_progress) g_pf_progress(i + N * (s + 1) / (L + 1));
            });
            i += N;
            if (!after(i)) return false;
            continue;
        }
        size_t j = std::min<size_t>(stop, i + P);
        const uint32_t T = uint32_t(j - i);
        e.prefill_chunk(ids.data() + i, T);
        if (mtp)
            for (uint32_t t = 0; t < T; ++t)
                if (i + t + 1 < ids.size()) e.mtp_fill(ids[i + t + 1], uint32_t(i + t), T == 1 ? 0 : t);
        i = j;
        if (!after(i)) return false;
    }
    return true;
}

// S16 speculative greedy decoding: MTP drafts k tokens, verify() checks them in one pass over the layers. Prints the
// same token stream as run_cli (greedy decoding is exact: every emitted token is the trunk's argmax).
static float g_spec_gate = 0.5f;  // S18: verify a draft only if its MTP top-1 probability >= this (0 = always); break-even ~0.51
static float g_spec_deeper = 0.9f;  // S18 variable depth: draft token j+1 only if draft j's confidence >= this
static float g_spec_keep = 0.6f;    // ... and keep a deeper draft (j >= 2) in the verify batch only if its confidence >= this
// S23 prompt-lookup drafts (PLD, --pld N): when the last N tokens (ending with the pending one) already occurred in this
// context, the tokens that followed them are the draft. No GPU work, no MTP expert reads; the verify pass is the same one
// MTP drafts go through, so the output is exact for greedy and for sampling (a deterministic draft, see serve()).
// Copy-heavy output (code edits, tool-call JSON, quoted context) is where this pays: the drafts cost nothing and may be
// deeper than MTP's. Prose rarely matches N tokens, and when matches stop paying the acceptance EMA switches PLD off
// (re-probed every 16 opportunities), so the MTP path is what runs there.
struct Pld {
    uint32_t min_match = 0;                         // 0 = off
    std::vector<int64_t> s;                         // mirror of the engine's token history
    std::unordered_map<uint64_t, uint32_t> last;    // hash of the N-gram ending at i -> i (latest; only i with a successor)
    double ema = 0.75;                              // accepted share of the drafted tokens
    uint32_t skipped = 0, cur_n = 0;                // opportunities passed while off; drafts proposed for the current cycle
    uint64_t cycles = 0, drafted = 0, accepted = 0, matches = 0;
    static uint64_t hash(const int64_t* g, uint32_t n) {
        uint64_t h = 0x243F6A8885A308D3ull;
        for (uint32_t i = 0; i < n; ++i) { h ^= uint64_t(g[i]); h *= 0x100000001B3ull; h ^= h >> 29; }
        return h;
    }
    void clear() { s.clear(); last.clear(); }
    void sync(const FlashEngine& e) {
        const uint32_t n = e.pos();
        if (n < s.size() || (!s.empty() && e.token_at(uint32_t(s.size() - 1)) != s.back())) clear();  // rewound / replaced
        s.reserve(n + 64);
        for (uint32_t i = uint32_t(s.size()); i < n; ++i) {
            s.push_back(e.token_at(i));
            if (s.size() > min_match) last[hash(&s[s.size() - 1 - min_match], min_match)] = uint32_t(s.size() - 2);
        }
    }
    // Drafts for the tokens after `cur` (pending, not fed yet). Returns how many (0 = no match, or PLD is resting).
    uint32_t propose(int64_t cur, uint32_t maxn, int64_t* out) {
        cur_n = 0;
        if (!min_match || min_match > 32 || s.size() + 1 < size_t(min_match) + 1 || !maxn) return 0;
        int64_t g[32];
        for (uint32_t i = 0; i + 1 < min_match; ++i) g[i] = s[s.size() - (min_match - 1) + i];
        g[min_match - 1] = cur;
        const auto it = last.find(hash(g, min_match));
        if (it == last.end()) return 0;
        const uint32_t j = it->second;
        for (uint32_t i = 0; i < min_match; ++i) if (s[j + 1 - min_match + i] != g[i]) return 0;  // hash collision
        ++matches;
        if (ema < 0.5) {  // not paying on this text (break-even of a T=2 verify is ~0.51, R68)
            if (++skipped < 16) return 0;
            maxn = 1;     // probe again with one draft
        } else if (ema < 0.75) maxn = 1;
        skipped = 0;
        uint32_t n = 0;
        for (; n < maxn; ++n) {
            const size_t q = size_t(j) + 1 + n;
            if (q < s.size()) out[n] = s[q];
            else if (q == s.size()) out[n] = cur;
            else break;
        }
        return cur_n = n;
    }
    void report(uint32_t m) {  // after the verify of a cycle: m drafts accepted
        if (!cur_n) return;
        ++cycles; drafted += cur_n; accepted += std::min(m, cur_n);
        ema = 0.8 * ema + 0.2 * double(std::min(m, cur_n)) / double(cur_n);
        cur_n = 0;
    }
};
static Pld g_pld;
// Drafts v[1..] after v[0] (chain from the residual of position idx in block blk). Returns the verify batch size T (1 = the
// first draft was not confident enough: feed v[0] plainly). conf[j] = draft j's MTP top-1 probability (1 for PLD drafts).
static uint32_t draft_chain(FlashEngine& e, std::vector<int64_t>& v, uint32_t Tmax, uint32_t idx, uint32_t blk, float* conf) {
    if (g_pld.min_match) {
        g_pld.sync(e);
        const uint32_t n = g_pld.propose(v[0], Tmax - 1, &v[1]);
        if (n) {
            e.mtp_fill(v[0], idx, blk);  // the MTP KV entry of position idx (an MTP draft would have written it)
            for (uint32_t j = 1; j <= n; ++j) conf[j] = 1.f;
            return 1 + n;
        }
    }
    uint32_t T = 1;
    for (uint32_t j = 1; j < Tmax; ++j) {
        v[j] = e.mtp_draft(v[j - 1], idx + j - 1, blk);
        conf[j] = e.mtp_conf();
        if (j == 1 && g_spec_gate > 0 && conf[1] < g_spec_gate) return 1;
        if (j >= 2 && conf[j] < g_spec_keep) break;
        T = j + 1;
        if (conf[j] < g_spec_deeper) break;
    }
    return T;
}
static SessionStore* g_store = nullptr;
static bool g_ctx_save = false, g_ctx_load = false;
static int run_spec(FlashEngine& e, const std::vector<int64_t>& prompt, uint32_t gen, uint32_t k) {
    const uint32_t n = uint32_t(prompt.size()), want = gen + 1;
    std::vector<uint32_t> outputs;
    uint32_t next = 0;
    {
        auto tp = Clock::now();
        const EngineStats s0 = e.stats();
        auto& cs = e.streamer().cache().stats();
        const uint64_t h0 = cs.hits, m0 = cs.misses, r0 = e.streamer().stats().reads_issued;
        const DiskStat dk0 = DiskStat::now();
        size_t from = 0;
        if (g_ctx_load && g_store && g_store->on()) {  // S25: the prompt's state from the drive
            const auto h = g_store->best(prompt, n - 1);
            if (h.s >= 0) { const double tl = g_store->load(e, h); from = h.pos; std::fprintf(stderr, "ctx-load: %u of %u prompt tokens in %.3f s\n", h.pos, n, tl); }
            else std::fprintf(stderr, "ctx-load: nothing stored for this prompt\n");
        }
        prefill_range(e, prompt, from, n - 1, true, {}, [](size_t) { return true; });  // S20 batched
        if (g_ctx_save && g_store && g_store->on()) {  // S25: state after n - 1 prompt tokens
            g_store->bind(prompt, 0);
            g_store->commit(e, true);
        }
        next = e.feed(prompt[n - 1], true);
        g_prefill_s = secs(tp, Clock::now());
        const EngineStats& s1 = e.stats();
        const double nn = double(n);
        std::fprintf(stderr, "prefill anatomy (per token): wall %.1f ms, GPU+sync %.1f, GPU-pure %.1f, I/O wait %.1f, MTP fill %.2f, host %.2f | "
                             "hit %.3f, reads %.1f/tok | ",
                     g_prefill_s / nn * 1e3, (s1.t_gpu - s0.t_gpu) / nn * 1e3, (s1.t_gpu_pure - s0.t_gpu_pure) / nn * 1e3, (s1.t_io - s0.t_io) / nn * 1e3,
                     (s1.t_mtp_fill - s0.t_mtp_fill) / nn * 1e3, (s1.t_host - s0.t_host) / nn * 1e3,
                     double(cs.hits - h0) / double(std::max<uint64_t>(1, cs.hits - h0 + cs.misses - m0)), double(e.streamer().stats().reads_issued - r0) / nn);
        DiskStat::report(dk0, DiskStat::now(), g_prefill_s);
        std::fprintf(stderr, "prefill phases (ms per token): GPU MoE %.3f, GDN pre-step %.3f, attention pre-step %.3f, MTP KV fill %.3f | host n-gram rows %.3f, "
                             "MoE tables %.3f, mails+requests %.3f, embedding rows %.3f\n",
                     (s1.pf_gpu[0] - s0.pf_gpu[0]) / nn * 1e3, (s1.pf_gpu[1] - s0.pf_gpu[1]) / nn * 1e3, (s1.pf_gpu[2] - s0.pf_gpu[2]) / nn * 1e3,
                     (s1.pf_gpu[3] - s0.pf_gpu[3]) / nn * 1e3, (s1.pf_host[0] - s0.pf_host[0]) / nn * 1e3, (s1.pf_host[1] - s0.pf_host[1]) / nn * 1e3,
                     (s1.pf_host[2] - s0.pf_host[2]) / nn * 1e3, (s1.pf_host[3] - s0.pf_host[3]) / nn * 1e3);
        if (s1.pfa_layers > s0.pfa_layers)
            std::fprintf(stderr, "prefill lookahead: %.0f reads ahead per layer, %.3f of them routed, %.3f of the routed experts read ahead or resident\n",
                         double(s1.pfa_issued - s0.pfa_issued) / double(s1.pfa_layers - s0.pfa_layers),
                         double(s1.pfa_used - s0.pfa_used) / double(std::max<uint64_t>(1, s1.pfa_issued - s0.pfa_issued)),
                         1.0 - double(cs.misses - m0 - (s1.pfa_issued - s0.pfa_issued)) / double(std::max<uint64_t>(1, s1.pfa_uniq - s0.pfa_uniq)));
        if (s1.wide_pairs > s0.wide_pairs)
            std::fprintf(stderr, "wide MoE: %.2f pairs per expert-major entry (%llu entries)\n", double(s1.wide_pairs - s0.wide_pairs) / double(std::max<uint64_t>(1, s1.wide_ents - s0.wide_ents)),
                         (unsigned long long)(s1.wide_ents - s0.wide_ents));
        std::fprintf(stderr, "prefill %u tokens in %.2f s (%.1f tok/s, chunks of <= %u)\n", n, g_prefill_s, double(n) / g_prefill_s,
                     std::min<uint32_t>(e.prefill_max(), std::max<uint32_t>(g_prefill_tb, 1)));
    }
    outputs.push_back(next);
    auto t_gen0 = Clock::now();
    const uint64_t reads0 = e.streamer().stats().reads_issued;
    const DiskStat disk0 = DiskStat::now();
    int64_t cur = next;
    uint32_t blk = 0, idx = n - 1;  // the residual of position idx (whose next token is cur) is in activation block blk
    std::vector<int64_t> v(k + 1);
    std::vector<uint32_t> out(k + 1);
    uint64_t hist[16] = {}, cal_n[10] = {}, cal_ok[10] = {}, gated = 0, tsz[16] = {};
    double t_draft = 0, t_fill = 0, t_ver = 0, vio = 0, vgpu = 0, vpure = 0, vgb = 0, vga = 0, vdem = 0;
    while (outputs.size() < want && e.pos() + 1 < e.max_ctx()) {
        const uint32_t remaining = want - uint32_t(outputs.size());
        const uint32_t T = std::min<uint32_t>({k + 1, remaining, e.max_ctx() - e.pos()});
        if (T < 2) {  // last token: a plain step
            const uint32_t g = e.feed(cur, true);
            outputs.push_back(g);
            break;
        }
        auto a = Clock::now();
        v[0] = cur;
        float conf[8] = {};
        const uint32_t Tv = draft_chain(e, v, T, idx, blk, conf);
        auto b = Clock::now();
        if (Tv < 2) {  // S18 low-confidence draft: one plain token instead of a verify
            const uint32_t g = e.feed(cur, true);
            outputs.push_back(g);
            cur = g;
            blk = 0;
            idx = e.pos() - 1;
            ++gated;
            t_draft += std::chrono::duration<double>(b - a).count();
            continue;
        }
        const uint32_t p = e.pos();
        const EngineStats s0 = e.stats();
        ++tsz[Tv];
        const bool was_pld = g_pld.cur_n != 0;
        const uint32_t m = e.verify(v.data(), Tv, out.data());
        g_pld.report(m);
        {  // S18 per-cycle anatomy of the verify pass
            const EngineStats& s1 = e.stats();
            vio += s1.t_io - s0.t_io; vgpu += s1.t_gpu - s0.t_gpu; vpure += s1.t_gpu_pure - s0.t_gpu_pure;
            vgb += s1.gate_t[1] - s0.gate_t[1]; vga += s1.gate_t[0] - s0.gate_t[0]; vdem += double(s1.n_dem - s0.n_dem);
        }
        auto c = Clock::now();
        for (uint32_t i = 0; i <= m && outputs.size() < want; ++i) outputs.push_back(out[i]);
        for (uint32_t i = 0; i < m; ++i) e.mtp_fill(v[i + 1], p + i, i);
        t_draft += std::chrono::duration<double>(b - a).count();
        t_ver += std::chrono::duration<double>(c - b).count();
        t_fill += secs(c, Clock::now());
        ++hist[m];
        if (!was_pld) {  // S18 calibration: acceptance of draft 1 by its confidence bucket (MTP drafts only)
            const int bk = std::min(9, int(conf[1] * 10.f));
            ++cal_n[bk];
            cal_ok[bk] += m >= 1;
        }
        cur = out[m];
        blk = m;
        idx = p + m;
    }
    for (size_t i = 0; i < outputs.size(); ++i) std::printf("%u%s", outputs[i], i + 1 < outputs.size() ? "," : "\n");
    std::fflush(stdout);
    const double tg = secs(t_gen0, Clock::now());
    DiskStat::report(disk0, DiskStat::now(), tg);
    stats_line(e, n, uint32_t(outputs.size()), tg);
    auto& s = e.stats();
    std::fprintf(stderr, "reads/tok(gen) %.1f\n", outputs.size() > 1 ? double(e.streamer().stats().reads_issued - reads0) / double(outputs.size() - 1) : 0.0);
    std::fprintf(stderr, "spec k=%u: %llu verify cycles, accepted %llu of %llu drafts (%.3f), %.3f tokens/cycle; histogram", k,
                 (unsigned long long)s.spec_cycles, (unsigned long long)s.spec_accepted, (unsigned long long)s.spec_drafted,
                 s.spec_drafted ? double(s.spec_accepted) / double(s.spec_drafted) : 0.0,
                 s.spec_cycles ? double(s.spec_accepted + s.spec_cycles) / double(s.spec_cycles) : 0.0);
    for (uint32_t m = 0; m <= k; ++m) std::fprintf(stderr, " %u:%llu", m, (unsigned long long)hist[m]);
    const double cyc = double(std::max<uint64_t>(s.spec_cycles, 1));
    std::fprintf(stderr, "\nspec time per cycle: draft %.2f ms, verify %.2f ms, MTP KV fill %.2f ms\n",
                 t_draft / cyc * 1e3, t_ver / cyc * 1e3, t_fill / cyc * 1e3);
    std::fprintf(stderr, "draft-1 acceptance by MTP confidence (bucket: n acc):");
    for (int bk = 0; bk < 10; ++bk) if (cal_n[bk]) std::fprintf(stderr, " %.1f: %llu %.2f", bk / 10.0, (unsigned long long)cal_n[bk], double(cal_ok[bk]) / double(cal_n[bk]));
    std::fprintf(stderr, "\nspec gate %.2f deeper %.2f keep %.2f: %llu drafts not verified (plain tokens); batch sizes", g_spec_gate, g_spec_deeper, g_spec_keep,
                 (unsigned long long)gated);
    for (uint32_t t = 2; t <= k + 1; ++t) std::fprintf(stderr, " T%u:%llu", t, (unsigned long long)tsz[t]);
    if (g_pld.min_match)
        std::fprintf(stderr, "\nPLD (match >= %u): %llu matches, %llu cycles, accepted %llu of %llu drafts (%.3f), ema %.2f", g_pld.min_match,
                     (unsigned long long)g_pld.matches, (unsigned long long)g_pld.cycles, (unsigned long long)g_pld.accepted,
                     (unsigned long long)g_pld.drafted, g_pld.drafted ? double(g_pld.accepted) / double(g_pld.drafted) : 0.0, g_pld.ema);
    std::fprintf(stderr, "\nMTP drafts: %llu, %.2f ms each (expert I/O wait %.2f ms, %.2f lookups/draft); MTP-expert prediction precision %.3f, skipped (lossy) %.2f/draft\n",
                 (unsigned long long)s.mtp_drafts, s.mtp_drafts ? s.t_mtp / double(s.mtp_drafts) * 1e3 : 0.0,
                 s.mtp_drafts ? s.t_mtp_io / double(s.mtp_drafts) * 1e3 : 0.0, s.mtp_drafts ? double(s.mtp_demand) / double(s.mtp_drafts) : 0.0,
                 s.mtp_pred_total ? double(s.mtp_pred_ok) / double(s.mtp_pred_total) : 0.0, s.mtp_drafts ? double(s.mtp_skipped) / double(s.mtp_drafts) : 0.0);
    std::fprintf(stderr, "verify anatomy per cycle: host waits GPU %.2f ms, binds/I-O %.2f ms; GPU-pure %.2f ms, gate A %.2f ms, gate B %.2f ms; "
                         "demand misses %.1f (unique)\n", vgpu / cyc * 1e3, vio / cyc * 1e3, vpure / cyc * 1e3, vga / cyc * 1e3, vgb / cyc * 1e3, vdem / cyc);
    e.print_profile();
    return 0;
}


// S18 quality instrument: teacher-forced scoring of fixed texts against a reference model's top-20.
//   ids file: one document per line, comma-separated token ids
//   ref file: "doc pos id1,id2,...,id20" = the reference's top-20 next-token ids after token pos of doc
//   out:      "doc pos argmax lp_actual lp_id1,...,lp_id20"  (natural-log softmax of this engine's logits)
static size_t g_score_pf = 0;  // S21 --score-prefill M
static int run_score(FlashEngine& e, const std::string& ids_path, const std::string& ref_path, const std::string& out_path) {
    std::vector<std::vector<int64_t>> docs;
    {
        std::FILE* f = std::fopen(ids_path.c_str(), "r");
        if (!f) { std::fprintf(stderr, "score: cannot read %s\n", ids_path.c_str()); return 2; }
        std::string line;
        int ch;
        while ((ch = std::fgetc(f)) != EOF) {
            if (ch == '\n') { if (!line.empty()) docs.push_back(parse_ids(line)); line.clear(); } else line.push_back(char(ch));
        }
        if (!line.empty()) docs.push_back(parse_ids(line));
        std::fclose(f);
    }
    std::unordered_map<uint64_t, std::vector<int64_t>> ref;
    if (std::FILE* f = std::fopen(ref_path.c_str(), "r")) {
        unsigned d, q;
        char buf[4096];
        while (std::fscanf(f, "%u %u %4095s", &d, &q, buf) == 3) ref[(uint64_t(d) << 32) | q] = parse_ids(buf);
        std::fclose(f);
    }
    std::FILE* out = std::fopen(out_path.c_str(), "w");
    if (!out) return 2;
    const uint32_t V = e.vocab();
    double nll = 0;
    uint64_t n = 0;
    auto t0 = Clock::now();
    for (size_t d = 0; d < docs.size(); ++d) {
        e.reset();
        const auto& ids = docs[d];
        size_t q0 = 0;
        if (g_score_pf > 0 && ids.size() > g_score_pf + 1) {  // S21: the first M tokens through the prefill path, then score the rest
            prefill_range(e, ids, 0, g_score_pf, false, {}, [](size_t) { return true; });
            q0 = g_score_pf;
        }
        for (size_t q = q0; q + 1 < ids.size() && e.pos() + 1 < e.max_ctx(); ++q) {
            const uint32_t am = e.feed(ids[q], true);
            const float* lg = e.logits();
            float mx = -1e30f;
            for (uint32_t i = 0; i < V; ++i) mx = std::max(mx, lg[i]);
            double se = 0;
            for (uint32_t i = 0; i < V; ++i) se += std::exp(double(lg[i] - mx));
            const double lse = double(mx) + std::log(se);
            const double lpa = double(lg[ids[q + 1]]) - lse;
            nll -= lpa;
            ++n;
            std::fprintf(out, "%zu %zu %u %.6f ", d, q, am, lpa);
            auto it = ref.find((uint64_t(d) << 32) | q);
            if (it != ref.end())
                for (size_t j = 0; j < it->second.size(); ++j)
                    std::fprintf(out, "%s%.6f", j ? "," : "", it->second[j] >= 0 && it->second[j] < int64_t(V) ? double(lg[it->second[j]]) - lse : -1e9);
            std::fprintf(out, "\n");
        }
        std::fprintf(stderr, "score: doc %zu/%zu done (%zu tokens), running ppl %.3f\n", d + 1, docs.size(), ids.size(), std::exp(nll / double(std::max<uint64_t>(n, 1))));
    }
    std::fclose(out);
    std::fprintf(stderr, "score: %llu positions, ppl %.4f, %.1f s\n", (unsigned long long)n, std::exp(nll / double(std::max<uint64_t>(n, 1))), secs(t0, Clock::now()));
    std::fprintf(stderr, "score: hit %.3f, %.1f expert reads/token, %.2f tok/s\n", e.streamer().cache().stats().hit_rate(),
                 double(e.streamer().stats().reads_issued) / double(std::max<uint64_t>(n, 1)), double(n) / secs(t0, Clock::now()));
    route_line(e);
    return 0;
}

static int run_cli(FlashEngine& e, const std::vector<int64_t>& prompt, uint32_t gen, const std::string& dump_path, const std::string& trace_out,
                   const std::string& moe_path, uint32_t mtp_depth) {
    FILE* dumpf = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "wb");
    FILE* moef = moe_path.empty() ? nullptr : std::fopen(moe_path.c_str(), "wb");  // [token][layer][2560] f32
    const uint32_t total = uint32_t(prompt.size()) + gen;
    Clock::time_point t_gen0 = Clock::now();
    uint64_t reads0 = 0;
    std::vector<int64_t> outputs, toks;
    std::vector<std::vector<uint32_t>> drafts;
    double t_mtp_gen = 0;
    DiskStat disk0;
    int64_t tok = prompt[0];
    uint32_t t_first = 0;
    if (!dumpf && !moef && !mtp_depth && trace_out.empty() && e.full() && prompt.size() > 1 && g_prefill_tb > 1) {  // S20 batched prefill
        auto tp = Clock::now();
        prefill_range(e, prompt, 0, prompt.size() - 1, false, {}, [](size_t) { return true; });
        t_first = uint32_t(prompt.size() - 1);
        toks.assign(prompt.begin(), prompt.end() - 1);
        g_prefill_s = secs(tp, Clock::now());
        std::fprintf(stderr, "prefill %u tokens (batched) in %.2f s (%.1f tok/s)\n", t_first, g_prefill_s, double(t_first) / g_prefill_s);
    }
    for (uint32_t t = t_first; t < total && t < e.max_ctx(); ++t) {
        if (t < prompt.size()) tok = prompt[t];
        if (t == prompt.size()) { t_gen0 = Clock::now(); reads0 = e.streamer().stats().reads_issued; disk0 = DiskStat::now(); }
        toks.push_back(tok);
        const bool want = !e.full() || t + 1 >= prompt.size();
        const uint32_t next = e.feed(tok, want || dumpf);
        if (dumpf) std::fwrite(e.dump_area(), 4, e.dump_floats(), dumpf);
        if (moef) std::fwrite(e.moe_in(), 4, e.moe_in_floats(), moef);
        if (mtp_depth) {
            drafts.emplace_back();
            if (t + 1 < prompt.size()) e.mtp_fill(prompt[t + 1], t);
            else if (t + 1 < total && t + 1 + mtp_depth < e.max_ctx()) {
                auto a = Clock::now();
                uint32_t d = e.mtp_draft(next, t);
                drafts.back().push_back(d);
                for (uint32_t j = 1; j < mtp_depth; ++j) { d = e.mtp_draft(d, t + j); drafts.back().push_back(d); }
                if (t >= prompt.size()) t_mtp_gen += secs(a, Clock::now());
            }
        }
        if (e.full() && t + 1 >= prompt.size()) {
            outputs.push_back(next);
            tok = next;
            std::printf("%u%s", next, t + 1 < total ? "," : "\n");
            std::fflush(stdout);
        }
    }
    if (mtp_depth) {
        toks.push_back(tok);
        mtp_report(e.stats(), toks, drafts, mtp_depth);
        if (outputs.size() > 1)
            std::fprintf(stderr, "  trunk-only gen rate (MTP time removed): %.2f tok/s\n", double(outputs.size() - 1) / (secs(t_gen0, Clock::now()) - t_mtp_gen));
    }
    e.print_profile();
    if (dumpf) std::fclose(dumpf);
    if (moef) std::fclose(moef);
    if (!trace_out.empty()) {  // ps5lm-trace-txt/1: REAL routing of this run
        FILE* f = std::fopen(trace_out.c_str(), "w");
        const auto& tr = e.trace();
        const uint32_t nl = tr.empty() ? 0 : uint32_t(tr[0].size() / K);
        std::fprintf(f, "ps5lm-trace-txt/1 %zu %u %u\n", tr.size(), nl, E);
        for (auto& tk : tr)
            for (uint32_t l = 0; l < nl; ++l) {
                std::fprintf(f, "%u", K);
                for (uint32_t k = 0; k < K; ++k) std::fprintf(f, " %u", tk[l * K + k]);
                std::fputc('\n', f);
            }
        std::fclose(f);
    }
    const double tg = secs(t_gen0, Clock::now());
    DiskStat::report(disk0, DiskStat::now(), tg);
    {
        const auto& s = e.stats();
        const double tk = double(std::max<uint64_t>(s.tokens, 1));
        std::fprintf(stderr, "wait anatomy (all fed tokens, per token): ready layers %.1f (%.2f ms), prefetch-only %.1f (%.2f ms), demand %.1f (%.2f ms); "
                             "demand misses %.1f, prefetches in flight at bind %.1f\n  by #demand misses in the layer:",
                     double(s.layers_cls[0]) / tk, s.wait_cls[0] / tk * 1e3, double(s.layers_cls[1]) / tk, s.wait_cls[1] / tk * 1e3,
                     double(s.layers_cls[2]) / tk, s.wait_cls[2] / tk * 1e3, double(s.n_dem) / tk, double(s.n_pin) / tk);
        for (int d = 0; d < 8; ++d)
            if (s.layers_by_dem[d]) std::fprintf(stderr, " %d: %.1f layers/tok %.3f ms avg", d, double(s.layers_by_dem[d]) / tk, s.wait_by_dem[d] / double(s.layers_by_dem[d]) * 1e3);
        std::fprintf(stderr, "\n");
        if (s.gate_t[0] + s.gate_t[1] > 0)
            std::fprintf(stderr, "one-cb gates (GPU waiting, ms/token): A (resident known) %.2f, B (all ready) %.2f, N (n-gram) %.3f\n",
                         s.gate_t[0] / tk * 1e3, s.gate_t[1] / tk * 1e3, s.gate_t[2] / tk * 1e3);
        if (s.early_total)
            std::fprintf(stderr, "early routing: precision %.3f, %.1f prefetches/token\n", double(s.early_correct) / double(s.early_total),
                         double(s.early_issued) / tk);
        if (s.pref_used[0] + s.pref_unused[0] + s.pref_def_issued)
            std::fprintf(stderr, "prefetch outcome (per token): immediate used %.1f unused %.1f | deferred issued %.1f used %.1f unused %.1f dropped %.1f\n",
                         double(s.pref_used[0]) / tk, double(s.pref_unused[0]) / tk, double(s.pref_def_issued) / tk, double(s.pref_used[1]) / tk,
                         double(s.pref_unused[1]) / tk, double(s.pref_def_dropped) / tk);
    }
    stats_line(e, uint32_t(prompt.size()), uint32_t(outputs.size()), tg);
    std::fprintf(stderr, "reads/tok(gen) %.1f\n", outputs.size() > 1 ? double(e.streamer().stats().reads_issued - reads0) / double(outputs.size() - 1) : 0.0);
    return 0;
}

// ---------------------------------------------------------------- server loop
struct Request {
    uint32_t max_new = 256;
    SampleParams sp;
    std::vector<int64_t> ids, stop, snap;
};
static bool parse_request(const std::string& line, Request& r) {
    std::istringstream is(line.substr(1));
    std::string kv;
    while (is >> kv) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) return false;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "n") r.max_new = uint32_t(std::stoul(v));
        else if (k == "t") r.sp.temperature = std::stof(v);
        else if (k == "p") r.sp.top_p = std::stof(v);
        else if (k == "k") r.sp.top_k = uint32_t(std::stoul(v));
        else if (k == "m") r.sp.min_p = std::stof(v);
        else if (k == "r") r.sp.rep_pen = std::stof(v);
        else if (k == "pp") r.sp.presence = std::stof(v);
        else if (k == "fp") r.sp.frequency = std::stof(v);
        else if (k == "s") r.sp.seed = std::stoull(v);
        else if (k == "ids") r.ids = parse_ids(v);
        else if (k == "stop") r.stop = parse_ids(v);
        else if (k == "snap") r.snap = parse_ids(v);
    }
    return !r.ids.empty();
}

static int run_server(FlashEngine& e, const std::string& residency, uint32_t spec_k, SessionStore& store) {
    LineIn in;
    Sampler smp;
    std::string line;
    std::printf("READY slots=%u pool_gib=%.2f resident_gib=%.2f snapshots=%u max_ctx=%u sessions=%zu store_gb=%.2f\n", e.pool_slots(), e.pool_gib(),
                e.resident_gib(), e.n_snapshots(), e.max_ctx(), store.count(), store.used_gb());
    std::fflush(stdout);
    uint64_t requests = 0;
    while (in.next(line, true)) {
        if (line.empty()) continue;
        const char c = line[0];
        if (c == 'Q') break;
        if (c == 'X') continue;  // cancel with nothing running
        if (c == 'S' || c == 'W') {
            const std::string path = line.size() > 2 ? line.substr(2) : residency;
            if (c == 'S') std::printf("OK %d\n", e.save_residency(path) ? 1 : 0);
            else std::printf("OK %u\n", e.warm(path));
            std::fflush(stdout);
            continue;
        }
        if (c == 'I') {
            auto& cs = e.streamer().cache().stats();
            std::printf("I pos=%u requests=%llu hit=%.4f hits=%llu misses=%llu reads=%llu slots=%u sessions=%zu store_gb=%.2f\n", e.pos(),
                        (unsigned long long)requests, cs.hit_rate(), (unsigned long long)cs.hits, (unsigned long long)cs.misses,
                        (unsigned long long)e.streamer().stats().reads_issued, e.pool_slots(), store.count(), store.used_gb());
            std::fflush(stdout);
            continue;
        }
        if (c != 'G') { std::printf("D reason=error msg=unknown_command\n"); std::fflush(stdout); continue; }
        Request rq;
        if (!parse_request(line, rq)) { std::printf("D reason=error msg=bad_request\n"); std::fflush(stdout); continue; }
        ++requests;
        const auto& ids = rq.ids;
        const size_t n = ids.size();
        if (n >= e.max_ctx()) { std::printf("D reason=ctx prompt=%zu\n", n); std::fflush(stdout); continue; }
        smp.rng.seed(rq.sp.seed ? rq.sp.seed : 0x5eed + requests);

        // ---- choose the resume point ----
        auto t0 = Clock::now();
        const std::vector<int64_t> cur = e.tokens();
        size_t lcp = 0;
        while (lcp < cur.size() && lcp < n && cur[lcp] == ids[lcp]) ++lcp;
        const size_t limit = n - 1;  // the last prompt token must be fed to produce logits
        size_t start = 0;
        const char* how = "reset";
        // In memory: the live state if the request extends it, else the longest valid snapshot. On the drive (S25,
        // session_store.hpp): the longest stored resume point of any conversation. The drive wins when it holds at least
        // min_gain more tokens; it is decided BEFORE anything is reset, so the rows memory already holds are kept.
        int best = -1;
        size_t best_p = 0;
        const bool live = lcp == cur.size() && cur.size() <= limit;
        if (live) { start = cur.size(); how = "live"; }
        else {
            for (uint32_t s = 0; s < e.n_snapshots(); ++s) {
                auto st = e.snapshot_tokens(s);
                if (st.empty() || st.size() > limit || st.size() <= best_p) continue;
                if (std::equal(st.begin(), st.end(), ids.begin())) { best = int(s); best_p = st.size(); }
            }
            start = best_p;
        }
        double t_load = 0;
        SessionStore::Hit h;
        if (store.on()) h = store.best(ids, limit);
        if (h.s >= 0 && h.pos >= start + store.opt().min_gain) {
            try {
                t_load = store.load(e, h);
                start = h.pos;
                how = "disk";
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "session store: load failed (%s)\n", ex.what());
                store.drop(h.s);
                e.reset();
                e.adopt({}, {});  // the rows may be half replaced: nothing in memory is trusted
                start = 0;
                how = "reset";
            }
            g_pld.clear();
        } else if (!live) {
            if (best >= 0) { e.restore(uint32_t(best)); how = "snapshot"; }
            else e.reset();
            g_pld.clear();  // S23: the history was replaced; the PLD index is rebuilt from the new one
        }
        if (store.on()) {
            size_t sys_end = 0;
            for (int64_t sp : rq.snap) if (sp > 0 && size_t(sp) < n) { sys_end = size_t(sp); break; }
            store.bind(ids, sys_end);
        }
        size_t last_commit = start;
        // A requested snapshot position this request already covers (continued live state / restored past it) still
        // marks that snapshot as in use: an agent loop continues live every turn and would otherwise let the shared
        // system/tools snapshot age out of the LRU (S14b: the next conversation re-prefilled its 330-token tools block).
        for (int64_t sp : rq.snap)
            if (sp > 0 && size_t(sp) <= start)
                for (uint32_t s = 0; s < e.n_snapshots(); ++s) {
                    auto st = e.snapshot_tokens(s);
                    if (st.size() == size_t(sp) && std::equal(st.begin(), st.end(), ids.begin())) e.touch_snapshot(s);
                }
        // ---- prefill (S20: batched chunks of <= 4 tokens, cut at the requested snapshot positions) ----
        auto snap_here = [&](size_t p) {
            for (uint32_t s = 0; s < e.n_snapshots(); ++s) {
                auto st = e.snapshot_tokens(s);
                if (st.size() == p && std::equal(st.begin(), st.end(), ids.begin())) { e.touch_snapshot(s); return; }  // already have it
            }
            uint32_t victim = 0;
            for (uint32_t s = 1; s < e.n_snapshots(); ++s) if (e.snapshot_last_use(s) < e.snapshot_last_use(victim)) victim = s;
            if (e.n_snapshots()) e.snapshot(victim);
        };
        auto& cs = e.streamer().cache().stats();
        const uint64_t h0 = cs.hits, m0 = cs.misses, r0 = e.streamer().stats().reads_issued;
        const uint64_t tf0 = e.streamer().tier() ? e.streamer().tier()->stats().hits : 0;  // S31
        const double gpu0 = e.stats().t_gpu, io0 = e.stats().t_io;
        uint32_t next = 0;
        bool cancelled = false;
        const bool spec = spec_k && e.has_mtp() && e.batch_max() >= 2;
        size_t last_report = start;
        auto t_pf0 = Clock::now();
        auto t_rep = Clock::now();
        g_pf_progress = [&](size_t p) {
            if (secs(t_rep, Clock::now()) < 0.5) return;
            t_rep = Clock::now();
            std::printf("P %zu %zu\n", p, n);
            std::fflush(stdout);
        };
        const bool done_pf = prefill_range(e, ids, start, n - 1, spec, rq.snap, [&](size_t p) {
            for (int64_t sp : rq.snap) if (sp > 0 && size_t(sp) == p) snap_here(p);
            if (store.on() && store.opt().every && p - last_commit >= store.opt().every) { store.commit(e, true); last_commit = p; }
            if (p - last_report >= 32 || p == n - 1) {  // progress (the front end keeps long streams alive with it)
                last_report = p;
                std::printf("P %zu %zu\n", p, n);
                std::fflush(stdout);
            }
            std::string l2;  // cancellation during a long prefill
            while (in.next(l2, false)) if (!l2.empty() && l2[0] == 'X') cancelled = true;
            return !cancelled;
        });
        g_pf_progress = nullptr;
        if (done_pf) {
            next = e.feed(ids[n - 1], true);  // the last prompt token: logits (+ its MTP entry comes with the first draft)
            for (int64_t sp : rq.snap) if (sp > 0 && size_t(sp) == n) snap_here(n);
        }
        const double t_prefill = secs(t0, Clock::now());
        const double t_pf = secs(t_pf0, Clock::now());
        // ---- generate ----
        std::unordered_map<int64_t, uint32_t> counts;  // generated tokens only (see Sampler::sample)
        std::unordered_set<int64_t> prompt_seen;
        if (rq.sp.rep_pen != 1) prompt_seen.insert(ids.begin(), ids.end());
        auto t_g0 = Clock::now();
        uint32_t gen = 0;
        const char* reason = cancelled ? "cancel" : "length";
        uint64_t cyc = 0, acc = 0, drafted = 0;
        const uint64_t pld_c0 = g_pld.cycles, pld_a0 = g_pld.accepted, pld_d0 = g_pld.drafted;
        if (spec && !cancelled) {
            // S18 speculative generation: MTP drafts spec_k tokens, one verify pass checks them. Greedy requests accept the
            // engine's argmax; sampled ones sample every verified row with the request's sampler and accept while the
            // sample equals the draft (exact for a deterministic draft: P(d) = p(d), else p conditioned on != d).
            const bool greedy = rq.sp.greedy_gpu();
            auto emit = [&](uint32_t tok) -> bool {  // false: stop
                ++gen;
                std::printf("T %u\n", tok);
                std::fflush(stdout);
                if (std::find(rq.stop.begin(), rq.stop.end(), int64_t(tok)) != rq.stop.end()) { reason = "stop"; return false; }
                if (rq.sp.rep_pen != 1 || rq.sp.presence != 0 || rq.sp.frequency != 0) ++counts[tok];
                return gen < rq.max_new;
            };
            uint32_t pending = greedy ? next : smp.sample(e.logits(), e.vocab(), rq.sp, counts, prompt_seen);
            uint32_t blk = 0, idx = e.pos() - 1;
            std::vector<int64_t> v(spec_k + 1);
            std::vector<uint32_t> out(spec_k + 1), smpl(spec_k + 1);
            bool go = emit(pending);
            while (go) {
                if (e.pos() + 1 >= e.max_ctx()) { reason = "ctx"; break; }
                std::string l2;
                while (in.next(l2, false)) if (!l2.empty() && l2[0] == 'X') cancelled = true;
                if (cancelled) { reason = "cancel"; break; }
                const uint32_t T = std::min<uint32_t>({spec_k + 1, rq.max_new - gen + 1, e.max_ctx() - e.pos()});
                if (T < 2) {
                    const uint32_t g = e.feed(pending, true);
                    pending = greedy ? g : smp.sample(e.logits(), e.vocab(), rq.sp, counts, prompt_seen);
                    blk = 0; idx = e.pos() - 1;
                    go = emit(pending);
                    continue;
                }
                v[0] = pending;
                float conf[8] = {};
                const uint32_t Tv = draft_chain(e, v, T, idx, blk, conf);
                if (Tv < 2) {  // low-confidence draft: a plain token (its MTP KV entry is written)
                    const uint32_t g = e.feed(pending, true);
                    pending = greedy ? g : smp.sample(e.logits(), e.vocab(), rq.sp, counts, prompt_seen);
                    blk = 0; idx = e.pos() - 1;
                    go = emit(pending);
                    continue;
                }
                const uint32_t p = e.pos();
                FlashEngine::Decide dec;
                if (!greedy)
                    dec = [&](uint32_t TT, const uint32_t*) -> uint32_t {
                        auto cnt = counts;  // the penalties see the accepted prefix, row by row
                        const bool pen = rq.sp.rep_pen != 1 || rq.sp.presence != 0 || rq.sp.frequency != 0;
                        uint32_t m = 0;
                        for (uint32_t t = 0; t < TT; ++t) {
                            smpl[t] = smp.sample(e.logits_row(t), e.vocab(), rq.sp, cnt, prompt_seen);
                            if (t + 1 < TT && int64_t(smpl[t]) == v[t + 1]) { ++m; if (pen) ++cnt[smpl[t]]; } else break;
                        }
                        return m;
                    };
                const uint32_t m = e.verify(v.data(), Tv, out.data(), dec);
                g_pld.report(m);
                ++cyc; acc += m; drafted += Tv - 1;
                const uint32_t* res = greedy ? out.data() : smpl.data();
                for (uint32_t i = 0; i < m; ++i) e.mtp_fill(v[i + 1], p + i, i);
                for (uint32_t i = 0; i <= m && go; ++i) go = emit(res[i]);
                pending = res[m];
                blk = m;
                idx = p + m;
            }
        }
        while (!spec && !cancelled && gen < rq.max_new) {
            const uint32_t tok = rq.sp.greedy_gpu() ? next : smp.sample(e.logits(), e.vocab(), rq.sp, counts, prompt_seen);
            ++gen;
            std::printf("T %u\n", tok);
            std::fflush(stdout);
            if (std::find(rq.stop.begin(), rq.stop.end(), int64_t(tok)) != rq.stop.end()) { reason = "stop"; break; }
            if (gen >= rq.max_new) break;
            if (e.pos() + 1 >= e.max_ctx()) { reason = "ctx"; break; }
            std::string l2;
            while (in.next(l2, false)) if (!l2.empty() && l2[0] == 'X') cancelled = true;
            if (cancelled) { reason = "cancel"; break; }
            if (rq.sp.rep_pen != 1 || rq.sp.presence != 0 || rq.sp.frequency != 0) ++counts[tok];
            next = e.feed(tok, true);
        }
        const double t_gen = secs(t_g0, Clock::now());
        const uint64_t dh = cs.hits - h0, dm = cs.misses - m0;
        const uint64_t fed = (n - start) + (gen ? gen - 1 : 0);
        std::printf("D reason=%s prompt=%zu reused=%zu resume=%s prefill_s=%.3f prefill_tps=%.1f gen=%u gen_s=%.3f tps=%.2f hit=%.4f "
                    "reads_per_tok=%.1f gpu_ms=%.2f io_ms=%.2f pos=%u spec_cycles=%llu spec_acc=%.3f pld_cycles=%llu pld_acc=%.3f load_s=%.3f tier_per_tok=%.2f\n",
                    reason, n, start, how, t_prefill, t_pf > 0 ? double(n - start) / t_pf : 0.0, gen, t_gen, gen > 1 ? double(gen - 1) / t_gen : 0.0,
                    dh + dm ? double(dh) / double(dh + dm) : 0.0, fed ? double(e.streamer().stats().reads_issued - r0) / double(fed) : 0.0,
                    fed ? (e.stats().t_gpu - gpu0) / double(fed) * 1e3 : 0.0, fed ? (e.stats().t_io - io0) / double(fed) * 1e3 : 0.0, e.pos(),
                    (unsigned long long)cyc, drafted ? double(acc) / double(drafted) : 0.0, (unsigned long long)(g_pld.cycles - pld_c0),
                    g_pld.drafted > pld_d0 ? double(g_pld.accepted - pld_a0) / double(g_pld.drafted - pld_d0) : 0.0, t_load,
                    fed && e.streamer().tier() ? double(e.streamer().tier()->stats().hits - tf0) / double(fed) : 0.0);
        std::fflush(stdout);
        e.selstat_report();  // S32 tiering probe (PS5LM_SELSTAT=1), to stderr
        if (store.on()) store.commit(e, true);  // after the reply: the next request finds this one on the drive
    }
    if (!residency.empty()) e.save_residency(residency);
    return 0;
}

int main(int argc, char** argv) {
    EngineOptions o;
    std::string ids_s, dump_path, trace_out, residency, warm_file, moe_path, score_ids, score_ref, score_out = "score.txt";
    uint32_t gen = 16, mtp_depth = 0, spec_k = 0;
    bool serve = false;
    SessionStore::Opt so;
    std::string ctx_save, ctx_load;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--model") o.model_dir = nx();
        else if (a == "--shaders") o.shader_dir = nx();
        else if (a == "--ids") {  // S33: --ids @FILE (a long-context prompt exceeds the kernel's 128 KiB argument limit)
            ids_s = nx();
            if (!ids_s.empty() && ids_s[0] == '@') {
                std::ifstream f(ids_s.substr(1));
                if (!f) { std::fprintf(stderr, "cannot read %s\n", ids_s.c_str() + 1); return 2; }
                ids_s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
                while (!ids_s.empty() && (ids_s.back() == '\n' || ids_s.back() == ' ')) ids_s.pop_back();
            }
        }
        else if (a == "--gen") gen = uint32_t(std::stoul(nx()));
        else if (a == "--pool-gib") { const std::string v = nx(); o.pool_gib = v == "auto" ? 0 : std::stod(v); }
        else if (a == "--pool-reserve-mib") o.pool_reserve_mib = uint32_t(std::stoul(nx()));
        else if (a == "--force-pool") o.force_pool = true;
        else if (a == "--slab-hugetlb") o.slab_hugetlb = true;
        else if (a == "--no-slab-hugetlb") o.slab_hugetlb = false;
        else if (a == "--dump") dump_path = nx();
        else if (a == "--dump-moe") moe_path = nx();
        else if (a == "--max-ctx") o.max_ctx = uint32_t(std::stoul(nx()));
        else if (a == "--pack") o.pack_name = nx();
        else if (a == "--tier") {  // S31: CPU-only expert tier, DEV[:MiB] (e.g. /dev/ps5lowmem)
            const std::string v = nx();
            const size_t c = v.rfind(':');
            o.tier_dev = c == std::string::npos ? v : v.substr(0, c);
            if (c != std::string::npos) o.tier_mib = std::stoull(v.substr(c + 1));
        }
        else if (a == "--tier-threads") o.tier_threads = uint32_t(std::stoul(nx()));
        else if (a == "--tier-window") o.tier_window = uint32_t(std::stoul(nx()));
        else if (a == "--tier-bias") o.tier_bias = std::stof(nx());
        else if (a == "--tier-min-score") o.tier_min_score = std::stod(nx());
        else if (a == "--tier-min-ctx") o.tier_min_ctx = uint32_t(std::stoul(nx()));  // S31: admit only at >= this context
        else if (a == "--pack-mirror") o.pack_mirrors.push_back(nx());  // S23: copy of the pack on another device, path[:MBps]
        else if (a == "--pack-mbps") o.pack_mbps = std::stod(nx());
        else if (a == "--bits") o.bits = std::stoi(nx());
        else if (a == "--profile") o.profile = true;
        else if (a == "--predict") o.predict = uint32_t(std::stoul(nx()));
        else if (a == "--pred-dir") o.pred_dir = nx();
        else if (a == "--trace-out") trace_out = nx();
        else if (a == "--layers") o.layers = uint32_t(std::stoul(nx()));
        else if (a == "--snapshots") o.snapshots = uint32_t(std::stoul(nx()));
        else if (a == "--prefill-batch") { g_prefill_tb = uint32_t(std::stoul(nx())); o.prefill_tb = g_prefill_tb; }  // S20 (0/1 = serial)
        else if (a == "--prefill-chunk") o.prefill_chunk = uint32_t(std::stoul(nx()));  // S20 layer-major chunk (0 = off)
        else if (a == "--prefill-wide") o.wide = std::stoi(nx()) != 0;   // S21 wide prefill (GEMMs + expert-major MoE), default on
        else if (a == "--wide-slice") o.wide_slice = uint32_t(std::stoul(nx()));
        else if (a == "--wide-mtp") o.wide_mtp = uint32_t(std::stoul(nx()));  // S22: MTP KV fill of the prefill as GEMMs (0 = per token)
        else if (a == "--wide-xm") o.wide_xm = uint32_t(std::stoul(nx()));  // S22: chunk-wide expert-major MoE (0 = S21 per-slice)  // S21 tokens per wide slice (64..chunk, default 256)
        else if (a == "--prefill-lm-min") g_lm_min = std::stoul(nx());
        else if (a == "--pf-ahead") o.pf_ahead = uint32_t(std::stoul(nx()));  // S25: prefill chunk lookahead reads per layer (0 = off)
        else if (a == "--serve") serve = true;
        else if (a == "--kv-store") so.root = nx();            // S25: conversations on the SSD (server), directory
        else if (a == "--kv-store-gb") so.cap_gb = std::stod(nx());
        else if (a == "--kv-store-every") so.every = uint32_t(std::stoul(nx()));
        else if (a == "--kv-store-min") so.min_gain = uint32_t(std::stoul(nx()));
        else if (a == "--kv-store-gap") so.ck_gap = uint32_t(std::stoul(nx()));  // S30: resume point spacing kept on the drive
        else if (a == "--ctx-save") ctx_save = nx();           // S25 CLI test: store the state after the prompt in this directory
        else if (a == "--ctx-load") ctx_load = nx();           // ... and resume a prompt from it instead of prefilling
        else if (a == "--attn-legacy") o.attn_legacy = true;
        else if (a == "--hc-legacy") o.hc_mode = 0;
        else if (a == "--hc-mode") o.hc_mode = uint32_t(std::stoul(nx()));
        else if (a == "--no-overlap") o.overlap = false;
        else if (a == "--spin-wait") o.spin_wait = true;
        else if (a == "--ngram-sync") o.ngram_async = false;  // S15b: old serial n-gram preads (A/B)
        else if (a == "--no-size-classes") o.size_classes = false;  // S15b EXL3: one slot size (the largest) for all layers
        else if (a == "--hc-rows") o.hc_rows = uint32_t(std::stoul(nx()));
        else if (a == "--fused-moe") o.fused_moe = true;
        else if (a == "--down-rows") o.down_rows = uint32_t(std::stoul(nx()));
        else if (a == "--read-split") o.read_split = uint32_t(std::stoul(nx()));
        else if (a == "--prefetch-min-w") o.prefetch_min_w = std::stof(nx());
        else if (a == "--prefetch-max") { o.prefetch_max = uint32_t(std::stoul(nx())); o.prefetch_max_set = true; }
        else if (a == "--hot-prefetch") o.cold_prefetch = false;  // S13 behaviour: prefetches enter at MRU
        else if (a == "--residency") residency = nx();  // server: save on quit
        else if (a == "--warm") warm_file = nx();       // preload the pool (residency or profile file)
        else if (a == "--bar-nohost") vk::Ctx::g_barrier_host = false;  // S17 A/B: compute-only barriers inside command buffers
        else if (a == "--pf-lfu") o.pf_lfu = uint32_t(std::stoul(nx()));  // S22: prefill LFU stamps (0 chunk end, 1 per token, 2 none)
        else if (a == "--lfu") o.lfu_half_life = std::stod(nx());  // S17: decayed-frequency eviction, half-life in tokens (0 = LRU)
        else if (a == "--early") o.early = true;
        else if (a == "--no-router-fuse") o.router_fuse = false;   // S17 A/B
        else if (a == "--hc-fused") o.hc_mode = 3;                   // S17: norm folded into a 4-wave/row down + upmix2   // S17: early routing on the attention input (one-cb)
        else if (a == "--onecb") o.onecb = true;               // S17: one command buffer per token (GPU gates on host flags)
        else if (a == "--no-onecb") o.onecb = false;
        else if (a == "--no-hits-down") o.hits_down = false;  // S17 A/B: resident experts' act+down back in POSTMOE
        else if (a == "--pref-expire") o.pref_expire = true;   // S17: unused prefetches -> LRU tail at their layer's bind
        else if (a == "--defer-from") o.defer_from = uint32_t(std::stoul(nx()));  // S17: depths >= N wait for an idle SSD
        else if (a == "--defer-qd") o.defer_qd = uint32_t(std::stoul(nx()));
        else if (a == "--prefetch-late") o.prefetch_late = true;  // S16: prefetches after the layer's demand misses
        else if (a == "--spec") { spec_k = uint32_t(std::stoul(nx())); o.spec_k = spec_k; }
        else if (a == "--hp-late") o.hp_early = false;
        else if (a == "--score") score_ids = nx();       // S18 quality instrument (see run_score)
        else if (a == "--score-ref") score_ref = nx();
        else if (a == "--score-prefill") g_score_pf = std::stoul(nx());
        else if (a == "--score-out") score_out = nx();     // S18 A/B: reserve the slab hugepages after the weights (S17)
        else if (a == "--vb-off") o.vb_off = true;
        else if (a == "--spec-gate") g_spec_gate = std::stof(nx());  // S18: min MTP confidence to verify a draft
        else if (a == "--pld") g_pld.min_match = std::min<uint32_t>(32, uint32_t(std::stoul(nx())));  // S23: prompt-lookup drafts, match length (0 = off)
        else if (a == "--spec-deeper") g_spec_deeper = std::stof(nx());
        else if (a == "--mtp-pref") o.mtp_pref = uint32_t(std::stoul(nx()));  // S18: predicted MTP experts prefetched per verify (0 = off)
        else if (a == "--mtp-lfu-w") o.mtp_lfu_w = std::stod(nx());  // S18: LFU weight of MTP-layer uses
        else if (a == "--route-bias") o.route_bias = std::stof(nx());   // S24 cache-aware routing, trunk (LOSSY; logit units)
        else if (a == "--route-mtp") o.route_mtp = std::stof(nx());     // S24 ... of the MTP draft layer only (output unchanged)
        else if (a == "--route-cand") o.route_cand = std::max<uint32_t>(10, std::min<uint32_t>(16, uint32_t(std::stoul(nx()))));
        else if (a == "--route-keep") o.route_keep = std::min<uint32_t>(10, uint32_t(std::stoul(nx())));
        else if (a == "--route-pf") o.route_pf = true;
        else if (a == "--route-ready") o.route_ready = true;  // S24 A/B: only ready experts earn the bonus (measured: no gain)
        else if (a == "--route-fetch") o.route_fetch = uint32_t(std::stoul(nx()));
        else if (a == "--route-shadow") o.route_shadow = std::stof(nx());  // S35: LFU shadow use of displaced experts (0 = off)
        else if (a == "--route-shadow-fetch") o.route_shadow_fetch = std::stof(nx());  // S35: background read once the decayed count reaches T  // S24: background reads of displaced experts while < N in flight
        else if (a == "--mtp-lossy") o.mtp_lossy = true;    // S18: drafts use only the resident MTP experts
        else if (a == "--mtp-lossy-w") { o.mtp_lossy = true; o.mtp_lossy_w = std::stof(nx()); }
        else if (a == "--mtp-exact") o.mtp_lossy = false;
        else if (a == "--spec-keep") g_spec_keep = std::stof(nx());
        else if (a == "--vb-nodedupe") o.vb_nodedupe = true;
        else if (a == "--mv-old") o.mv_xrs = false;  // S19 A/B: the S18 dense kernels (matvec_lds / matvec_xr / xrb / ldsb)
        else if (a == "--hc-old") o.hc_fast = false;  // S19 A/B: S18 HC kernels (hc_norm, matvec.comp, hc_upmix)
        else if (a == "--tok-route") o.tok_route = uint32_t(std::stoul(nx()));  // S19: token-route table width (0 off)
        else if (a == "--ctx") o.max_ctx = uint32_t(std::stoul(nx()));        // S19/S20: context length (<= 262144)
        else if (a == "--kv16") o.kv16 = 1;                                   // S19: fp16 K/V cache (auto above 2048)
        else if (a == "--kv32") o.kv16 = 0;
        else if (a == "--kv-base") o.kv_base = uint32_t(std::stoul(nx()));  // S20 elastic KV (positions backed at start)
        else if (a == "--kv") { const std::string k = nx(); o.kvq = k == "q8" ? 8 : k == "k8v4" ? 4 : k == "k4v4" ? 2 : 0; if (k == "f16") o.kv16 = 1; if (k == "f32") o.kv16 = 0; }  // S20
        else if (a == "--no-indexer") o.indexer = false;                       // S19: dense attention beyond 2051 tokens
        else if (a == "--tok-route-mode") o.tok_route_mode = uint32_t(std::stoul(nx()));  // S19: 1 feed/verify, 2 draft->layer 0, 4 draft->MTP
        else if (a == "--ldsb-rpw") o.ldsb_rpw = uint32_t(std::stoul(nx()));  // S18 A/B: no unique-expert decode in verify
        else if (a == "--mtp-share-old") o.mtp_class_share = false;          // S18: stage-B verify (per-step submits) instead of the one-cb verify
        else if (a == "--vb-nobatch") o.vb_nobatch = true;  // S18 A/B: one-cb verify without the batched dense kernels  // S16: MTP drafts + batched verify
        else if (a == "--mtp") o.mtp = true;            // S16: load the MTP draft head (bits 3)
        else if (a == "--mtp-measure") { o.mtp = true; mtp_depth = uint32_t(std::stoul(nx())); }  // S16: acceptance of D-deep drafts
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (o.model_dir.empty() || (!serve && ids_s.empty() && score_ids.empty())) { std::fprintf(stderr, "need --model and --ids (or --serve)\n"); return 2; }
    o.dump = !dump_path.empty();
    o.dump_moe = !moe_path.empty();
    o.record_trace = !trace_out.empty();
    if (!serve) o.snapshots = 0;
    if (!ctx_save.empty() || !ctx_load.empty()) so.root = !ctx_save.empty() ? ctx_save : ctx_load;
    o.persist = !so.root.empty();
    so.tag = (o.pack_name.empty() ? std::string("default") : o.pack_name) + " kvq=" + std::to_string(o.kvq) + " kv16=" + std::to_string(o.kv16);
    // Under memory pressure the kernel must kill US, never sshd/NetworkManager (S14: it killed both).
    if (FILE* f = std::fopen("/proc/self/oom_score_adj", "w")) { std::fputs("1000", f); std::fclose(f); }
    for (int sg : {SIGTERM, SIGHUP, SIGINT, SIGSEGV, SIGBUS, SIGABRT, SIGFPE}) std::signal(sg, hugepage_signal_restore);  // S15: never leak the hugetlb slab (S16: crashes too)
    // Served engine: lock our pages so the 256 MiB swap (/dev/ps5lowswap, swappiness 10) only ever takes other
    // processes' memory (the front end, sshd, other daemons). ONFAULT: only pages we touch are locked, so reserved-but-
    // unbacked regions (elastic KV, prefill carry) stay unbacked. Hugetlb/GPU/VM_IO (lowmem tier) maps are unaffected.
    if (serve && mlockall(MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT) != 0)
        std::fprintf(stderr, "mlockall failed (errno %d): engine pages stay swappable\n", errno);
    try {
        platform::pin_to_cpu(2);
        FlashEngine e(o);
        if (!warm_file.empty()) {
            auto t = Clock::now();
            const uint32_t n = e.warm(warm_file);
            std::fprintf(stderr, "warm: %u experts preloaded in %.2f s\n", n, secs(t, Clock::now()));
        }
        SessionStore store;
        if (!so.root.empty()) store.open(e, so);
        g_store = &store;
        g_ctx_save = !ctx_save.empty();
        g_ctx_load = !ctx_load.empty();
        if (serve) return run_server(e, residency, spec_k, store);
        if (!score_ids.empty()) return run_score(e, score_ids, score_ref, score_out);
        if (spec_k) return run_spec(e, parse_ids(ids_s), gen, spec_k);
        return run_cli(e, parse_ids(ids_s), gen, dump_path, trace_out, moe_path, mtp_depth);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
