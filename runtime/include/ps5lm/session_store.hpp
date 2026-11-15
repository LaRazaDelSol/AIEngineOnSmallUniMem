// session_store.hpp — S25: conversations kept on the SSD (docs/PERSISTENT-CONTEXT.md).
//
// A position of context costs ~3 ms of GPU to compute and 19 KB to store, and the drive reads 6 GB/s: reading a
// conversation back is 300-1000x faster than prefilling it again. The store keeps, per conversation (a directory):
//   kv        the per-position caches (K, V, indexer keys of the 12 attention layers + the MTP layer), rows [0, n)
//   rs.<pos>  the recurrent state (GatedDeltaNet S, conv, PLE conv: 118 MB) at a few positions = resume points
//   meta      the n tokens the rows were computed from, the checkpoint positions, a format fingerprint
// A request resumes from the longest checkpoint whose tokens are a prefix of its ids, in ANY stored conversation, when
// that beats what the engine holds in memory (live state / in-memory snapshots). After every request (and every
// `every` tokens of a long prefill) the rows written since the last save and the new resume points go to the drive, so
// a restart, an OOM kill or a switch to another conversation costs a read instead of a prefill.
//
// Crash safety: rows are only ever described by `meta` after they are on the device (O_DIRECT + fdatasync, then
// meta.tmp -> rename). Before rows that meta already describes are overwritten, meta is cut back first.
#pragma once

#include "ps5lm/flash_engine.hpp"

#include <dirent.h>

#include <set>

namespace ps5lm::flash {

class SessionStore {
public:
    struct Opt {
        std::string root;
        std::string tag;            // model / pack / KV format: sessions written under another tag are never loaded
        double cap_gb = 16;         // least recently used conversations are deleted above this
        uint32_t max_sessions = 16;
        // S30: resume points are kept along the WHOLE conversation, one per ck_gap tokens (+ the first and the newest
        // two). An agent client that prunes an old tool result at depth D makes every later token differ; the request
        // resumes from the last point <= D. With only the first + 3 newest points kept (S25) that was the system block:
        // a prune at 125k of a 191k conversation re-prefilled 160k tokens (11 min, past the client's 300 s timeout).
        uint32_t max_ckpt = 48;     // resume points kept per conversation at most (meta allows 64)
        uint32_t ck_gap = 8192;     // spacing of the kept resume points below the newest two
        uint32_t min_gain = 256;    // load from the drive only if it saves at least this many tokens of prefill
        uint32_t live_min = 256;    // write the end-of-request state only if it is this far past the last resume point
        uint32_t every = 8192;      // during a long prefill: save every this many tokens (0 = only at the end)
    };
    struct Hit { int s = -1; uint32_t pos = 0; };

    bool on() const { return on_; }
    const Opt& opt() const { return o_; }
    size_t count() const { size_t n = 0; for (auto& s : ss_) n += !s.foreign; return n; }
    double used_gb() const { uint64_t t = 0; for (auto& s : ss_) t += s.bytes; return double(t) / 1e9; }

    bool open(FlashEngine& e, const Opt& o) {
        o_ = o;
        if (o_.root.empty()) return false;
        if (!e.persist_ok()) { std::fprintf(stderr, "session store: off (needs a context length that is a multiple of %u)\n", FlashEngine::kKvGran); return false; }
        char b[256];
        std::snprintf(b, sizeof b, "v1 %s bits=%d caches=%zu row=%llu rs=%llu", o_.tag.c_str(), e.bits(), e.kv_caches(),
                      (unsigned long long)e.kv_row_bytes(), (unsigned long long)e.rstate_bytes());
        fp_ = b;
        for (size_t i = 0; i < e.kv_caches(); ++i) fp_ += (i ? "," : " pp=") + std::to_string(e.kv_per_pos(i));
        row_bytes_ = e.kv_row_bytes();
        rs_bytes_ = e.rstate_bytes();
        ::mkdir(o_.root.c_str(), 0755);
        DIR* d = ::opendir(o_.root.c_str());
        if (!d) { std::fprintf(stderr, "session store: cannot open %s\n", o_.root.c_str()); return false; }
        while (dirent* de = ::readdir(d)) {
            if (de->d_name[0] != 's') continue;
            Sess s;
            s.dir = o_.root + "/" + de->d_name;
            next_id_ = std::max<uint64_t>(next_id_, std::strtoull(de->d_name + 1, nullptr, 10) + 1);
            if (!read_meta(s)) { remove_dir(s.dir); continue; }  // never completed its first commit
            s.bytes = disk_bytes(s);
            clock_ = std::max(clock_, s.last_use);
            ss_.push_back(std::move(s));
        }
        ::closedir(d);
        on_ = true;
        std::fprintf(stderr, "session store: %s, %zu conversations (%.2f GB of %.0f GB), %llu B per position + %.0f MB per resume point\n",
                     o_.root.c_str(), count(), used_gb(), o_.cap_gb, (unsigned long long)row_bytes_, double(rs_bytes_) / 1e6);
        return true;
    }

    // The longest stored resume point that is a prefix of ids (position <= limit)
    Hit best(const std::vector<int64_t>& ids, size_t limit) const {
        Hit h;
        for (size_t i = 0; i < ss_.size(); ++i) {
            const auto& s = ss_[i];
            if (s.foreign || s.ck.empty() || s.ck.front() > limit) continue;
            const size_t l = std::min(lcp(s.toks, ids), limit);
            for (auto it = s.ck.rbegin(); it != s.ck.rend(); ++it)
                if (*it <= l) { if (*it > h.pos) { h.s = int(i); h.pos = *it; } break; }
        }
        return h;
    }

    // Engine -> position h.pos of conversation h.s. Returns seconds. Throws on I/O errors (the caller resets the engine).
    double load(FlashEngine& e, const Hit& h) {
        auto t0 = Clock::now();
        auto& s = ss_.at(size_t(h.s));
        // rows the engine already holds for the same tokens stay (two conversations behind one system prompt)
        const auto& kt = e.kv_tokens();
        size_t m = 0;
        while (m < kt.size() && m < h.pos && kt[m] == int64_t(s.toks[m])) ++m;
        const uint64_t lo = m / G * G, hi = std::min<uint64_t>((uint64_t(h.pos) + G - 1) / G * G, e.max_ctx());
        Fd kv(s.dir + "/kv", O_RDONLY | O_DIRECT);
        e.kv_io(kv.fd, false, lo, hi);
        Fd rs(s.dir + "/rs." + std::to_string(h.pos), O_RDONLY | O_DIRECT);
        e.rstate_io(rs.fd, false);
        std::vector<int64_t> toks(s.toks.begin(), s.toks.begin() + h.pos);
        std::vector<int64_t> kvt(s.toks.begin(), s.toks.begin() + std::min<size_t>(s.toks.size(), hi));
        e.adopt(toks, kvt);
        s.last_use = ++clock_;
        const double t_rows = secs(t0, Clock::now());
        // S30: the experts this conversation had resident at its last save, most recently used first. Without them the
        // first few hundred tokens after a restore ran at hit 0.70 / 228 reads per token (194k positions) instead of ~0.80.
        const uint32_t warmed = e.warm(s.dir + "/res", false);
        const double t = secs(t0, Clock::now());
        std::fprintf(stderr, "session store: resumed %u tokens of %s from the drive in %.3f s (rows %llu-%llu, %.0f MB; %u experts warmed in %.3f s)\n",
                     h.pos, s.dir.c_str(), t, (unsigned long long)lo, (unsigned long long)hi, double((hi - lo) * row_bytes_ + rs_bytes_) / 1e6, warmed,
                     t - t_rows);
        return t;
    }

    // A load failed half-way: the conversation is unusable, forget it
    void drop(int s) {
        if (s < 0 || size_t(s) >= ss_.size()) return;
        remove_dir(ss_[size_t(s)].dir);
        ss_.erase(ss_.begin() + s);
        bound_ = -1;
    }

    // Which stored conversation this request continues (-1: a new one, created by the first commit). It is the same
    // conversation when the request shares more than the system/tools block with it and at least half of either side.
    void bind(const std::vector<int64_t>& ids, size_t sys_end) {
        bound_ = -1;
        size_t bl = 0;
        for (size_t i = 0; i < ss_.size(); ++i) {
            const auto& s = ss_[i];
            if (s.foreign) continue;
            const size_t l = lcp(s.toks, ids);
            if (l > sys_end && l >= 16 && (2 * l >= s.toks.size() || 2 * l >= ids.size()) && l > bl) { bl = l; bound_ = int(i); }
        }
    }

    // Engine state -> the bound conversation: rows written since its last save, the in-memory snapshots that are not on
    // the drive yet, and (live) the state at the current position. Returns seconds.
    double commit(FlashEngine& e, bool live) {
        const uint32_t pos = e.pos();
        if (!on_ || !pos) return 0;
        auto t0 = Clock::now();
        try {
            if (bound_ < 0) {
                make_room(1);
                Sess s;
                char b[32];
                std::snprintf(b, sizeof b, "/s%06llu", (unsigned long long)next_id_++);
                s.dir = o_.root + b;
                if (::mkdir(s.dir.c_str(), 0755) != 0) throw std::runtime_error("mkdir " + s.dir);
                ss_.push_back(std::move(s));
                bound_ = int(ss_.size() - 1);
            }
            auto& s = ss_[size_t(bound_)];
            std::vector<int32_t> toks(pos);
            for (uint32_t i = 0; i < pos; ++i) toks[i] = int32_t(e.token_at(i));
            size_t l = 0;
            while (l < s.toks.size() && l < pos && s.toks[l] == toks[l]) ++l;
            if (l < s.toks.size()) {  // rows that meta describes are about to be replaced: cut meta back first
                s.toks.resize(l);
                while (!s.ck.empty() && s.ck.back() > l) { ::unlink(rs_path(s, s.ck.back()).c_str()); s.ck.pop_back(); }
                write_meta(s);
            }
            const uint64_t lo = l / G * G, hi = std::min<uint64_t>((uint64_t(pos) + G - 1) / G * G, e.kv_rows_backed());
            uint64_t wrote = 0;
            {
                Fd kv(s.dir + "/kv", O_RDWR | O_CREAT | O_DIRECT);
                if (hi > lo) { e.kv_io(kv.fd, true, lo, hi); wrote += (hi - lo) * row_bytes_; }
                const uint64_t end = (hi + FlashEngine::kKvBlock - 1) / FlashEngine::kKvBlock * FlashEngine::kKvBlock * row_bytes_;
                if (::ftruncate(kv.fd, off_t(end)) != 0) throw std::runtime_error("ftruncate kv");
                ::fdatasync(kv.fd);
            }
            std::set<uint32_t> have(s.ck.begin(), s.ck.end());
            auto add_ck = [&](uint32_t p, int slot) {
                const std::string f = rs_path(s, p), tmp = f + ".tmp";
                {
                    Fd rs(tmp, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT);
                    if (slot >= 0) { if (!e.snapshot_write(uint32_t(slot), rs.fd)) return; }
                    else e.rstate_io(rs.fd, true);
                    ::fdatasync(rs.fd);
                }
                if (::rename(tmp.c_str(), f.c_str()) != 0) throw std::runtime_error("rename " + tmp);
                have.insert(p);
                wrote += rs_bytes_;
            };
            for (uint32_t k = 0; k < e.n_snapshots(); ++k) {
                const auto st = e.snapshot_tokens(k);
                const size_t p = st.size();
                if (!p || p > pos || p > hi || have.count(uint32_t(p))) continue;
                bool same = true;
                for (size_t i = 0; i < p && same; ++i) same = st[i] == int64_t(toks[i]);
                if (same) add_ck(uint32_t(p), int(k));
            }
            if (live && !have.count(pos)) {
                uint32_t last = 0;
                for (uint32_t c : have) if (c <= pos) last = std::max(last, c);
                if (pos - last >= o_.live_min || have.empty()) add_ck(pos, -1);
            }
            s.ck = thin(std::vector<uint32_t>(have.begin(), have.end()), s);
            if (live) {  // S30: the pool's residency for load() (text, ~60 KB); tmp + rename so a crash leaves the old one
                const std::string rf = s.dir + "/res";
                if (e.save_residency(rf + ".tmp")) ::rename((rf + ".tmp").c_str(), rf.c_str());
            }
            s.toks = std::move(toks);
            if (s.toks.size() > hi) s.toks.resize(hi);
            s.last_use = ++clock_;
            write_meta(s);
            s.bytes = disk_bytes(s);
            make_room(0);
            const double t = secs(t0, Clock::now());
            std::fprintf(stderr, "session store: saved %s at %u tokens (%.0f MB written, %zu resume points) in %.3f s\n", s.dir.c_str(), pos,
                         double(wrote) / 1e6, s.ck.size(), t);
            return t;
        } catch (const std::exception& ex) {  // a full or failing drive must never take the server down
            std::fprintf(stderr, "session store: save failed (%s); the conversation stays in memory only\n", ex.what());
            if (bound_ >= 0) drop(bound_);
            return secs(t0, Clock::now());
        }
    }

private:
    static constexpr uint64_t G = FlashEngine::kKvGran;
    struct Sess {
        std::string dir;
        std::vector<int32_t> toks;   // what the rows on the drive were computed from
        std::vector<uint32_t> ck;    // resume points, ascending
        uint64_t last_use = 0, bytes = 0;
        bool foreign = false;        // written under another fingerprint: only counted and evicted
    };
    struct Fd {
        int fd;
        Fd(const std::string& p, int flags) : fd(::open(p.c_str(), flags | O_CLOEXEC, 0644)) {
            if (fd < 0) throw std::runtime_error("open " + p + ": " + std::strerror(errno));
        }
        ~Fd() { ::close(fd); }
        Fd(const Fd&) = delete;
    };
    static size_t lcp(const std::vector<int32_t>& a, const std::vector<int64_t>& b) {
        size_t l = 0;
        while (l < a.size() && l < b.size() && int64_t(a[l]) == b[l]) ++l;
        return l;
    }
    static std::string rs_path(const Sess& s, uint32_t p) { return s.dir + "/rs." + std::to_string(p); }
    // S30 retention (ascending in, ascending out; dropped files unlinked): the first point (system block), one point per
    // ck_gap tokens below the newest two, the newest two. Above max_ckpt the oldest after the first go.
    std::vector<uint32_t> thin(const std::vector<uint32_t>& ck, const Sess& s) const {
        const size_t n = ck.size(), cap = std::min<size_t>(std::max<uint32_t>(o_.max_ckpt, 3), 64);
        std::vector<uint32_t> keep;
        for (size_t i = 0; i < n; ++i)
            if (i == 0 || i + 2 >= n || ck[i] - keep.back() >= o_.ck_gap) keep.push_back(ck[i]);
        while (keep.size() > cap) keep.erase(keep.begin() + 1);
        for (uint32_t c : ck)
            if (!std::binary_search(keep.begin(), keep.end(), c)) ::unlink(rs_path(s, c).c_str());
        return keep;
    }
    static uint64_t file_bytes(const std::string& p) {
        struct stat st;
        return ::stat(p.c_str(), &st) == 0 ? uint64_t(st.st_blocks) * 512 : 0;
    }
    uint64_t disk_bytes(const Sess& s) const {
        uint64_t t = 0;
        if (DIR* d = ::opendir(s.dir.c_str())) {
            while (dirent* de = ::readdir(d)) if (de->d_name[0] != '.') t += file_bytes(s.dir + "/" + de->d_name);
            ::closedir(d);
        }
        return t;
    }
    static void remove_dir(const std::string& dir) {
        if (DIR* d = ::opendir(dir.c_str())) {
            while (dirent* de = ::readdir(d)) if (de->d_name[0] != '.') ::unlink((dir + "/" + de->d_name).c_str());
            ::closedir(d);
        }
        ::rmdir(dir.c_str());
    }
    // meta: "PS5LMSS1" u32 fp_len fp u64 last_use u32 n_tok u32 n_ck ck[] toks[]
    void write_meta(const Sess& s) const {
        const std::string f = s.dir + "/meta", tmp = f + ".tmp";
        std::vector<uint8_t> b;
        auto put = [&](const void* p, size_t n) { const auto* q = static_cast<const uint8_t*>(p); b.insert(b.end(), q, q + n); };
        const uint32_t fl = uint32_t(fp_.size()), nt = uint32_t(s.toks.size()), nc = uint32_t(s.ck.size());
        put("PS5LMSS1", 8); put(&fl, 4); put(fp_.data(), fl); put(&s.last_use, 8); put(&nt, 4); put(&nc, 4);
        put(s.ck.data(), size_t(nc) * 4); put(s.toks.data(), size_t(nt) * 4);
        {
            Fd o(tmp, O_WRONLY | O_CREAT | O_TRUNC);
            size_t done = 0;
            while (done < b.size()) { const ssize_t r = ::write(o.fd, b.data() + done, b.size() - done); if (r <= 0) throw std::runtime_error("meta write"); done += size_t(r); }
            ::fsync(o.fd);
            ::posix_fadvise(o.fd, 0, 0, POSIX_FADV_DONTNEED);
        }
        if (::rename(tmp.c_str(), f.c_str()) != 0) throw std::runtime_error("meta rename");
    }
    bool read_meta(Sess& s) const {
        std::vector<uint8_t> b;
        try { b = read_file(s.dir + "/meta"); } catch (const std::exception&) { return false; }
        size_t q = 0;
        auto get = [&](void* p, size_t n) { if (q + n > b.size()) return false; std::memcpy(p, b.data() + q, n); q += n; return true; };
        char mg[8];
        uint32_t fl = 0, nt = 0, nc = 0;
        if (!get(mg, 8) || std::memcmp(mg, "PS5LMSS1", 8) != 0 || !get(&fl, 4) || fl > 4096) return false;
        std::string fp(fl, 0);
        if (!get(fp.data(), fl) || !get(&s.last_use, 8) || !get(&nt, 4) || !get(&nc, 4) || nc > 64) return false;
        s.ck.resize(nc);
        s.toks.resize(nt);
        if (!get(s.ck.data(), size_t(nc) * 4) || !get(s.toks.data(), size_t(nt) * 4)) return false;
        s.foreign = fp != fp_;
        if (s.foreign) return true;
        // a resume point is usable only if its file is complete and its rows are described
        std::vector<uint32_t> ok;
        for (uint32_t c : s.ck) {
            struct stat st;
            if (c <= nt && ::stat(rs_path(s, c).c_str(), &st) == 0 && uint64_t(st.st_size) >= rs_bytes_) ok.push_back(c);
        }
        s.ck = ok;
        return true;
    }
    // Delete least recently used conversations (never the bound one) until `extra` more fit
    void make_room(uint32_t extra) {
        for (;;) {
            uint64_t tot = 0;
            for (auto& s : ss_) tot += s.bytes;
            if (ss_.size() + extra <= o_.max_sessions && double(tot) <= o_.cap_gb * 1e9) return;
            int v = -1;
            for (size_t i = 0; i < ss_.size(); ++i)
                if (int(i) != bound_ && (v < 0 || ss_[i].foreign > ss_[size_t(v)].foreign ||
                                         (ss_[i].foreign == ss_[size_t(v)].foreign && ss_[i].last_use < ss_[size_t(v)].last_use)))
                    v = int(i);
            if (v < 0) return;
            std::fprintf(stderr, "session store: evicting %s (%.2f GB)\n", ss_[size_t(v)].dir.c_str(), double(ss_[size_t(v)].bytes) / 1e9);
            remove_dir(ss_[size_t(v)].dir);
            ss_.erase(ss_.begin() + v);
            if (bound_ > v) --bound_;
        }
    }

    Opt o_;
    bool on_ = false;
    std::string fp_;
    uint64_t row_bytes_ = 0, rs_bytes_ = 0, clock_ = 0, next_id_ = 1;
    std::vector<Sess> ss_;
    int bound_ = -1;
};

}  // namespace ps5lm::flash
