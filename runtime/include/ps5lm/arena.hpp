// arena.hpp — zero-copy tensor memory manager (portable core; no glibc malloc
// on the hot path).
//
// The engine owns exactly one pinned host mapping per budget class (from
// platform::map_pinned). That mapping is registered once with io_uring and
// once with the GPU (hipHostRegister / vkImportMemoryHostPointerEXT); every
// tensor is an OFFSET into it, so the same bytes are visible to the NVMe DMA
// engine, the CPU and the GPU with zero copies.
//
// Layout of the 16 GiB budget (RUNTIME-SPEC §2.3), carved at load, never freed:
//   [ resident weights | shared experts | KV | activations | staging | expert slab ]
// Regions are bump-allocated with 4 KiB (O_DIRECT) or 256 B (GPU cacheline)
// alignment. The expert slab is a fixed-size slot array managed by SlotCache.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps5lm {

inline constexpr size_t kDirectIoAlign = 4096;  // O_DIRECT + registered buffers
inline constexpr size_t kGpuAlign = 256;        // RDNA2 L2 line pairs / buffer offsets

struct Region {
    std::string name;
    size_t offset = 0;
    size_t bytes = 0;
};

// A tensor view: device code receives base_device + offset.
struct TensorRef {
    size_t offset = 0;
    size_t bytes = 0;
};

class Arena {
public:
    Arena() = default;
    Arena(std::byte* base, size_t capacity) : base_(base), cap_(capacity) {
        if (!base || !capacity) throw std::invalid_argument("Arena: empty backing");
        if (reinterpret_cast<uintptr_t>(base) % kDirectIoAlign) throw std::invalid_argument("Arena: base not 4K aligned");
    }

    // Carve a named top-level region (done once at load).
    Region& carve(const std::string& name, size_t bytes, size_t align = kDirectIoAlign) {
        size_t off = align_up(top_, align);
        if (off + bytes > cap_)
            throw std::length_error("Arena: region '" + name + "' (" + std::to_string(bytes) + " B) exceeds budget; " +
                                    std::to_string(cap_ - off) + " B left");
        regions_.push_back({name, off, bytes});
        top_ = off + bytes;
        return regions_.back();
    }

    // Whatever is left (the expert slab takes the remainder), rounded down to `unit`.
    Region& carve_rest(const std::string& name, size_t unit, size_t align = kDirectIoAlign) {
        size_t off = align_up(top_, align);
        if (off >= cap_) throw std::length_error("Arena: no space left for '" + name + "'");
        size_t bytes = (cap_ - off) / unit * unit;
        if (!bytes) throw std::length_error("Arena: remainder smaller than one unit for '" + name + "'");
        regions_.push_back({name, off, bytes});
        top_ = off + bytes;
        return regions_.back();
    }

    const Region& region(const std::string& name) const {
        for (auto& r : regions_) if (r.name == name) return r;
        throw std::out_of_range("Arena: no region " + name);
    }

    std::byte* host(size_t offset) const { return base_ + offset; }
    std::byte* base() const { return base_; }
    size_t capacity() const { return cap_; }
    size_t used() const { return top_; }
    const std::vector<Region>& regions() const { return regions_; }

    static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

private:
    std::byte* base_ = nullptr;
    size_t cap_ = 0, top_ = 0;
    std::vector<Region> regions_;
};

// Sub-allocator inside one region (tensors of the resident model, KV pages...).
class RegionBump {
public:
    RegionBump(const Arena& a, const Region& r) : arena_(&a), region_(r) {}
    TensorRef alloc(size_t bytes, size_t align = kGpuAlign) {
        size_t off = Arena::align_up(region_.offset + top_, align);
        if (off + bytes > region_.offset + region_.bytes)
            throw std::length_error("RegionBump: '" + region_.name + "' exhausted");
        top_ = off + bytes - region_.offset;
        return {off, bytes};
    }
    void reset() { top_ = 0; }  // activations: reset per token, never freed piecewise
    size_t used() const { return top_; }
    std::byte* host(const TensorRef& t) const { return arena_->host(t.offset); }

private:
    const Arena* arena_;
    Region region_;
    size_t top_ = 0;
};

}  // namespace ps5lm
