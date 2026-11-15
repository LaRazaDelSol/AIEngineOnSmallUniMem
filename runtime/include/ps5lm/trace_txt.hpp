// trace_txt.hpp — reader for ps5lm-trace-txt/1 (see runtime/tools/trace_to_txt.py).
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps5lm {

struct Trace {
    uint32_t tokens = 0, layers = 0, experts = 0;
    // Flattened routing: ids for (token t, layer l) are
    // ids[offs[t*layers+l] .. offs[t*layers+l+1]).
    std::vector<uint16_t> ids;
    std::vector<uint32_t> offs;

    const uint16_t* begin(uint32_t t, uint32_t l) const { return ids.data() + offs[size_t(t) * layers + l]; }
    const uint16_t* end(uint32_t t, uint32_t l) const { return ids.data() + offs[size_t(t) * layers + l + 1]; }
    uint64_t reads() const { return ids.size(); }
};

inline Trace load_trace_txt(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open trace: " + path);
    std::string magic;
    Trace tr;
    in >> magic >> tr.tokens >> tr.layers >> tr.experts;
    if (magic != "ps5lm-trace-txt/1" || !in) throw std::runtime_error("bad trace header: " + path);
    const size_t rows = size_t(tr.tokens) * tr.layers;
    tr.offs.reserve(rows + 1);
    tr.offs.push_back(0);
    for (size_t r = 0; r < rows; ++r) {
        uint32_t k = 0;
        if (!(in >> k)) throw std::runtime_error("truncated trace at row " + std::to_string(r));
        for (uint32_t i = 0; i < k; ++i) {
            uint32_t e = 0;
            if (!(in >> e) || e >= tr.experts) throw std::runtime_error("bad expert id at row " + std::to_string(r));
            tr.ids.push_back(uint16_t(e));
        }
        tr.offs.push_back(uint32_t(tr.ids.size()));
    }
    return tr;
}

}  // namespace ps5lm
