// chunk_streaming.cpp
// - Streaming preprocessor to disk (as before)
// - read_chunk function
// - NEW: ChunkCache (fixed-size FIFO) with get_chunks(ids)
//   * NumPy API always available
//   * Optional PyTorch API if compiled with -DCHUNKIO_WITH_TORCH
//
// Build (NumPy-only):
//   python setup.py build_ext --inplace
//
// Build (with PyTorch tensor API too):
//   Add -DCHUNKIO_WITH_TORCH and torch include/link flags in setup.py

#include <algorithm>
#include <cstdint>
#include <deque>
#include <fstream>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <mutex>

#include <omp.h>

// ---- pybind11 ----
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

/// increment
#include <cstring>      // memcpy
#include <unordered_map>
#include <cmath>

/// end increment



#ifdef CHUNKIO_WITH_TORCH
  #include <torch/extension.h>
#endif

namespace py = pybind11;

using std::int64_t;
using std::size_t;
using std::vector;

// =======================
// Streaming preprocessor
// =======================

struct ChunkResultLite {
    vector<vector<int64_t>> chunk_map;     // [num_nodes][max_chunk_per_node], -1 if none
    vector<vector<double>>  chunk_last_ts; // [num_nodes][max_chunk_per_node]
    int64_t chunk_size{0};
    int64_t total_chunks{0};
    std::string ts_path, eid_path, other_path, out_dir;
};

struct DiskEdge {
    int64_t node;  // owner node
    double  ts;
    int64_t eid;
    int64_t other;
};

struct ChunkDelta {
    int64_t cid;     // chunk id
    int64_t pos;     // position within chunk [0, chunk_size-1]
    double  ts;
    int64_t eid;
    int64_t other;
};

static inline size_t shard_id_for(int64_t node, size_t num_shards) {
    return static_cast<size_t>(node) % num_shards;
}

ChunkResultLite preprocess_streaming(
    const vector<int64_t>& src,
    const vector<int64_t>& dst,
    const vector<double>&  ts,
    const vector<int64_t>& eid,
    int64_t num_nodes,
    int chunk_size = 4,
    int max_chunk_per_node = 3,
    bool duplicate_undirected = true,
    const std::string& out_dir = "preproc_out",
    size_t num_shards = 256
) {
    std::filesystem::create_directories(out_dir);
    const std::string ts_path    = out_dir + "/ts.bin";
    const std::string eid_path   = out_dir + "/eid.bin";
    const std::string other_path = out_dir + "/other.bin";

    // Phase 0: open shard writers
    std::vector<std::ofstream> shard_files;
    shard_files.reserve(num_shards);
    for (size_t s = 0; s < num_shards; ++s) {
        std::string path = out_dir + "/shard_" + std::to_string(s) + ".bin";
        shard_files.emplace_back(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!shard_files.back()) throw std::runtime_error("Failed to open shard file: " + path);
    }

    // Phase 1: stream records to shards (append-only)
    const size_t M = src.size();
    for (size_t i = 0; i < M; ++i) {
        int64_t u = src[i], v = dst[i];
        double  t = ts[i];
        int64_t e = eid[i];

        {   // record for dst (owner = v, other = u)
            DiskEdge rec{v, t, e, u};
            auto sid = shard_id_for(v, num_shards);
            shard_files[sid].write(reinterpret_cast<const char*>(&rec), sizeof(DiskEdge));
        }
        if (duplicate_undirected && u != v) {
            DiskEdge rec{u, t, e, v};
            auto sid = shard_id_for(u, num_shards);
            shard_files[sid].write(reinterpret_cast<const char*>(&rec), sizeof(DiskEdge));
        }
    }
    for (auto& f : shard_files) f.close();

    // Phase 2: open final outputs
    std::ofstream ts_out   (ts_path,    std::ios::binary | std::ios::out | std::ios::trunc);
    std::ofstream eid_out  (eid_path,   std::ios::binary | std::ios::out | std::ios::trunc);
    std::ofstream other_out(other_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!ts_out || !eid_out || !other_out) throw std::runtime_error("Failed to open final chunk files.");

    const double INF = std::numeric_limits<double>::infinity();

    vector<vector<int64_t>> chunk_map(num_nodes, vector<int64_t>(max_chunk_per_node, -1));
    vector<vector<double>>  chunk_last_ts(num_nodes, vector<double>(max_chunk_per_node, INF));

    int64_t global_chunk_id = 0;

    // Phase 3: process shards
    for (size_t s = 0; s < num_shards; ++s) {
        std::string path = out_dir + "/shard_" + std::to_string(s) + ".bin";
        std::ifstream in(path, std::ios::binary | std::ios::in);
        if (!in) continue;

        in.seekg(0, std::ios::end);
        std::streampos sz = in.tellg();
        in.seekg(0, std::ios::beg);
        size_t nrec = static_cast<size_t>(sz / sizeof(DiskEdge));
        vector<DiskEdge> recs(nrec);
        if (nrec) in.read(reinterpret_cast<char*>(recs.data()), nrec * sizeof(DiskEdge));
        in.close();
        std::filesystem::remove(path);

        if (recs.empty()) continue;

        std::sort(recs.begin(), recs.end(), [](const DiskEdge& a, const DiskEdge& b){
            if (a.node != b.node) return a.node < b.node;
            return a.ts < b.ts;
        });

        size_t i = 0;
        while (i < recs.size()) {
            int64_t node = recs[i].node;
            size_t j = i;
            while (j < recs.size() && recs[j].node == node) ++j; // [i, j)

            size_t deg = j - i;
            for (size_t start = 0; start < deg; start += (size_t)chunk_size) {
                size_t end = std::min(start + (size_t)chunk_size, deg);
                size_t len = end - start;

                // fixed-size chunks (pad)
                vector<double>  t_chunk(chunk_size, INF);
                vector<int64_t> e_chunk(chunk_size, -1);
                vector<int64_t> o_chunk(chunk_size, -1);

                for (size_t k = 0; k < len; ++k) {
                    const auto& r = recs[i + start + k];
                    t_chunk[k] = r.ts;
                    e_chunk[k] = r.eid;
                    o_chunk[k] = r.other;
                }

                ts_out.write(reinterpret_cast<const char*>(t_chunk.data()), sizeof(double)  * chunk_size);
                eid_out.write(reinterpret_cast<const char*>(e_chunk.data()), sizeof(int64_t) * chunk_size);
                other_out.write(reinterpret_cast<const char*>(o_chunk.data()), sizeof(int64_t) * chunk_size);

                // per-node metadata: first max_chunk_per_node chunks
                int64_t local_idx = (int64_t)(start / (size_t)chunk_size);
                if (local_idx < (int64_t)chunk_last_ts[node].size()) {
                    chunk_map[node][local_idx] = global_chunk_id;
                    chunk_last_ts[node][local_idx] = t_chunk[ (len == (size_t)chunk_size) ? (chunk_size - 1) : (len - 1) ];
                }

                ++global_chunk_id;
            }

            i = j;
        }
    }

    ts_out.close();
    eid_out.close();
    other_out.close();

    return { std::move(chunk_map), std::move(chunk_last_ts),
             (int64_t)chunk_size, global_chunk_id,
             ts_path, eid_path, other_path, out_dir };
}

// ============
// read_chunk()
// ============

void read_chunk(const std::string& ts_path,
                const std::string& eid_path,
                const std::string& other_path,
                int64_t chunk_size,
                int64_t k,
                vector<double>&  t_chunk,
                vector<int64_t>& e_chunk,
                vector<int64_t>& o_chunk)
{
    if (k < 0) throw std::invalid_argument("k must be non-negative");

    std::ifstream ts_in(ts_path, std::ios::binary);
    std::ifstream eid_in(eid_path, std::ios::binary);
    std::ifstream oth_in(other_path, std::ios::binary);
    if (!ts_in || !eid_in || !oth_in) throw std::runtime_error("Failed to open chunk files.");

    const int64_t offD = k * chunk_size * (int64_t)sizeof(double);
    const int64_t offI = k * chunk_size * (int64_t)sizeof(int64_t);

    ts_in.seekg(offD);
    eid_in.seekg(offI);
    oth_in.seekg(offI);

    t_chunk.resize((size_t)chunk_size);
    e_chunk.resize((size_t)chunk_size);
    o_chunk.resize((size_t)chunk_size);

    ts_in.read(reinterpret_cast<char*>(t_chunk.data()), sizeof(double)  * (size_t)chunk_size);
    eid_in.read(reinterpret_cast<char*>(e_chunk.data()), sizeof(int64_t) * (size_t)chunk_size);
    oth_in.read(reinterpret_cast<char*>(o_chunk.data()), sizeof(int64_t) * (size_t)chunk_size);

    if (!ts_in || !eid_in || !oth_in) {
        throw std::runtime_error("read_chunk: I/O error or k out of range");
    }
}

// ==========================
// Fixed-size FIFO cache
// ==========================

class ChunkCache {
public:
    ChunkCache(const std::string& out_dir, int64_t chunk_size, size_t capacity = 100)
    : out_dir_(out_dir), chunk_size_(chunk_size), capacity_(capacity)
    {
        if (capacity_ == 0) throw std::invalid_argument("capacity must be > 0");
        ts_path_    = out_dir_ + "/ts.bin";
        eid_path_   = out_dir_ + "/eid.bin";
        other_path_ = out_dir_ + "/other.bin";

        // open persistent file streams
        ts_in_.open(ts_path_, std::ios::binary);
        eid_in_.open(eid_path_, std::ios::binary);
        oth_in_.open(other_path_, std::ios::binary);
        if (!ts_in_ || !eid_in_ || !oth_in_) {
            throw std::runtime_error("ChunkCache: failed to open chunk files in " + out_dir_);
        }

        // allocate in-memory cache storage [capacity x chunk_size]
        ts_buf_.resize(capacity_ * (size_t)chunk_size_);
        eid_buf_.resize(capacity_ * (size_t)chunk_size_);
        oth_buf_.resize(capacity_ * (size_t)chunk_size_);
        slot_used_.assign(capacity_, false);
        next_free_slot_ = 0;
    }

    ~ChunkCache() {
        // close streams
        ts_in_.close();
        eid_in_.close();
        oth_in_.close();
    }

    // NumPy API: ids is a 1-D int64 array
    py::tuple get_chunks_numpy(py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids) {
        py::gil_scoped_release release; // don't block Python threads during IO

        auto buf = ids.unchecked<1>();
        const ssize_t N = buf.shape(0);

        // allocate outputs (N, chunk_size)
        py::array_t<double>  ts_out  ({N, (ssize_t)chunk_size_});
        py::array_t<int64_t> eid_out ({N, (ssize_t)chunk_size_});
        py::array_t<int64_t> oth_out ({N, (ssize_t)chunk_size_});

        double*  ts_ptr  = ts_out.mutable_data();
        int64_t* eid_ptr = eid_out.mutable_data();
        int64_t* oth_ptr = oth_out.mutable_data();

        // For each id, ensure cached and then copy from slot into output row
        for (ssize_t i = 0; i < N; ++i) {
            int64_t cid = buf(i);
            size_t slot = ensure_in_cache_(cid); // loads if absent (with FIFO eviction)

            const size_t offset = slot * (size_t)chunk_size_;
            // copy into row i
            std::memcpy(ts_ptr  + i * (size_t)chunk_size_, &ts_buf_[offset],  sizeof(double)  * (size_t)chunk_size_);
            std::memcpy(eid_ptr + i * (size_t)chunk_size_, &eid_buf_[offset], sizeof(int64_t) * (size_t)chunk_size_);
            std::memcpy(oth_ptr + i * (size_t)chunk_size_, &oth_buf_[offset], sizeof(int64_t) * (size_t)chunk_size_);
        }

        py::gil_scoped_acquire acquire; // reacquire GIL before returning
        return py::make_tuple(ts_out, eid_out, oth_out);
    }


    // Apply deltas: only updates chunks that are already cached.
    // Inputs are 1-D arrays of equal length:
    //   cids[i], pos[i], ts[i], eid[i], other[i]
    void apply_deltas_numpy(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> cids,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> pos,
        py::array_t<double,  py::array::c_style | py::array::forcecast> ts,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> eid,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> oth
    ) {
        // Validate shapes
        auto c = cids.unchecked<1>();
        auto p = pos.unchecked<1>();
        auto t = ts.unchecked<1>();
        auto e = eid.unchecked<1>();
        auto o = oth.unchecked<1>();

        const ssize_t N = c.shape(0);
        if (p.shape(0) != N || t.shape(0) != N || e.shape(0) != N || o.shape(0) != N) {
            throw std::invalid_argument("apply_deltas: all inputs must have same length");
        }

        // Lock cache metadata + buffers
        std::lock_guard<std::mutex> lock(mu_);

        for (ssize_t i = 0; i < N; ++i) {
            const int64_t cid = c(i);
            const int64_t at  = p(i);

            if (at < 0 || at >= chunk_size_) {
                throw std::invalid_argument("apply_deltas: pos out of range");
            }

            auto it = id2slot_.find(cid);
            if (it == id2slot_.end()) {
                // Not cached => do nothing (per your design)
                continue;
            }

            const size_t slot = it->second;
            const size_t base = slot * (size_t)chunk_size_ + (size_t)at;

            ts_buf_[base]  = t(i);
            eid_buf_[base] = e(i);
            oth_buf_[base] = o(i);
        }
    }

#ifdef CHUNKIO_WITH_TORCH
    // Apply deltas using Torch CPU tensors.
    // cids, pos, eid, oth: int64 CPU 1-D
    // ts: double CPU 1-D
    void apply_deltas_torch(
        const at::Tensor& cids,
        const at::Tensor& pos,
        const at::Tensor& ts,
        const at::Tensor& eid,
        const at::Tensor& oth
    ) {
        TORCH_CHECK(cids.device().is_cpu(), "cids must be on CPU");
        TORCH_CHECK(pos.device().is_cpu(),  "pos must be on CPU");
        TORCH_CHECK(ts.device().is_cpu(),   "ts must be on CPU");
        TORCH_CHECK(eid.device().is_cpu(),  "eid must be on CPU");
        TORCH_CHECK(oth.device().is_cpu(),  "other must be on CPU");

        TORCH_CHECK(cids.scalar_type() == at::kLong, "cids must be torch.int64");
        TORCH_CHECK(pos.scalar_type()  == at::kLong, "pos must be torch.int64");
        TORCH_CHECK(ts.scalar_type()   == at::kDouble, "ts must be torch.float64");
        TORCH_CHECK(eid.scalar_type()  == at::kLong, "eid must be torch.int64");
        TORCH_CHECK(oth.scalar_type()  == at::kLong, "other must be torch.int64");

        TORCH_CHECK(cids.dim() == 1 && pos.dim() == 1 && ts.dim() == 1 && eid.dim() == 1 && oth.dim() == 1,
                    "apply_deltas_torch: all inputs must be 1-D");
        TORCH_CHECK(cids.numel() == pos.numel() &&
                    cids.numel() == ts.numel() &&
                    cids.numel() == eid.numel() &&
                    cids.numel() == oth.numel(),
                    "apply_deltas_torch: all inputs must have same length");

        auto cids_c = cids.contiguous();
        auto pos_c  = pos.contiguous();
        auto ts_c   = ts.contiguous();
        auto eid_c  = eid.contiguous();
        auto oth_c  = oth.contiguous();

        const int64_t N = cids_c.numel();

        const int64_t* cids_ptr = cids_c.data_ptr<int64_t>();
        const int64_t* pos_ptr  = pos_c.data_ptr<int64_t>();
        const double*  ts_ptr   = ts_c.data_ptr<double>();
        const int64_t* eid_ptr  = eid_c.data_ptr<int64_t>();
        const int64_t* oth_ptr  = oth_c.data_ptr<int64_t>();

        std::lock_guard<std::mutex> lock(mu_);

        for (int64_t i = 0; i < N; ++i) {
            const int64_t cid = cids_ptr[i];
            const int64_t at  = pos_ptr[i];

            if (at < 0 || at >= chunk_size_) {
                throw std::invalid_argument("apply_deltas_torch: pos out of range");
            }

            auto it = id2slot_.find(cid);
            if (it == id2slot_.end()) {
                // Not cached => do nothing (same policy as numpy)
                continue;
            }

            const size_t slot = it->second;
            const size_t idx  = slot * (size_t)chunk_size_ + (size_t)at;

            ts_buf_[idx]  = ts_ptr[i];
            eid_buf_[idx] = eid_ptr[i];
            oth_buf_[idx] = oth_ptr[i];
        }
    }
#endif

#ifdef CHUNKIO_WITH_TORCH
    // PyTorch API: ids is torch.LongTensor on CPU, 1-D and contiguous
    std::tuple<at::Tensor, at::Tensor, at::Tensor> get_chunks_torch(const at::Tensor& ids) {
        TORCH_CHECK(ids.device().is_cpu(), "ids must be on CPU");
        TORCH_CHECK(ids.scalar_type() == at::kLong, "ids must be torch.int64 (Long)");
        TORCH_CHECK(ids.dim() == 1, "ids must be 1-D");
        auto ids_c = ids.contiguous();
        const int64_t N = ids_c.size(0);

        // allocate outputs
        auto ts_out  = at::empty({N, chunk_size_}, at::dtype(at::kDouble).device(at::kCPU));
        auto eid_out = at::empty({N, chunk_size_}, at::dtype(at::kLong).device(at::kCPU));
        auto oth_out = at::empty({N, chunk_size_}, at::dtype(at::kLong).device(at::kCPU));

        const int64_t* ids_ptr = ids_c.data_ptr<int64_t>();
        double*  ts_ptr  = ts_out.data_ptr<double>();
        int64_t* eid_ptr = eid_out.data_ptr<int64_t>();
        int64_t* oth_ptr = oth_out.data_ptr<int64_t>();

        // release GIL during IO
        py::gil_scoped_release release;

        for (int64_t i = 0; i < N; ++i) {
            int64_t cid = ids_ptr[i];
            size_t slot = ensure_in_cache_(cid);
            const size_t offset = slot * (size_t)chunk_size_;
            std::memcpy(ts_ptr  + i * (size_t)chunk_size_, &ts_buf_[offset],  sizeof(double)  * (size_t)chunk_size_);
            std::memcpy(eid_ptr + i * (size_t)chunk_size_, &eid_buf_[offset], sizeof(int64_t) * (size_t)chunk_size_);
            std::memcpy(oth_ptr + i * (size_t)chunk_size_, &oth_buf_[offset], sizeof(int64_t) * (size_t)chunk_size_);
        }

        return {ts_out, eid_out, oth_out};
    }
#endif

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        id2slot_.clear();
        fifo_order_.clear();
        std::fill(slot_used_.begin(), slot_used_.end(), false);
        next_free_slot_ = 0;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return id2slot_.size();
    }

    size_t capacity() const { return capacity_; }
    int64_t chunk_size() const { return chunk_size_; }

private:
    // Ensure chunk 'cid' is loaded and return its slot index.
    size_t ensure_in_cache_(int64_t cid) {
        std::lock_guard<std::mutex> lock(mu_);

        auto it = id2slot_.find(cid);
        if (it != id2slot_.end()) {
            // Already cached (FIFO policy doesn't update position on hit)
            return it->second;
        }

        // Need a slot: either a free slot or evict oldest
        size_t slot = 0;
        if (next_free_slot_ < capacity_) {
            slot = next_free_slot_++;
        } else {
            // evict FIFO head
            if (fifo_order_.empty()) {
                throw std::runtime_error("ChunkCache: internal FIFO empty on eviction");
            }
            int64_t victim = fifo_order_.front();
            fifo_order_.pop_front();
            auto itv = id2slot_.find(victim);
            if (itv == id2slot_.end()) {
                throw std::runtime_error("ChunkCache: victim not found in map");
            }
            slot = itv->second;
            id2slot_.erase(itv);
        }

        // Load chunk 'cid' into 'slot'
        load_chunk_into_slot_(cid, slot);

        // Register
        id2slot_[cid] = slot;
        fifo_order_.push_back(cid);
        slot_used_[slot] = true;
        return slot;
    }

    void load_chunk_into_slot_(int64_t cid, size_t slot) {
        // compute offsets
        const int64_t offD = cid * chunk_size_ * (int64_t)sizeof(double);
        const int64_t offI = cid * chunk_size_ * (int64_t)sizeof(int64_t);

        // seek and read in each file (streams are persistent)
        ts_in_.clear();  ts_in_.seekg(offD);
        eid_in_.clear(); eid_in_.seekg(offI);
        oth_in_.clear(); oth_in_.seekg(offI);

        if (!ts_in_ || !eid_in_ || !oth_in_) {
            throw std::runtime_error("ChunkCache: seek failed for chunk id " + std::to_string(cid));
        }

        const size_t dst = slot * (size_t)chunk_size_;
        ts_in_.read(reinterpret_cast<char*>(&ts_buf_[dst]),  sizeof(double)  * (size_t)chunk_size_);
        eid_in_.read(reinterpret_cast<char*>(&eid_buf_[dst]), sizeof(int64_t) * (size_t)chunk_size_);
        oth_in_.read(reinterpret_cast<char*>(&oth_buf_[dst]), sizeof(int64_t) * (size_t)chunk_size_);

        if (!ts_in_ || !eid_in_ || !oth_in_) {
            throw std::runtime_error("ChunkCache: read failed for chunk id " + std::to_string(cid));
        }
    }

private:
    // file paths & streams
    std::string out_dir_;
    std::string ts_path_, eid_path_, other_path_;
    std::ifstream ts_in_, eid_in_, oth_in_;

    // cache geometry
    int64_t chunk_size_;
    size_t  capacity_;

    // storage
    vector<double>  ts_buf_;
    vector<int64_t> eid_buf_;
    vector<int64_t> oth_buf_;

    // metadata
    std::unordered_map<int64_t, size_t> id2slot_;
    std::deque<int64_t> fifo_order_;
    vector<bool> slot_used_;
    size_t next_free_slot_{0};

    // lock for thread-safety
    mutable std::mutex mu_;
};




//==== increment
static inline int64_t num_chunks_in_row(const std::vector<int64_t>& row) {
    int64_t c = 0;
    for (int64_t i = 0; i < (int64_t)row.size(); ++i) {
        if (row[(size_t)i] == -1) break;
        ++c;
    }
    return c;
}

// Read tail chunk once and count filled entries by scanning eid != -1.
static inline int64_t tail_fill_from_disk(
    const std::string& ts_path,
    const std::string& eid_path,
    const std::string& other_path,
    int64_t chunk_size,
    int64_t tail_cid
) {
    std::vector<double>  t;
    std::vector<int64_t> e, o;
    read_chunk(ts_path, eid_path, other_path, chunk_size, tail_cid, t, e, o);

    int64_t fill = 0;
    for (int64_t i = 0; i < chunk_size; ++i) {
        if (e[(size_t)i] == -1) break;
        ++fill;
    }
    return fill;
}

static inline void write_entry_inplace(
    std::fstream& ts_io,
    std::fstream& eid_io,
    std::fstream& oth_io,
    int64_t chunk_size,
    int64_t cid,
    int64_t pos,
    double  ts_val,
    int64_t eid_val,
    int64_t oth_val
) {
    const int64_t offD = (cid * chunk_size + pos) * (int64_t)sizeof(double);
    const int64_t offI = (cid * chunk_size + pos) * (int64_t)sizeof(int64_t);

    ts_io.seekp(offD);
    eid_io.seekp(offI);
    oth_io.seekp(offI);

    ts_io.write(reinterpret_cast<const char*>(&ts_val),  sizeof(double));
    eid_io.write(reinterpret_cast<const char*>(&eid_val), sizeof(int64_t));
    oth_io.write(reinterpret_cast<const char*>(&oth_val), sizeof(int64_t));

    if (!ts_io || !eid_io || !oth_io) {
        throw std::runtime_error("extend_latestk: inplace write failed");
    }
}

// Overwrite a whole chunk cid with a padded chunk whose [0] is the event.
static inline void overwrite_chunk_with_first_event(
    std::fstream& ts_io,
    std::fstream& eid_io,
    std::fstream& oth_io,
    int64_t chunk_size,
    int64_t cid,
    double  ts0,
    int64_t eid0,
    int64_t oth0
) {
    const double INF = std::numeric_limits<double>::infinity();

    std::vector<double>  t((size_t)chunk_size, INF);
    std::vector<int64_t> e((size_t)chunk_size, -1);
    std::vector<int64_t> o((size_t)chunk_size, -1);

    t[0] = ts0; e[0] = eid0; o[0] = oth0;

    const int64_t baseD = (cid * chunk_size) * (int64_t)sizeof(double);
    const int64_t baseI = (cid * chunk_size) * (int64_t)sizeof(int64_t);

    ts_io.seekp(baseD);
    eid_io.seekp(baseI);
    oth_io.seekp(baseI);

    ts_io.write(reinterpret_cast<const char*>(t.data()), sizeof(double)  * (size_t)chunk_size);
    eid_io.write(reinterpret_cast<const char*>(e.data()), sizeof(int64_t) * (size_t)chunk_size);
    oth_io.write(reinterpret_cast<const char*>(o.data()), sizeof(int64_t) * (size_t)chunk_size);

    if (!ts_io || !eid_io || !oth_io) {
        throw std::runtime_error("extend_latestk: overwrite chunk failed");
    }
}

// Append a new padded chunk at end, return its cid (must be prev.total_chunks).
static inline void append_new_chunk_with_first_event(
    std::fstream& ts_io,
    std::fstream& eid_io,
    std::fstream& oth_io,
    int64_t chunk_size,
    int64_t cid,
    double  ts0,
    int64_t eid0,
    int64_t oth0
) {
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double>  t((size_t)chunk_size, INF);
    std::vector<int64_t> e((size_t)chunk_size, -1);
    std::vector<int64_t> o((size_t)chunk_size, -1);
    t[0] = ts0; e[0] = eid0; o[0] = oth0;

    ts_io.seekp(0, std::ios::end);
    eid_io.seekp(0, std::ios::end);
    oth_io.seekp(0, std::ios::end);

    ts_io.write(reinterpret_cast<const char*>(t.data()), sizeof(double)  * (size_t)chunk_size);
    eid_io.write(reinterpret_cast<const char*>(e.data()), sizeof(int64_t) * (size_t)chunk_size);
    oth_io.write(reinterpret_cast<const char*>(o.data()), sizeof(int64_t) * (size_t)chunk_size);

    if (!ts_io || !eid_io || !oth_io) {
        throw std::runtime_error("extend_latestk: append new chunk failed");
    }
}

//==== end increment
//== increment 2
static ChunkResultLite extend_impl_with_deltas(
    const ChunkResultLite& prev,
    const std::vector<int64_t>& src_new,
    const std::vector<int64_t>& dst_new,
    const std::vector<double>&  ts_new,
    const std::vector<int64_t>& eid_new,
    std::vector<ChunkDelta>& deltas,
    bool duplicate_undirected
) {

    if (src_new.size() != dst_new.size() || src_new.size() != ts_new.size() || src_new.size() != eid_new.size())
        throw std::invalid_argument("extend_latestk: input arrays must have same length");

    ChunkResultLite out = prev;  // copy metadata
    const int64_t chunk_size = out.chunk_size;
    const int64_t K = (int64_t)out.chunk_map[0].size();  // max_chunk_per_node

    const std::string ts_path    = out.out_dir + "/ts.bin";
    const std::string eid_path   = out.out_dir + "/eid.bin";
    const std::string other_path = out.out_dir + "/other.bin";

    std::fstream ts_io(ts_path,     std::ios::in | std::ios::out | std::ios::binary);
    std::fstream eid_io(eid_path,   std::ios::in | std::ios::out | std::ios::binary);
    std::fstream oth_io(other_path, std::ios::in | std::ios::out | std::ios::binary);
    if (!ts_io || !eid_io || !oth_io) throw std::runtime_error("extend_latestk: failed to open bin files");

    // std::vector<ChunkDelta> deltas;
    // deltas.reserve(src_new.size() * 2);

    struct Rec { double ts; int64_t eid; int64_t other; };
    std::unordered_map<int64_t, std::vector<Rec>> by_node;
    by_node.reserve(src_new.size() * 2);

    for (size_t i = 0; i < src_new.size(); ++i) {
        int64_t u = src_new[i], v = dst_new[i];
        double  t = ts_new[i];
        int64_t e = eid_new[i];

        by_node[v].push_back(Rec{t, e, u});
        if (duplicate_undirected && u != v) {
            by_node[u].push_back(Rec{t, e, v});
        }
    }



    // Drop-in replacement: per-node processing inside extend_streaming_latestk_ordered_reuse()
    // Key fix: ALWAYS keep filling the current tail chunk (tail_fill < chunk_size) before creating/evicting a new one.
    // Also: after appending/reusing a chunk, we keep tail_cid + tail_fill updated so subsequent records pack correctly.

    for (auto& kv : by_node) {
        const int64_t node = kv.first;
        auto& recs = kv.second;
        std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b){ return a.ts < b.ts; });

        auto& row_map  = out.chunk_map.at((size_t)node);
        auto& row_last = out.chunk_last_ts.at((size_t)node);

        int64_t count = num_chunks_in_row(row_map);  // current number of chunks (<=K)

        // Track tail chunk id + how many entries are already filled in it.
        int64_t tail_cid  = -1;
        int64_t tail_fill = 0;

        if (count > 0) {
            tail_cid  = row_map[(size_t)(count - 1)];
            tail_fill = tail_fill_from_disk(ts_path, eid_path, other_path, chunk_size, tail_cid);

            // monotonic guard: new timestamps must be >= last existing ts in tail (if any)
            if (tail_fill > 0) {
                const double last_ts = row_last[(size_t)(count - 1)];
                if (!recs.empty() && recs.front().ts < last_ts) {
                    throw std::runtime_error("extend_latestk: non-monotonic timestamps for node " + std::to_string(node));
                }
            }
        }

        // Process all new records for this node, packing into the tail chunk when possible.
        for (size_t idx = 0; idx < recs.size(); /* increment inside */) {
            const auto& r = recs[idx];

            // Case A: No chunks yet -> create first chunk and write r at pos 0.
            if (count == 0) {
                const int64_t new_cid = out.total_chunks;
                append_new_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, new_cid, r.ts, r.eid, r.other);
                out.total_chunks++;

                row_map[0]  = new_cid;
                row_last[0] = r.ts;

                count     = 1;
                tail_cid  = new_cid;
                tail_fill = 1;

                ++idx;
                continue;
            }

            // Case B: Tail chunk exists and has free space -> write into it in-place.
            if (tail_fill < chunk_size) {
                deltas.push_back(ChunkDelta{tail_cid, tail_fill, r.ts, r.eid, r.other});

                write_entry_inplace(ts_io, eid_io, oth_io, chunk_size, tail_cid, tail_fill, r.ts, r.eid, r.other);
                tail_fill++;
                row_last[(size_t)(count - 1)] = r.ts;

                ++idx;
                continue;
            }

            // From here: tail is full, so we must get a "fresh" tail chunk (append or evict+reuse),
            // then write current record as the first entry in that new tail chunk.

            // Case C: We still have capacity (<K chunks) -> append a new chunk and make it the tail.
            if (count < K) {
                const int64_t new_cid = out.total_chunks;
                append_new_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, new_cid, r.ts, r.eid, r.other);
                out.total_chunks++;

                row_map[(size_t)count]  = new_cid;
                row_last[(size_t)count] = r.ts;

                count++;
                tail_cid  = new_cid;
                tail_fill = 1;

                ++idx;
                continue;
            }

            // Case D: count == K -> evict oldest (index 0), shift left, reuse evicted cid as new tail.
            {
                const int64_t evict_cid = row_map[0];

                for (int64_t i = 0; i < K - 1; ++i) {
                    row_map[(size_t)i]  = row_map[(size_t)(i + 1)];
                    row_last[(size_t)i] = row_last[(size_t)(i + 1)];
                }

                overwrite_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, evict_cid, r.ts, r.eid, r.other);

                row_map[(size_t)(K - 1)]  = evict_cid;
                row_last[(size_t)(K - 1)] = r.ts;

                tail_cid  = evict_cid;
                tail_fill = 1;

                ++idx;
                continue;
            }
        }
    }


    ts_io.flush(); eid_io.flush(); oth_io.flush();
    return out;

}





ChunkResultLite extend_streaming_latestk_ordered_reuse(
    const ChunkResultLite& prev,
    const std::vector<int64_t>& src_new,
    const std::vector<int64_t>& dst_new,
    const std::vector<double>&  ts_new,
    const std::vector<int64_t>& eid_new,
    bool duplicate_undirected = true
) {
    if (src_new.size() != dst_new.size() || src_new.size() != ts_new.size() || src_new.size() != eid_new.size())
        throw std::invalid_argument("extend_latestk: input arrays must have same length");

    ChunkResultLite out = prev;  // copy metadata
    const int64_t chunk_size = out.chunk_size;
    const int64_t K = (int64_t)out.chunk_map[0].size();  // max_chunk_per_node

    const std::string ts_path    = out.out_dir + "/ts.bin";
    const std::string eid_path   = out.out_dir + "/eid.bin";
    const std::string other_path = out.out_dir + "/other.bin";

    std::fstream ts_io(ts_path,     std::ios::in | std::ios::out | std::ios::binary);
    std::fstream eid_io(eid_path,   std::ios::in | std::ios::out | std::ios::binary);
    std::fstream oth_io(other_path, std::ios::in | std::ios::out | std::ios::binary);
    if (!ts_io || !eid_io || !oth_io) throw std::runtime_error("extend_latestk: failed to open bin files");

    std::vector<ChunkDelta> deltas;
    deltas.reserve(src_new.size() * 2);

    struct Rec { double ts; int64_t eid; int64_t other; };
    std::unordered_map<int64_t, std::vector<Rec>> by_node;
    by_node.reserve(src_new.size() * 2);

    for (size_t i = 0; i < src_new.size(); ++i) {
        int64_t u = src_new[i], v = dst_new[i];
        double  t = ts_new[i];
        int64_t e = eid_new[i];

        by_node[v].push_back(Rec{t, e, u});
        if (duplicate_undirected && u != v) {
            by_node[u].push_back(Rec{t, e, v});
        }
    }



    // Drop-in replacement: per-node processing inside extend_streaming_latestk_ordered_reuse()
    // Key fix: ALWAYS keep filling the current tail chunk (tail_fill < chunk_size) before creating/evicting a new one.
    // Also: after appending/reusing a chunk, we keep tail_cid + tail_fill updated so subsequent records pack correctly.

    for (auto& kv : by_node) {
        const int64_t node = kv.first;
        auto& recs = kv.second;
        std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b){ return a.ts < b.ts; });

        auto& row_map  = out.chunk_map.at((size_t)node);
        auto& row_last = out.chunk_last_ts.at((size_t)node);

        int64_t count = num_chunks_in_row(row_map);  // current number of chunks (<=K)

        // Track tail chunk id + how many entries are already filled in it.
        int64_t tail_cid  = -1;
        int64_t tail_fill = 0;

        if (count > 0) {
            tail_cid  = row_map[(size_t)(count - 1)];
            tail_fill = tail_fill_from_disk(ts_path, eid_path, other_path, chunk_size, tail_cid);

            // monotonic guard: new timestamps must be >= last existing ts in tail (if any)
            if (tail_fill > 0) {
                const double last_ts = row_last[(size_t)(count - 1)];
                if (!recs.empty() && recs.front().ts < last_ts) {
                    throw std::runtime_error("extend_latestk: non-monotonic timestamps for node " + std::to_string(node));
                }
            }
        }

        // Process all new records for this node, packing into the tail chunk when possible.
        for (size_t idx = 0; idx < recs.size(); /* increment inside */) {
            const auto& r = recs[idx];

            // Case A: No chunks yet -> create first chunk and write r at pos 0.
            if (count == 0) {
                const int64_t new_cid = out.total_chunks;
                append_new_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, new_cid, r.ts, r.eid, r.other);
                out.total_chunks++;

                row_map[0]  = new_cid;
                row_last[0] = r.ts;

                count     = 1;
                tail_cid  = new_cid;
                tail_fill = 1;

                ++idx;
                continue;
            }

            // Case B: Tail chunk exists and has free space -> write into it in-place.
            if (tail_fill < chunk_size) {
                deltas.push_back(ChunkDelta{tail_cid, tail_fill, r.ts, r.eid, r.other});

                write_entry_inplace(ts_io, eid_io, oth_io, chunk_size, tail_cid, tail_fill, r.ts, r.eid, r.other);
                tail_fill++;
                row_last[(size_t)(count - 1)] = r.ts;

                ++idx;
                continue;
            }

            // From here: tail is full, so we must get a "fresh" tail chunk (append or evict+reuse),
            // then write current record as the first entry in that new tail chunk.

            // Case C: We still have capacity (<K chunks) -> append a new chunk and make it the tail.
            if (count < K) {
                const int64_t new_cid = out.total_chunks;
                append_new_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, new_cid, r.ts, r.eid, r.other);
                out.total_chunks++;

                row_map[(size_t)count]  = new_cid;
                row_last[(size_t)count] = r.ts;

                count++;
                tail_cid  = new_cid;
                tail_fill = 1;

                ++idx;
                continue;
            }

            // Case D: count == K -> evict oldest (index 0), shift left, reuse evicted cid as new tail.
            {
                const int64_t evict_cid = row_map[0];

                for (int64_t i = 0; i < K - 1; ++i) {
                    row_map[(size_t)i]  = row_map[(size_t)(i + 1)];
                    row_last[(size_t)i] = row_last[(size_t)(i + 1)];
                }

                overwrite_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, evict_cid, r.ts, r.eid, r.other);

                row_map[(size_t)(K - 1)]  = evict_cid;
                row_last[(size_t)(K - 1)] = r.ts;

                tail_cid  = evict_cid;
                tail_fill = 1;

                ++idx;
                continue;
            }
        }
    }

































    // // Process each node independently
    // for (auto& kv : by_node) {
    //     const int64_t node = kv.first;
    //     auto& recs = kv.second;
    //     std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b){ return a.ts < b.ts; });

    //     auto& row_map  = out.chunk_map.at((size_t)node);
    //     auto& row_last = out.chunk_last_ts.at((size_t)node);

    //     int64_t count = num_chunks_in_row(row_map);  // current number of chunks (<=K)
    //     int64_t tail_fill = 0;

    //     if (count > 0) {
    //         const int64_t tail_cid = row_map[(size_t)(count - 1)];
    //         tail_fill = tail_fill_from_disk(ts_path, eid_path, other_path, chunk_size, tail_cid);

    //         // monotonic guard: new timestamps must be >= last existing ts in tail (if any)
    //         if (tail_fill > 0) {
    //             // We can use metadata row_last[count-1] as "last ts" for the tail chunk if kept updated.
    //             // But to be safe, we rely on row_last which your preprocess sets and our extend updates.
    //             const double last_ts = row_last[(size_t)(count - 1)];
    //             if (!recs.empty() && recs.front().ts < last_ts) {
    //                 throw std::runtime_error("extend_latestk: non-monotonic timestamps for node " + std::to_string(node));
    //             }
    //         }
    //     }

    //     size_t idx = 0;

    //     // 1) Fill partially-filled tail chunk first
    //     if (count > 0 && tail_fill < chunk_size) {
    //         const int64_t tail_cid = row_map[(size_t)(count - 1)];
    //         while (idx < recs.size() && tail_fill < chunk_size) {
    //             const auto& r = recs[idx++];
    //             write_entry_inplace(ts_io, eid_io, oth_io, chunk_size, tail_cid, tail_fill, r.ts, r.eid, r.other);
    //             tail_fill++;
    //             row_last[(size_t)(count - 1)] = r.ts;
    //         }
    //     }

    //     // 2) Remaining records: may create new chunks; if full K, evict oldest by shifting and reuse its cid
    //     while (idx < recs.size()) {
    //         const auto& r = recs[idx++];

    //         // If tail is full OR node has no chunks, we need a fresh chunk to place this event at pos 0.
    //         // (At this point tail is guaranteed full, because we filled it above.)
    //         if (count == 0) {
    //             // allocate first chunk physically at end
    //             const int64_t new_cid = out.total_chunks;
    //             append_new_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, new_cid, r.ts, r.eid, r.other);
    //             out.total_chunks++;

    //             row_map[0]  = new_cid;
    //             row_last[0] = r.ts;
    //             count = 1;
    //             tail_fill = 1;
    //             continue;
    //         }

    //         if (count < K) {
    //             // allocate a new chunk id at end and append it
    //             const int64_t new_cid = out.total_chunks;
    //             append_new_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, new_cid, r.ts, r.eid, r.other);
    //             out.total_chunks++;

    //             row_map[(size_t)count]  = new_cid;
    //             row_last[(size_t)count] = r.ts;
    //             count++;
    //             tail_fill = 1;
    //             continue;
    //         }

    //         // count == K: evict oldest (index 0), shift left, reuse evicted cid as new tail chunk
    //         const int64_t evict_cid = row_map[0];

    //         for (int64_t i = 0; i < K - 1; ++i) {
    //             row_map[(size_t)i]  = row_map[(size_t)(i + 1)];
    //             row_last[(size_t)i] = row_last[(size_t)(i + 1)];
    //         }

    //         // reuse evicted chunk id: overwrite it with empty padded chunk containing this event at pos 0
    //         overwrite_chunk_with_first_event(ts_io, eid_io, oth_io, chunk_size, evict_cid, r.ts, r.eid, r.other);

    //         row_map[(size_t)(K - 1)]  = evict_cid;
    //         row_last[(size_t)(K - 1)] = r.ts;
    //         tail_fill = 1;
    //     }
    // }

    ts_io.flush(); eid_io.flush(); oth_io.flush();
    return out;
}




ChunkResultLite make_tci(
    std::vector<std::vector<int64_t>> chunk_map,
    std::vector<std::vector<double>>  chunk_last_ts,
    int64_t chunk_size,
    int64_t total_chunks,
    const std::string& out_dir
) {
    const std::string ts_path    = out_dir + "/ts.bin";
    const std::string eid_path   = out_dir + "/eid.bin";
    const std::string other_path = out_dir + "/other.bin";
    return { std::move(chunk_map), std::move(chunk_last_ts),
             chunk_size, total_chunks,
             ts_path, eid_path, other_path, out_dir };
}


//== end 2 increment



// ======================
// Python bindings
// ======================

PYBIND11_MODULE(chunkio, m) {
    m.doc() = "Streaming chunk preprocessor + on-disk IO + fixed-size FIFO cache";

    // --- Results DTO
    py::class_<ChunkResultLite>(m, "ChunkResultLite")
        .def_readonly("chunk_map",     &ChunkResultLite::chunk_map)
        .def_readonly("chunk_last_ts", &ChunkResultLite::chunk_last_ts)
        .def_readonly("chunk_size",    &ChunkResultLite::chunk_size)
        .def_readonly("total_chunks",  &ChunkResultLite::total_chunks)
        .def_readonly("ts_path",       &ChunkResultLite::ts_path)
        .def_readonly("eid_path",      &ChunkResultLite::eid_path)
        .def_readonly("other_path",    &ChunkResultLite::other_path)
        .def_readonly("out_dir",       &ChunkResultLite::out_dir);

    // --- Preprocess
    m.def("preprocess_streaming", &preprocess_streaming,
          py::arg("src"), py::arg("dst"), py::arg("ts"), py::arg("eid"),
          py::arg("num_nodes"),
          py::arg("chunk_size") = 4,
          py::arg("max_chunk_per_node") = 3,
          py::arg("duplicate_undirected") = true,
          py::arg("out_dir") = "preproc_out",
          py::arg("num_shards") = 256);

    // -- extend
    m.def("make_tci", &make_tci,
      py::arg("chunk_map"),
      py::arg("chunk_last_ts"),
      py::arg("chunk_size"),
      py::arg("total_chunks"),
      py::arg("out_dir"));



    m.def("extend_tci",
    [](const ChunkResultLite& prev,
        const std::vector<int64_t>& src_new,
        const std::vector<int64_t>& dst_new,
        const std::vector<double>&  ts_new,
        const std::vector<int64_t>& eid_new,
        bool duplicate_undirected) {

        std::vector<ChunkDelta> deltas;
        deltas.reserve(src_new.size() * 2);

        ChunkResultLite out = extend_impl_with_deltas(
            prev, src_new, dst_new, ts_new, eid_new, deltas, duplicate_undirected
        );

        // Convert deltas to NumPy 1-D arrays
        const ssize_t N = (ssize_t)deltas.size();

        py::array_t<int64_t> cids({N});
        py::array_t<int64_t> pos ({N});
        py::array_t<double>  ts  ({N});
        py::array_t<int64_t> eid ({N});
        py::array_t<int64_t> oth ({N});

        auto cids_m = cids.mutable_unchecked<1>();
        auto pos_m  = pos.mutable_unchecked<1>();
        auto ts_m   = ts.mutable_unchecked<1>();
        auto eid_m  = eid.mutable_unchecked<1>();
        auto oth_m  = oth.mutable_unchecked<1>();

        for (ssize_t i = 0; i < N; ++i) {
            const auto& d = deltas[(size_t)i];
            cids_m(i) = d.cid;
            pos_m(i)  = d.pos;
            ts_m(i)   = d.ts;
            eid_m(i)  = d.eid;
            oth_m(i)  = d.other;
        }

        return py::make_tuple(out, cids, pos, ts, eid, oth);
    },
    py::arg("prev"),
    py::arg("src_new"), py::arg("dst_new"), py::arg("ts_new"), py::arg("eid_new"),
    py::arg("duplicate_undirected") = true,
    R"pbdoc(
    Extend an existing TCI and also return deltas for tail in-place writes.
    Returns:
    (out, cids, pos, ts, eid, other)
    Where deltas correspond only to in-place tail writes performed during extend.
    )pbdoc"
    );


    m.def("extend_streaming_latestk_ordered_reuse", &extend_streaming_latestk_ordered_reuse,
        py::arg("prev"),
        py::arg("src_new"), py::arg("dst_new"), py::arg("ts_new"), py::arg("eid_new"),
        py::arg("duplicate_undirected") = true,
        R"pbdoc(
    Extend an existing TCI (later timestamps only), maintaining latest-K chunks per node.
    Policy:
    - Fill partially-filled tail chunk first
    - If <K chunks, append new chunk(s)
    - If already K and need a new chunk, evict oldest by shifting mapping and reuse its chunk-id on disk
    )pbdoc");
    // --- read_chunk convenience
    m.def("read_chunk", [](const std::string& out_dir, int64_t chunk_size, int64_t k) {
        vector<double>  t;
        vector<int64_t> e, o;
        std::string ts_path    = out_dir + "/ts.bin";
        std::string eid_path   = out_dir + "/eid.bin";
        std::string other_path = out_dir + "/other.bin";
        read_chunk(ts_path, eid_path, other_path, chunk_size, k, t, e, o);

        auto ts_arr  = py::array_t<double> ( { (ssize_t)1, (ssize_t)chunk_size }, t.data() );
        auto eid_arr = py::array_t<int64_t>( { (ssize_t)1, (ssize_t)chunk_size }, e.data() );
        auto oth_arr = py::array_t<int64_t>( { (ssize_t)1, (ssize_t)chunk_size }, o.data() );
        return py::make_tuple(py::array(ts_arr), py::array(eid_arr), py::array(oth_arr));
    }, py::arg("out_dir"), py::arg("chunk_size"), py::arg("k"));

    // --- ChunkReader (simple, no cache)
    class ChunkReader {
    public:
        ChunkReader(std::string out_dir, int64_t chunk_size)
        : out_dir_(std::move(out_dir)), chunk_size_(chunk_size)
        {
            ts_path_    = out_dir_ + "/ts.bin";
            eid_path_   = out_dir_ + "/eid.bin";
            other_path_ = out_dir_ + "/other.bin";
        }

        py::tuple read_chunk_np(int64_t k) const {
            vector<double>  t;
            vector<int64_t> e, o;
            read_chunk(ts_path_, eid_path_, other_path_, chunk_size_, k, t, e, o);
            auto ts_arr  = py::array_t<double> ( { (ssize_t)1, (ssize_t)chunk_size_ }, t.data() );
            auto eid_arr = py::array_t<int64_t>( { (ssize_t)1, (ssize_t)chunk_size_ }, e.data() );
            auto oth_arr = py::array_t<int64_t>( { (ssize_t)1, (ssize_t)chunk_size_ }, o.data() );
            return py::make_tuple(py::array(ts_arr), py::array(eid_arr), py::array(oth_arr));
        }

        std::string ts_path() const    { return ts_path_; }
        std::string eid_path() const   { return eid_path_; }
        std::string other_path() const { return other_path_; }
        int64_t chunk_size() const     { return chunk_size_; }

    private:
        std::string out_dir_;
        std::string ts_path_, eid_path_, other_path_;
        int64_t chunk_size_;
    };

    py::class_<ChunkReader>(m, "ChunkReader")
        .def(py::init<std::string, int64_t>(), py::arg("out_dir"), py::arg("chunk_size"))
        .def("read_chunk", &ChunkReader::read_chunk_np, py::arg("k"))
        .def_property_readonly("ts_path", &ChunkReader::ts_path)
        .def_property_readonly("eid_path", &ChunkReader::eid_path)
        .def_property_readonly("other_path", &ChunkReader::other_path)
        .def_property_readonly("chunk_size", &ChunkReader::chunk_size);

    // --- ChunkCache
    py::class_<ChunkCache>(m, "ChunkCache")
        .def(py::init<const std::string&, int64_t, size_t>(),
             py::arg("out_dir"), py::arg("chunk_size"), py::arg("capacity") = 100)
        .def("get_chunks", &ChunkCache::get_chunks_numpy, py::arg("ids"),
             R"pbdoc(Return (ts, eid, other) as NumPy arrays of shape (N, chunk_size))pbdoc")
        .def("apply_deltas", &ChunkCache::apply_deltas_numpy,
            py::arg("cids"), py::arg("pos"), py::arg("ts"), py::arg("eid"), py::arg("other"),
            R"pbdoc(Apply in-place updates to cached chunks. Arrays are 1-D, same length.)pbdoc")
        .def("clear", &ChunkCache::clear)
        .def_property_readonly("size", &ChunkCache::size)
        .def_property_readonly("capacity", &ChunkCache::capacity)
        .def_property_readonly("chunk_size", &ChunkCache::chunk_size);

#ifdef CHUNKIO_WITH_TORCH
    // If compiled with Torch, also expose the torch API
    py::class_<ChunkCache>(m, "TorchChunkCache", py::module_local())
        .def(py::init<const std::string&, int64_t, size_t>(),
             py::arg("out_dir"), py::arg("chunk_size"), py::arg("capacity") = 100)
        .def("get_chunks_torch", &ChunkCache::get_chunks_torch, py::arg("ids"),
             R"pbdoc(Return (ts, eid, other) as torch tensors of shape (N, chunk_size))pbdoc")
        .def("apply_deltas_torch", &ChunkCache::apply_deltas_torch,
             py::arg("cids"), py::arg("pos"), py::arg("ts"), py::arg("eid"), py::arg("other"),
             R"pbdoc(Apply in-place updates to cached chunks. Torch CPU tensors, 1-D, same length.)pbdoc");

#endif
}




