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
             R"pbdoc(Return (ts, eid, other) as torch tensors of shape (N, chunk_size))pbdoc");
#endif
}





// // chunk_streaming.cpp
// // Streaming chunk preprocessor with on-disk storage + read_chunk + pybind11 bindings.
// // Builds into a Python extension module named `chunkio`.
// // Usage (Python):
// //   import chunkio
// //   res = chunkio.preprocess_streaming(src, dst, ts, eid, num_nodes,
// //                                      chunk_size=4, max_chunk_per_node=3,
// //                                      duplicate_undirected=True,
// //                                      out_dir="preproc_out", num_shards=256)
// //   ts_k, eid_k, other_k = chunkio.read_chunk(res.out_dir, res.chunk_size, 0)
// //   reader = chunkio.ChunkReader(res.out_dir, res.chunk_size)
// //   ts_k, eid_k, other_k = reader.read_chunk(0)

// #include <algorithm>
// #include <cstdint>
// #include <fstream>
// #include <filesystem>
// #include <limits>
// #include <stdexcept>
// #include <string>
// #include <tuple>
// #include <vector>

// #include <omp.h>

// // ---- pybind11 ----
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/numpy.h>

// namespace py = pybind11;

// using std::int64_t;
// using std::size_t;
// using std::vector;

// struct ChunkResultLite {
//     vector<vector<int64_t>> chunk_map;     // [num_nodes][max_chunk_per_node], -1 if none
//     vector<vector<double>>  chunk_last_ts; // [num_nodes][max_chunk_per_node]
//     int64_t chunk_size{0};
//     int64_t total_chunks{0};
//     std::string ts_path, eid_path, other_path, out_dir;
// };

// struct DiskEdge {
//     int64_t node;  // owner node
//     double  ts;
//     int64_t eid;
//     int64_t other;
// };

// static inline size_t shard_id_for(int64_t node, size_t num_shards) {
//     return static_cast<size_t>(node) % num_shards;
// }

// ChunkResultLite preprocess_streaming(
//     const vector<int64_t>& src,
//     const vector<int64_t>& dst,
//     const vector<double>&  ts,
//     const vector<int64_t>& eid,
//     int64_t num_nodes,
//     int chunk_size = 4,
//     int max_chunk_per_node = 3,
//     bool duplicate_undirected = true,
//     const std::string& out_dir = "preproc_out",
//     size_t num_shards = 256
// ) {
//     std::filesystem::create_directories(out_dir);
//     const std::string ts_path    = out_dir + "/ts.bin";
//     const std::string eid_path   = out_dir + "/eid.bin";
//     const std::string other_path = out_dir + "/other.bin";

//     // -------- Phase 0: open shard writers --------
//     std::vector<std::ofstream> shard_files;
//     shard_files.reserve(num_shards);
//     for (size_t s = 0; s < num_shards; ++s) {
//         std::string path = out_dir + "/shard_" + std::to_string(s) + ".bin";
//         shard_files.emplace_back(path, std::ios::binary | std::ios::out | std::ios::trunc);
//         if (!shard_files.back()) throw std::runtime_error("Failed to open shard file: " + path);
//     }

//     // -------- Phase 1: stream records to shards (append-only) --------
//     const size_t M = src.size();
//     for (size_t i = 0; i < M; ++i) {
//         int64_t u = src[i], v = dst[i];
//         double  t = ts[i];
//         int64_t e = eid[i];

//         {   // record for dst (owner = v, other = u)
//             DiskEdge rec{v, t, e, u};
//             auto sid = shard_id_for(v, num_shards);
//             shard_files[sid].write(reinterpret_cast<const char*>(&rec), sizeof(DiskEdge));
//         }
//         if (duplicate_undirected && u != v) {
//             DiskEdge rec{u, t, e, v};
//             auto sid = shard_id_for(u, num_shards);
//             shard_files[sid].write(reinterpret_cast<const char*>(&rec), sizeof(DiskEdge));
//         }
//     }
//     for (auto& f : shard_files) f.close();

//     // -------- Phase 2: open final outputs --------
//     std::ofstream ts_out   (ts_path,    std::ios::binary | std::ios::out | std::ios::trunc);
//     std::ofstream eid_out  (eid_path,   std::ios::binary | std::ios::out | std::ios::trunc);
//     std::ofstream other_out(other_path, std::ios::binary | std::ios::out | std::ios::trunc);
//     if (!ts_out || !eid_out || !other_out) throw std::runtime_error("Failed to open final chunk files.");

//     const double INF = std::numeric_limits<double>::infinity();

//     vector<vector<int64_t>> chunk_map(num_nodes, vector<int64_t>(max_chunk_per_node, -1));
//     vector<vector<double>>  chunk_last_ts(num_nodes, vector<double>(max_chunk_per_node, INF));

//     int64_t global_chunk_id = 0;

//     // -------- Phase 3: process shards --------
//     for (size_t s = 0; s < num_shards; ++s) {
//         std::string path = out_dir + "/shard_" + std::to_string(s) + ".bin";
//         std::ifstream in(path, std::ios::binary | std::ios::in);
//         if (!in) continue;

//         in.seekg(0, std::ios::end);
//         std::streampos sz = in.tellg();
//         in.seekg(0, std::ios::beg);
//         size_t nrec = static_cast<size_t>(sz / sizeof(DiskEdge));
//         vector<DiskEdge> recs(nrec);
//         if (nrec) in.read(reinterpret_cast<char*>(recs.data()), nrec * sizeof(DiskEdge));
//         in.close();
//         std::filesystem::remove(path);

//         if (recs.empty()) continue;

//         std::sort(recs.begin(), recs.end(), [](const DiskEdge& a, const DiskEdge& b){
//             if (a.node != b.node) return a.node < b.node;
//             return a.ts < b.ts;
//         });

//         size_t i = 0;
//         while (i < recs.size()) {
//             int64_t node = recs[i].node;
//             size_t j = i;
//             while (j < recs.size() && recs[j].node == node) ++j; // [i, j)

//             size_t deg = j - i;
//             for (size_t start = 0; start < deg; start += (size_t)chunk_size) {
//                 size_t end = std::min(start + (size_t)chunk_size, deg);
//                 size_t len = end - start;

//                 // fixed-size chunks (pad)
//                 vector<double>  t_chunk(chunk_size, INF);
//                 vector<int64_t> e_chunk(chunk_size, -1);
//                 vector<int64_t> o_chunk(chunk_size, -1);

//                 for (size_t k = 0; k < len; ++k) {
//                     const auto& r = recs[i + start + k];
//                     t_chunk[k] = r.ts;
//                     e_chunk[k] = r.eid;
//                     o_chunk[k] = r.other;
//                 }

//                 ts_out.write(reinterpret_cast<const char*>(t_chunk.data()), sizeof(double)  * chunk_size);
//                 eid_out.write(reinterpret_cast<const char*>(e_chunk.data()), sizeof(int64_t) * chunk_size);
//                 other_out.write(reinterpret_cast<const char*>(o_chunk.data()), sizeof(int64_t) * chunk_size);

//                 // per-node metadata: first max_chunk_per_node chunks
//                 int64_t local_idx = (int64_t)(start / (size_t)chunk_size);
//                 if (local_idx < (int64_t)chunk_last_ts[node].size()) {
//                     chunk_map[node][local_idx] = global_chunk_id;
//                     chunk_last_ts[node][local_idx] = t_chunk[ (len == (size_t)chunk_size) ? (chunk_size - 1) : (len - 1) ];
//                 }

//                 ++global_chunk_id;
//             }

//             i = j;
//         }
//     }

//     ts_out.close();
//     eid_out.close();
//     other_out.close();

//     return { std::move(chunk_map), std::move(chunk_last_ts),
//              (int64_t)chunk_size, global_chunk_id,
//              ts_path, eid_path, other_path, out_dir };
// }

// // ---- read_chunk (file-scope) ----
// void read_chunk(const std::string& ts_path,
//                 const std::string& eid_path,
//                 const std::string& other_path,
//                 int64_t chunk_size,
//                 int64_t k,
//                 vector<double>&  t_chunk,
//                 vector<int64_t>& e_chunk,
//                 vector<int64_t>& o_chunk)
// {
//     if (k < 0) throw std::invalid_argument("k must be non-negative");

//     std::ifstream ts_in(ts_path, std::ios::binary);
//     std::ifstream eid_in(eid_path, std::ios::binary);
//     std::ifstream oth_in(other_path, std::ios::binary);
//     if (!ts_in || !eid_in || !oth_in) throw std::runtime_error("Failed to open chunk files.");

//     const int64_t offD = k * chunk_size * (int64_t)sizeof(double);
//     const int64_t offI = k * chunk_size * (int64_t)sizeof(int64_t);

//     ts_in.seekg(offD);
//     eid_in.seekg(offI);
//     oth_in.seekg(offI);

//     t_chunk.resize((size_t)chunk_size);
//     e_chunk.resize((size_t)chunk_size);
//     o_chunk.resize((size_t)chunk_size);

//     ts_in.read(reinterpret_cast<char*>(t_chunk.data()), sizeof(double)  * (size_t)chunk_size);
//     eid_in.read(reinterpret_cast<char*>(e_chunk.data()), sizeof(int64_t) * (size_t)chunk_size);
//     oth_in.read(reinterpret_cast<char*>(o_chunk.data()), sizeof(int64_t) * (size_t)chunk_size);

//     if (!ts_in || !eid_in || !oth_in) {
//         throw std::runtime_error("read_chunk: I/O error or k out of range");
//     }
// }

// // ---- Reader class for Python ----
// class ChunkReader {
// public:
//     ChunkReader(std::string out_dir, int64_t chunk_size)
//     : out_dir_(std::move(out_dir)), chunk_size_(chunk_size)
//     {
//         ts_path_    = out_dir_ + "/ts.bin";
//         eid_path_   = out_dir_ + "/eid.bin";
//         other_path_ = out_dir_ + "/other.bin";
//     }

//     // Returns (ts, eid, other) as numpy arrays of shape (chunk_size,)
//     py::tuple read_chunk_np(int64_t k) const {
//         vector<double>  t;
//         vector<int64_t> e, o;
//         read_chunk(ts_path_, eid_path_, other_path_, chunk_size_, k, t, e, o);

//         auto ts_arr  = py::array_t<double>(  (size_t)chunk_size_, t.data() );
//         auto eid_arr = py::array_t<int64_t>( (size_t)chunk_size_, e.data() );
//         auto oth_arr = py::array_t<int64_t>( (size_t)chunk_size_, o.data() );
//         // Return copies so NumPy owns the memory independently
//         return py::make_tuple(py::array(ts_arr), py::array(eid_arr), py::array(oth_arr));
//     }

//     std::string ts_path() const    { return ts_path_; }
//     std::string eid_path() const   { return eid_path_; }
//     std::string other_path() const { return other_path_; }
//     int64_t chunk_size() const     { return chunk_size_; }

// private:
//     std::string out_dir_;
//     std::string ts_path_, eid_path_, other_path_;
//     int64_t chunk_size_;
// };

// // ---- pybind11 module ----
// PYBIND11_MODULE(chunkio, m) {
//     m.doc() = "Streaming chunk preprocessor with on-disk storage and Python access";

//     py::class_<ChunkResultLite>(m, "ChunkResultLite")
//         .def_readonly("chunk_map",     &ChunkResultLite::chunk_map)
//         .def_readonly("chunk_last_ts", &ChunkResultLite::chunk_last_ts)
//         .def_readonly("chunk_size",    &ChunkResultLite::chunk_size)
//         .def_readonly("total_chunks",  &ChunkResultLite::total_chunks)
//         .def_readonly("ts_path",       &ChunkResultLite::ts_path)
//         .def_readonly("eid_path",      &ChunkResultLite::eid_path)
//         .def_readonly("other_path",    &ChunkResultLite::other_path)
//         .def_readonly("out_dir",       &ChunkResultLite::out_dir);

//     m.def("preprocess_streaming", &preprocess_streaming,
//           py::arg("src"), py::arg("dst"), py::arg("ts"), py::arg("eid"),
//           py::arg("num_nodes"),
//           py::arg("chunk_size") = 4,
//           py::arg("max_chunk_per_node") = 3,
//           py::arg("duplicate_undirected") = true,
//           py::arg("out_dir") = "preproc_out",
//           py::arg("num_shards") = 256,
//           R"pbdoc(
//               Stream edges to disk, sort per node, emit fixed-size padded chunks to ts/eid/other *.bin files.
//               Returns small in-memory metadata and file paths.
//           )pbdoc");

//     m.def("read_chunk", [](const std::string& out_dir, int64_t chunk_size, int64_t k) {
//         vector<double>  t;
//         vector<int64_t> e, o;
//         std::string ts_path    = out_dir + "/ts.bin";
//         std::string eid_path   = out_dir + "/eid.bin";
//         std::string other_path = out_dir + "/other.bin";
//         read_chunk(ts_path, eid_path, other_path, chunk_size, k, t, e, o);

//         auto ts_arr  = py::array_t<double>(  (size_t)chunk_size, t.data() );
//         auto eid_arr = py::array_t<int64_t>( (size_t)chunk_size, e.data() );
//         auto oth_arr = py::array_t<int64_t>( (size_t)chunk_size, o.data() );
//         return py::make_tuple(py::array(ts_arr), py::array(eid_arr), py::array(oth_arr));
//     }, py::arg("out_dir"), py::arg("chunk_size"), py::arg("k"),
//        R"pbdoc(Read a chunk k from out_dir/{ts,eid,other}.bin)pbdoc");

//     py::class_<ChunkReader>(m, "ChunkReader")
//         .def(py::init<std::string, int64_t>(), py::arg("out_dir"), py::arg("chunk_size"))
//         .def("read_chunk", &ChunkReader::read_chunk_np, py::arg("k"))
//         .def_property_readonly("ts_path", &ChunkReader::ts_path)
//         .def_property_readonly("eid_path", &ChunkReader::eid_path)
//         .def_property_readonly("other_path", &ChunkReader::other_path)
//         .def_property_readonly("chunk_size", &ChunkReader::chunk_size);
// }
