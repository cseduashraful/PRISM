#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

struct ChunkResult {
    std::vector<std::vector<double>> ts_chunks;
    std::vector<std::vector<int64_t>> eid_chunks;
    std::vector<std::vector<int64_t>> other_node_chunks; // <--- Added
    std::vector<std::vector<int64_t>> chunk_map;
    std::vector<std::vector<double>> chunk_last_ts;
};

// Forward declare C++ functions
ChunkResult preprocess(
    const std::vector<int64_t>& src,
    const std::vector<int64_t>& dst,
    const std::vector<double>& ts,
    const std::vector<int64_t>& eid,
    int64_t num_nodes,
    int chunk_size = 4,
    int max_chunk_per_node = 3
);

namespace py = pybind11;

py::dict preprocess_py(
    const std::vector<int64_t>& src,
    const std::vector<int64_t>& dst,
    const std::vector<double>& ts,
    const std::vector<int64_t>& eid,
    int64_t num_nodes,
    int chunk_size,
    int max_chunk_per_node
) {
    ChunkResult result = preprocess(src, dst, ts, eid, num_nodes, chunk_size, max_chunk_per_node);

    py::dict output;
    output["ts_chunks"] = result.ts_chunks;
    output["eid_chunks"] = result.eid_chunks;
    output["other_node_chunks"] = result.other_node_chunks; // <--- Added
    output["chunk_map"] = result.chunk_map;
    output["chunk_last_ts"] = result.chunk_last_ts;
    return output;
}

PYBIND11_MODULE(preprocessor, m) {
    m.doc() = "High-performance preprocessor module (C++ + OpenMP)";

    m.def("preprocess", &preprocess_py,
          py::arg("src"), py::arg("dst"), py::arg("ts"), py::arg("eid"),
          py::arg("num_nodes"), py::arg("chunk_size") = 4, py::arg("max_chunk_per_node") = 3);
}
