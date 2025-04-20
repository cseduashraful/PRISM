#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Forward declare the C++ functions from preprocess.cpp
struct ChunkResult {
    std::vector<std::vector<double>> ts_chunks;
    std::vector<std::vector<int64_t>> eid_chunks;
    std::vector<std::vector<int64_t>> chunk_map;
    std::vector<std::vector<double>> chunk_last_ts;
};

ChunkResult preprocess(
    const std::vector<int64_t>& src,
    const std::vector<int64_t>& dst,
    const std::vector<double>& ts,
    const std::vector<int64_t>& eid,
    int64_t num_nodes,
    int chunk_size = 4,
    int max_chunk_per_node = 3
);

void incremental_update(
    const std::vector<int64_t>& src_new,
    const std::vector<int64_t>& dst_new,
    const std::vector<double>& ts_new,
    const std::vector<int64_t>& eid_new,
    std::vector<std::vector<double>>& ts_chunks,
    std::vector<std::vector<int64_t>>& eid_chunks,
    std::vector<std::vector<int64_t>>& chunk_map,
    std::vector<std::vector<double>>& chunk_last_ts,
    int chunk_size,
    int max_chunk_per_node,
    int64_t& global_chunk_counter
);

namespace py = pybind11;

// This function converts the ChunkResult into a Python dictionary
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
    output["chunk_map"] = result.chunk_map;
    output["chunk_last_ts"] = result.chunk_last_ts;
    return output;
}

PYBIND11_MODULE(preprocessor, m) {
    m.doc() = "High-performance preprocessor module (C++ + OpenMP)";

    m.def("preprocess", &preprocess_py,
          py::arg("src"), py::arg("dst"), py::arg("ts"), py::arg("eid"),
          py::arg("num_nodes"), py::arg("chunk_size") = 4, py::arg("max_chunk_per_node") = 3);

    m.def("incremental_update", &incremental_update,
          py::arg("src_new"), py::arg("dst_new"), py::arg("ts_new"), py::arg("eid_new"),
          py::arg("ts_chunks"), py::arg("eid_chunks"), py::arg("chunk_map"), py::arg("chunk_last_ts"),
          py::arg("chunk_size"), py::arg("max_chunk_per_node"), py::arg("global_chunk_counter"));
}
