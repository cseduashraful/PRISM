#include <tuple>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <cstdint>
#include <vector>

using namespace std;

struct ChunkResult {
    vector<vector<double>> ts_chunks;
    vector<vector<int64_t>> eid_chunks;
    vector<vector<int64_t>> other_node_chunks;
    vector<vector<int64_t>> chunk_map;
    vector<vector<double>> chunk_last_ts;
};

const double INF = numeric_limits<double>::infinity();
using Edge = tuple<double, int64_t, int64_t>; // (timestamp, eid, other_node)

ChunkResult preprocess(
    const vector<int64_t>& src,
    const vector<int64_t>& dst,
    const vector<double>& ts,
    const vector<int64_t>& eid,
    int64_t num_nodes,
    int chunk_size = 4,
    int max_chunk_per_node = 3
) {
    vector<vector<Edge>> node_edges(num_nodes);

    for (size_t i = 0; i < dst.size(); i++) {
        node_edges[dst[i]].emplace_back(ts[i], eid[i], src[i]);
        node_edges[src[i]].emplace_back(ts[i], eid[i], dst[i]);
    }

    vector<vector<double>> ts_chunks;
    vector<vector<int64_t>> eid_chunks;
    vector<vector<int64_t>> other_node_chunks;
    vector<vector<int64_t>> chunk_map(num_nodes, vector<int64_t>(max_chunk_per_node, -1));
    vector<vector<double>> chunk_last_ts(num_nodes, vector<double>(max_chunk_per_node, INF));

    int64_t global_chunk_counter = 0;

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int node = 0; node < num_nodes; node++) {
            auto& edges = node_edges[node];
            sort(edges.begin(), edges.end());

            vector<int64_t> local_chunk_ids;
            vector<double> local_last_ts;

            for (size_t i = 0; i < edges.size(); i += chunk_size) {
                vector<double> t_chunk;
                vector<int64_t> e_chunk;
                vector<int64_t> o_chunk;

                size_t end_i = min(i + chunk_size, edges.size());
                for (size_t j = i; j < end_i; j++) {
                    t_chunk.push_back(get<0>(edges[j]));
                    e_chunk.push_back(get<1>(edges[j]));
                    o_chunk.push_back(get<2>(edges[j]));
                }

                while (t_chunk.size() < (size_t)chunk_size) {
                    t_chunk.push_back(INF);
                    e_chunk.push_back(-1);
                    o_chunk.push_back(-1);
                }

                #pragma omp critical
                {
                    int64_t chunk_id = global_chunk_counter++;
                    ts_chunks.push_back(t_chunk);
                    eid_chunks.push_back(e_chunk);
                    other_node_chunks.push_back(o_chunk);

                    if (local_chunk_ids.size() < (size_t)max_chunk_per_node) {
                        local_chunk_ids.push_back(chunk_id);
                        local_last_ts.push_back(t_chunk[chunk_size - 1]);
                    }
                }
            }

            while (local_chunk_ids.size() < (size_t)max_chunk_per_node) {
                local_chunk_ids.push_back(-1);
                local_last_ts.push_back(INF);
            }

            chunk_map[node] = local_chunk_ids;
            chunk_last_ts[node] = local_last_ts;
        }
    }

    return {ts_chunks, eid_chunks, other_node_chunks, chunk_map, chunk_last_ts};
}
