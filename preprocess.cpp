// preprocess.cpp
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
    vector<vector<int64_t>> chunk_map;
    vector<vector<double>> chunk_last_ts;
};

const double INF = numeric_limits<double>::infinity();
using Edge = tuple<double, int64_t>;

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
        node_edges[dst[i]].emplace_back(ts[i], eid[i]);
        node_edges[src[i]].emplace_back(ts[i], eid[i]);
    }

    vector<vector<double>> ts_chunks;
    vector<vector<int64_t>> eid_chunks;
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

                size_t end_i = min(i + chunk_size, edges.size());
                for (size_t j = i; j < end_i; j++) {
                    t_chunk.push_back(get<0>(edges[j]));
                    e_chunk.push_back(get<1>(edges[j]));
                }

                while (t_chunk.size() < (size_t)chunk_size) {
                    t_chunk.push_back(INF);
                    e_chunk.push_back(-1);
                }

                #pragma omp critical
                {
                    int64_t chunk_id = global_chunk_counter++;
                    ts_chunks.push_back(t_chunk);
                    eid_chunks.push_back(e_chunk);

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

    return {ts_chunks, eid_chunks, chunk_map, chunk_last_ts};
}

void incremental_update(
    const vector<int64_t>& src_new,
    const vector<int64_t>& dst_new,
    const vector<double>& ts_new,
    const vector<int64_t>& eid_new,
    vector<vector<double>>& ts_chunks,
    vector<vector<int64_t>>& eid_chunks,
    vector<vector<int64_t>>& chunk_map,
    vector<vector<double>>& chunk_last_ts,
    int chunk_size,
    int max_chunk_per_node,
    int64_t& global_chunk_counter
) {
    int num_nodes = chunk_map.size();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < src_new.size(); i++) {
        for (int rep = 0; rep < 2; rep++) {
            int node = (rep == 0) ? src_new[i] : dst_new[i];

            if (node >= num_nodes) continue;

            int last_chunk_id = -1;
            for (int j = 0; j < max_chunk_per_node; j++) {
                if (chunk_map[node][j] != -1) {
                    last_chunk_id = chunk_map[node][j];
                } else {
                    break;
                }
            }

            bool inserted = false;

            if (last_chunk_id != -1) {
                auto& t_chunk = ts_chunks[last_chunk_id];
                auto& e_chunk = eid_chunks[last_chunk_id];

                for (int pos = 0; pos < chunk_size; pos++) {
                    if (t_chunk[pos] == INF && e_chunk[pos] == -1) {
                        t_chunk[pos] = ts_new[i];
                        e_chunk[pos] = eid_new[i];
                        inserted = true;
                        break;
                    }
                }
            }

            if (!inserted) {
                vector<double> new_t_chunk(chunk_size, INF);
                vector<int64_t> new_e_chunk(chunk_size, -1);

                new_t_chunk[0] = ts_new[i];
                new_e_chunk[0] = eid_new[i];

                int64_t new_chunk_id;

                #pragma omp critical
                {
                    new_chunk_id = global_chunk_counter++;
                    ts_chunks.push_back(new_t_chunk);
                    eid_chunks.push_back(new_e_chunk);
                }

                for (int j = 0; j < max_chunk_per_node; j++) {
                    if (chunk_map[node][j] == -1) {
                        chunk_map[node][j] = new_chunk_id;
                        chunk_last_ts[node][j] = INF;
                        break;
                    }
                }
            }
        }
    }
}
