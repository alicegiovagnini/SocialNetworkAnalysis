# Data — Collected Artifacts

Network produced by the pipeline (`crawl_graph.py` → `fetch_metadata.py` →
`build_network.py`) plus the community-detection output reused by later analyses.

Large artifacts are stored **gzipped** (`.gz`) to fit on GitHub; small tables are
**also kept as plain `.csv`** for direct reading.

```bash
gzip -dk network_undirected.edgelist.gz   # decompress, keep the .gz
```
```python
import gzip, networkx as nx
with gzip.open("network_undirected.edgelist.gz", "rt") as f:
    G = nx.read_edgelist(f)
```

## Contents

| File | Format | Size | Description |
|------|--------|------|-------------|
| `node_attributes.csv` | CSV | 1.6 MB | Node attribute table (decompressed copy). |
| `node_attributes.csv.gz` | gzip CSV | 0.8 MB | Same table, compressed. |
| `node_attributes.json.gz` | gzip JSON | ~1.5 MB | Full attribute dict incl. the free-text **bio/description** (absent from the CSV); used by `community_detection.py` for the per-community keyword labels. |
| `communities.csv` | CSV | 0.5 MB | Louvain community per node. |
| `network_undirected.edgelist.gz` | gzip | 27 MB | **Simple, undirected, unweighted** network — Part 2. |
| `network_directed.edgelist.gz` | gzip | 31 MB | Directed follow network (`u → v`). |
| `posts_sample.jsonl.gz` | gzip JSONL | 33 MB | Posts per node (text + timestamps). |
| `network_viz.gexf` | GEXF | 3.5 MB | Community-stratified ~2k-node sample for Gephi (from `export_gephi.py`). |
| `figures/` | PNG | <1 MB | All figures produced by the analysis scripts (Parts 2–4). |

`network_full.gexf` (the full 15k-node network in GEXF, ~181 MB) is also
written by `export_gephi.py` but exceeds GitHub's file-size limit: it stays
local and is re-creatable by re-running the script.

## Schemas

**`node_attributes.csv`** — one row per node (~15k): `did` (node id), `handle`,
`displayName`, `followersCount`, `followsCount`, `postsCount`, `createdAt`.

**`communities.csv`** — `did`, `louvain_community` (joins on `node_attributes.csv`).

**Edge lists** — one `source target` pair of DIDs per line, no attributes. The
undirected file lists each edge once; the directed file keeps the follow direction.

**`posts_sample.jsonl.gz`** — one JSON post per line: `author` (DID), `uri`,
`createdAt`, `text`, `replyParentUri`, `likeCount`, `repostCount`.

## Notes

- Nodes are keyed by **DID** (stable across handle renames) — the safe join key.
- **Louvain is a stochastic heuristic**: re-running `community_detection.py`
  yields a slightly different (but structurally equivalent) partition.
  `communities.csv` is the **frozen partition** all the results reported in the
  report refer to.
- Intermediate crawl files (`nodes.json`, `raw_edges.jsonl`, `crawl_state.pkl`, …)
  and the uncompressed copies (`node_attributes.json`, `posts_sample.jsonl`) are
  not committed (gitignored) — they are re-creatable from the pipeline / the
  `.gz` files. The compressed `node_attributes.json.gz` (which also carries the
  free-text bio/description, absent from the CSV) **is** committed, so
  `community_detection.py` reproduces the per-community keyword labels offline.
