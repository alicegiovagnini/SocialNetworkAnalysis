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
| `communities.csv` | CSV | 0.5 MB | Louvain community per node. |
| `network_undirected.edgelist.gz` | gzip | 28 MB | **Simple, undirected, unweighted** network — Part 2. |
| `network_directed.edgelist.gz` | gzip | 32 MB | Directed follow network (`u → v`). |
| `posts_sample.jsonl.gz` | gzip JSONL | 34 MB | Posts per node (text + timestamps). |

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
- Intermediate crawl files (`nodes.json`, `raw_edges.jsonl`, `crawl_state.pkl`, …)
  are not needed for the analyses and are not committed.
