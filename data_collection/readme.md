# Data Collection — Bluesky

Builds the social network **from scratch** from Bluesky (pre-built datasets such
as SNAP/KONECT are not allowed by the assignment).

**Why Bluesky:** open AT Protocol, public read endpoints (`public.api.bsky.app`),
no auth and no API key. Exposes the follow graph, node attributes and post
timestamps.

## What the network models

- **Node** = a Bluesky account, keyed by its stable **DID**.
- **Edge** = a *follow* (`u → v` = *u follows v*). Part 2 uses the **simple,
  undirected, unweighted** version.
- **Attributes** = handle, displayName, bio, followers/follows/posts counts,
  createdAt.

## Collection: snowball (BFS) sampling

Start from 1–5 thematic seed accounts, BFS over their follows until `MAX_NODES`
(≥ 10–15k) is reached, then keep recording edges internal to the collected set.
The final network is the subgraph induced by those nodes.

## Files

| File | Role |
|------|------|
| `config.py` | Shared parameters (seeds, caps, pacing, output dir). |
| `api.py` | Wrapper around the public endpoints (pacing + backoff retry). |
| `find_seeds.py` | Search accounts by keyword to pick the seeds. |
| `crawl_graph.py` | **Stage 1** — build the follow graph (resumable). |
| `fetch_metadata.py` | **Stage 2** — download node attributes (and optionally posts). |
| `build_network.py` | **Stage 3** — assemble the network, stats and compressed exports. |
| `data/` | Produced artifacts — see `data/readme.md`. |

## Run

```bash
pip install -r requirements.txt

python find_seeds.py <keywords>   # optional: pick seeds → config.py SEED_HANDLES
python crawl_graph.py             # resumable: re-run to resume
python fetch_metadata.py [--posts]
python build_network.py
```

Outputs land in `data/` (large files gzipped, small tables also as plain `.csv`).

## Notes for the report

- Snowball from thematic seeds → the sample represents the community *around the
  seeds*, not the whole platform; the seed choice conditions the structure.
- `MAX_FOLLOWS_PAGES_PER_NODE` caps mega-hubs to bound the number of requests.
- The follow graph is directed; Part 2 uses the required undirected simplification.
- All requests are read-only on public endpoints, with pacing and 429 backoff.
