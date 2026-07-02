# Analysis of the Bluesky Social Network: the Scientific Community

Final project for the **Social Network Analysis** course — University of Pisa,
M.Sc. in Data Science and Business Informatics, A.Y. 2025/2026.

**Group members**
- Sara Biavasco
- Alice Giovagnini
- Maria Paula Pena Gamez

## Overview

We build and analyse the **follow network of the scientific community on
Bluesky** (a platform based on the open *AT Protocol*). Starting from five
thematic seed accounts, a snowball/BFS crawl over the public read endpoints
produces a connected network of **15,000 nodes** and **~1.67M edges**, enriched
with node attributes (profile metadata) and a sample of posts (text +
timestamps). On this network we run the analyses required by the four parts of
the project.

## Repository structure

```
.
├── data_collection/      # Part 1 — crawling & network construction
│   ├── api.py            #   minimal AT Protocol client (rate limit + backoff)
│   ├── config.py         #   parameters (seeds, MAX_NODES, pacing...)
│   ├── find_seeds.py     #   seed discovery via account search
│   ├── crawl_graph.py    #   snowball/BFS of the follow graph
│   ├── fetch_metadata.py #   node attributes + post sample
│   ├── build_network.py  #   induced subgraph + stats + compressed exports
│   ├── requirements.txt
│   └── data/             #   FINAL data (compressed) — see below
├── network_analysis/     # Part 2 & Part 3
│   ├── analysis_starter.py    # Part 2: stats, degree dist, ER/BA, centrality
│   ├── community_detection.py # Louvain / Label Propagation / Infomap (CDlib)
│   ├── link_prediction.py     # Part 3 (analytical): CN/Jaccard/AA/PA
│   ├── diffusion.py           # Part 3 (manipulation): SI/SIS/SIR/Threshold
│   └── game_theoretic.py      # extension: coordination-game cascade
├── open_problem/         # Part 4 — open question
│   └── open_question.py  #   topical echo chambers (TF-IDF + communities)
├── report/               # Part 5 — report (ACM template, acmart)
│   └── main.tex          #   compile on Overleaf (pdfLaTeX + BibTeX)
├── README.md
└── SNA Final Project.pdf # assignment text
```

## Data source

- **Source:** Bluesky Social, via the AT Protocol public read endpoints
  (`https://public.api.bsky.app`) — no authentication or API key required.
- **Node:** an account, identified by its DID (stable across handle changes).
- **Edge:** a *follow* relationship (directed; analysed as simple, undirected,
  unweighted for Part 2).
- **Node attributes:** handle, display name, bio, followers/follows/posts
  counts, account creation date.
- **Post layer:** up to 20 posts per account (text, timestamp, likes, reposts).

The **final, compressed data** is in `data_collection/data/`:

| File | Content |
|------|---------|
| `network_undirected.edgelist.gz` | simple undirected network (Part 2) |
| `network_directed.edgelist.gz`   | directed follow network |
| `node_attributes.csv.gz`         | node attribute table |
| `posts_sample.jsonl.gz`          | sampled posts (text + timestamps) |
| `communities.csv`                | Louvain community of each node |

Heavy intermediate files (`raw_edges.jsonl`, uncompressed edgelists, the raw
post dump, crawl checkpoint…) are **not** versioned — they are re-creatable by
running the scripts (see `.gitignore`).

## How to run

```bash
pip install -r data_collection/requirements.txt
```

**Part 1 — data collection** (run from `data_collection/`):
```bash
python find_seeds.py science        # (optional) pick seeds -> config.py
python crawl_graph.py               # build the follow graph (resumable)
python fetch_metadata.py --posts    # node attributes + post sample
python build_network.py             # final network + stats + .gz exports
```

**Parts 2–4** (run after the network is built):
```bash
python network_analysis/analysis_starter.py     # Part 2
python network_analysis/community_detection.py  # community detection
python network_analysis/link_prediction.py      # Part 3 (analytical)
python network_analysis/diffusion.py            # Part 3 (manipulation)
python network_analysis/game_theoretic.py       # extension (cascade)
python open_problem/open_question.py            # Part 4
```

## Report

The written report is in `report/` and uses the official ACM template
(`acmart.cls`). Compile `report/main.tex` on Overleaf with **pdfLaTeX +
BibTeX** (references in `biblio.bib`).
