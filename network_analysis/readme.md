# Network Analysis (Parts 2 & 3)

Analyses on the network built in `../data_collection/` (edge lists in
`../data_collection/data/`). Figures go to `plots/`.

| Script | Part | What it does |
|--------|------|--------------|
| `analysis_starter.py` | 2 | Basic stats, degree distribution, clustering/density, paths, centrality; ER/BA comparison. |
| `community_detection.py` | 2 | Louvain, Label Propagation, Infomap (CDlib); modularity, conductance, NMI/ARI; semantic labels. Saves `communities.csv`. |
| `link_prediction.py` | 3 (analytical) | 80/20 split + Common Neighbors, Jaccard, Adamic–Adar, Pref. Attachment (AUC, precision@k). |
| `diffusion.py` | 3 (manipulation) | NDlib SI/SIS/SIR/Threshold on the real network vs ER/BA, varying params and seeds. |
| `game_theoretic.py` | extension | Coordination-game cascade: critical threshold vs seeding strategy; communities as diffusion barriers. |

`community_detection.py` produces `communities.csv`, reused by `open_problem/`.
