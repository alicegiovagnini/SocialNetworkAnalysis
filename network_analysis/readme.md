# Network Analysis (Parts 2 & 3)

Analyses on the network built in `../data_collection/` (edge lists in
`../data_collection/data/`). Scripts save their figures to
`../data_collection/data/figures/`; the figures used in the report are also
kept in `plots/`.

| Script | Part | What it does |
|--------|------|--------------|
| `analysis_starter.py` | 2 | Basic stats, degree distribution, clustering/density, paths, centrality; ER/BA comparison. |
| `community_detection.py` | 2 | Louvain, Label Propagation, Infomap (CDlib); modularity, conductance, NMI/ARI; semantic labels. Saves `communities.csv`. |
| `export_gephi.py` | 2 | Exports the network to GEXF for Gephi: `network_full.gexf` (all 15k nodes) and `network_viz.gexf` (community-stratified ~2k-node sample), with `community`, `degree` and `handle` node attributes for the ForceAtlas2 visualisation. |
| `temporal_analysis.py` | 2 | Account-creation dynamics: adoption waves over time (`account_creation.png`) and disciplinary mix of the 2023 founding cohort vs the Nov-2024 migration (`cohort_composition.png`). |
| `link_prediction.py` | 3 (analytical) | 80/20 split + Common Neighbors, Jaccard, Adamic–Adar, Pref. Attachment (AUC, precision@k). |
| `diffusion.py` | 3 (manipulation) | NDlib SI/SIS/SIR/Threshold on the real network vs ER/BA, varying params and seeds. |
| `game_theoretic.py` | 3 (custom model) | Coordination-game cascade coded from scratch: (1) critical threshold q* vs seeding strategy; (2) Louvain communities as diffusion barriers; (3) community-aware semantic variant that down-weights cross-group coordination by ω, on topical groups derived from post text (TF-IDF + KMeans). |

`community_detection.py` produces `communities.csv`, reused by
`export_gephi.py`, `temporal_analysis.py`, `game_theoretic.py` and
`../open_problem/`.
