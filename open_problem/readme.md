# Open Problem (Part 4)

**Question:** is the scientific community on Bluesky organised into *topical echo
chambers*? (1) Do network communities match coherent discussion topics? (2) Are
connected accounts more topically similar than unconnected ones (topical
homophily)? (3) How does that homophily evolve across account-creation
generations (2023 / 2024 / 2025 cohorts)?

`open_question.py` combines SNA with text mining: each node becomes a document
from its sampled posts, vectorised with TF-IDF; each Louvain community is
labelled by its top terms; the structure↔topic alignment is quantified by
clustering the documents into topical groups (KMeans) and comparing that
partition with the Louvain one (NMI/ARI); topical homophily (cosine similarity)
is measured for same- vs different-community pairs and for real edges vs
non-edges (with a significance test); finally, the homophily of the edges
internal to each account-creation cohort tracks the echo chamber over time.

Figures (`open_question_homophily.png`, `open_question_temporal.png`) are saved
to `../data_collection/data/figures/`; the copies used in the report are in
`plots/`.

Requires `communities.csv` (from `../network_analysis/community_detection.py`)
and the post sample in `../data_collection/data/`.
