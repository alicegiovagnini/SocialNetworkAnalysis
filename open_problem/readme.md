# Open Problem (Part 4)

**Question:** is the scientific community on Bluesky organised into *topical echo
chambers*? Do network communities match coherent topics, and are connected
accounts more topically similar than unconnected ones (topical homophily)?

`open_question.py` combines SNA with text mining: each node becomes a document
from its sampled posts, vectorised with TF-IDF; each Louvain community is labelled
by its top terms, and topical homophily (cosine similarity) is measured for
same- vs different-community pairs and for real edges vs non-edges.

Requires `communities.csv` (from `../network_analysis/community_detection.py`)
and the post sample in `../data_collection/data/`.
