"""
PART 4 - Open question.

Research question:
  "Is the Bluesky scientific community organised into topical ECHO CHAMBERS?
   Do the communities found from the follow structure correspond to coherent
   discussion topics, and is there TOPICAL HOMOPHILY (do connected accounts /
   accounts in the same community talk about similar things)?"

Combines SNA tools (community detection) with a textual analysis (TF-IDF) of
the posts, exploiting the additional information collected during crawling, as
encouraged by the assignment. In particular it links partition quality to an
external data factor (homophily of node labels/topics), as suggested by the
report guidelines.

Pipeline:
  1. group the posts by author -> one "document" per node;
  2. vectorise with TF-IDF;
  3. for each (Louvain) community extract the keywords -> topic;
  4. measure topical homophily:
       (a) mean cosine similarity between nodes in the SAME community vs
           DIFFERENT communities;
       (b) mean cosine similarity between the endpoints of REAL edges vs NON-edges.
     If "same community" and "real edges" have higher similarity =>
     evidence of topical homophily / echo chambers.

Usage:
    python open_question.py
"""

import os
import csv
import json
import gzip
import random
import collections

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

import config

random.seed(42)
np.random.seed(42)


def load_graph(data_dir):
    plain = os.path.join(data_dir, "network_undirected.edgelist")
    if os.path.exists(plain):
        return nx.read_edgelist(plain)
    with gzip.open(plain + ".gz", "rt") as f:
        return nx.parse_edgelist(f)


def load_communities(data_dir):
    path = os.path.join(data_dir, "communities.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("communities.csv missing: run "
                                "community_detection.py first")
    comm = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            comm[row["did"]] = int(row["louvain_community"])
    return comm


def load_texts(data_dir):
    """did -> concatenated text of its posts."""
    texts = collections.defaultdict(list)
    path = os.path.join(data_dir, "posts_sample.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("text"):
                texts[r["author"]].append(r["text"])
    return {did: " ".join(t) for did, t in texts.items()}


def cosine_pairs(X, idx, pairs):
    """Cosine similarity for a list of index pairs (X is L2-normalised)."""
    sims = []
    for u, v in pairs:
        if u in idx and v in idx:
            sims.append(float(X[idx[u]].multiply(X[idx[v]]).sum()))
    return sims


def main():
    data_dir = config.DATA_DIR
    G = load_graph(data_dir)
    comm = load_communities(data_dir)
    texts = load_texts(data_dir)
    print(f"[oq] nodes with posts: {len(texts)} / {G.number_of_nodes()}")

    # usable nodes: they have text and a community
    dids = [d for d in texts if d in comm]
    docs = [texts[d] for d in dids]
    idx = {d: i for i, d in enumerate(dids)}

    # --- TF-IDF ---
    # extended stopwords: English + platform/URL tokens and generic words
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    extra = {"bsky", "social", "com", "www", "org", "net", "http", "https",
             "doi", "new", "just", "like", "people", "really", "today", "via",
             "amp", "don", "ve", "ll", "twitter", "post", "thread"}
    stop = list(ENGLISH_STOP_WORDS.union(extra))
    vec = TfidfVectorizer(max_features=3000, stop_words=stop,
                          min_df=5, token_pattern=r"[a-zA-Z]{3,}")
    X = vec.fit_transform(docs)          # already L2-normalised (norm='l2')
    vocab = np.array(vec.get_feature_names_out())
    print(f"[oq] TF-IDF: {X.shape[0]} documents, {X.shape[1]} terms")

    # --- 3. keywords for the largest communities ---
    by_comm = collections.defaultdict(list)
    for d in dids:
        by_comm[comm[d]].append(idx[d])
    big = sorted(by_comm.items(), key=lambda kv: len(kv[1]), reverse=True)[:6]
    print("\nCOMMUNITY TOPICS (top TF-IDF words)")
    for cid, rows in big:
        mean_tfidf = np.asarray(X[rows].mean(axis=0)).ravel()
        top = vocab[mean_tfidf.argsort()[::-1][:10]]
        print(f"  community {cid} (n={len(rows)}): {', '.join(top)}")

    # --- 4a. homophily: same community vs different communities ---
    NP = 5000
    same, diff = [], []
    pool = dids
    while len(same) < NP or len(diff) < NP:
        u, v = random.choice(pool), random.choice(pool)
        if u == v:
            continue
        s = float(X[idx[u]].multiply(X[idx[v]]).sum())
        if comm[u] == comm[v] and len(same) < NP:
            same.append(s)
        elif comm[u] != comm[v] and len(diff) < NP:
            diff.append(s)
    print("\nTOPICAL HOMOPHILY (mean cosine similarity)")
    print(f"  SAME-community pairs:      {np.mean(same):.4f}")
    print(f"  DIFFERENT-community pairs: {np.mean(diff):.4f}")
    print(f"  ratio:                     {np.mean(same)/max(np.mean(diff),1e-9):.2f}x")

    # --- 4b. homophily on edges: real edges vs non-edges ---
    edges = [(u, v) for u, v in G.edges() if u in idx and v in idx]
    real = random.sample(edges, min(NP, len(edges)))
    non = []
    while len(non) < len(real):
        u, v = random.choice(pool), random.choice(pool)
        if u != v and not G.has_edge(u, v):
            non.append((u, v))
    sim_real = cosine_pairs(X, idx, real)
    sim_non = cosine_pairs(X, idx, non)
    print(f"\n  endpoints of REAL edges:   {np.mean(sim_real):.4f}")
    print(f"  endpoints of NON-edges:    {np.mean(sim_non):.4f}")
    print(f"  ratio:                     {np.mean(sim_real)/max(np.mean(sim_non),1e-9):.2f}x")

    # --- statistical significance (Mann-Whitney U, one-sided) ---
    from scipy.stats import mannwhitneyu
    _, p_comm = mannwhitneyu(same, diff, alternative="greater")
    _, p_edge = mannwhitneyu(sim_real, sim_non, alternative="greater")
    print("\nSIGNIFICANCE (Mann-Whitney U, one-sided)")
    print(f"  same > different community:  p = {p_comm:.2e}")
    print(f"  real edges > non-edges:      p = {p_edge:.2e}")

    # --- violin plot of the four similarity distributions ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.violinplot([same, diff, sim_real, sim_non], showmeans=True, showextrema=False)
    plt.xticks([1, 2, 3, 4],
               ["same\ncommunity", "different\ncommunity", "real\nedges", "non-\nedges"])
    plt.ylabel("cosine similarity of topic vectors")
    plt.title("Topical homophily: pairwise post-content similarity")
    plt.tight_layout()
    out = os.path.join(fig_dir, "open_question_homophily.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[oq] plot -> {out}")

    print("\n[oq] Interpretation (to write in the report): if 'same community' "
          "and 'real edges' show markedly higher similarity, the follow network "
          "aligns structure and topics => evidence of topical echo chambers in "
          "the Bluesky scientific community.")


if __name__ == "__main__":
    main()
