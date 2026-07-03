"""
PART 4 - Open question.

Research question:
  "Is the Bluesky scientific community organised into topical ECHO CHAMBERS?
   (1) Do the communities found from the follow structure correspond to
       coherent discussion topics?
   (2) Is there TOPICAL HOMOPHILY (do connected accounts / accounts in the
       same community talk about similar things)?
   (3) How does that homophily evolve across account generations (tracking)?"

Combines SNA tools (community detection) with a textual analysis (TF-IDF) of
the posts, exploiting the additional information collected during crawling. It
links partition quality to an external data factor (topical content).

Pipeline:
  1. group the posts by author -> one "document" per node;
  2. vectorise with TF-IDF;
  3. label each (Louvain) community by its top TF-IDF terms;
  4. QUANTIFY structure<->topic alignment: cluster the documents into topical
     groups (KMeans) and compare that topical partition with the Louvain one
     via NMI and ARI; rank communities by internal topical cohesion;
  5. measure TOPICAL HOMOPHILY: mean cosine similarity same vs different
     community, and real edges vs non-edges (with a significance test);
  6. TRACK the echo chamber over time: topical homophily of the follow edges
     internal to each account-creation cohort (2023 / 2024 / 2025).

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
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

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


def load_created(data_dir):
    """did -> account creation YEAR (int) (external temporal information)."""
    out = {}
    with open(os.path.join(data_dir, "node_attributes.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = r.get("createdAt", "")
            if len(c) >= 4 and c[:4].isdigit():
                out[r["did"]] = int(c[:4])
    return out


def cosine_pairs(X, idx, pairs):
    """Cosine similarity for a list of (did, did) pairs (X is L2-normalised)."""
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
    extra = {"bsky", "social", "com", "www", "org", "net", "http", "https",
             "doi", "new", "just", "like", "people", "really", "today", "via",
             "amp", "don", "ve", "ll", "twitter", "post", "thread"}
    stop = list(ENGLISH_STOP_WORDS.union(extra))
    vec = TfidfVectorizer(max_features=3000, stop_words=stop,
                          min_df=5, token_pattern=r"[a-zA-Z]{3,}")
    X = vec.fit_transform(docs)          # already L2-normalised (norm='l2')
    vocab = np.array(vec.get_feature_names_out())
    print(f"[oq] TF-IDF: {X.shape[0]} documents, {X.shape[1]} terms")

    # --- keywords for the largest communities ---
    by_comm = collections.defaultdict(list)
    for d in dids:
        by_comm[comm[d]].append(idx[d])
    big = sorted(by_comm.items(), key=lambda kv: len(kv[1]), reverse=True)[:6]
    topic_kw = {}
    print("\nCOMMUNITY TOPICS (top TF-IDF words)")
    for cid, rows in big:
        mean_tfidf = np.asarray(X[rows].mean(axis=0)).ravel()
        top = vocab[mean_tfidf.argsort()[::-1][:10]]
        topic_kw[cid] = ", ".join(top[:3])
        print(f"  community {cid} (n={len(rows)}): {', '.join(top)}")

    # --- (1) STRUCTURE vs TOPIC alignment: is a follow-community a topic? ---
    # Cluster the documents into topical groups from the TEXT only and compare
    # that partition with the (structural) Louvain partition via NMI / ARI.
    louvain_lab = [comm[d] for d in dids]
    k_topics = len(set(louvain_lab))              # same granularity as Louvain
    topic_lab = KMeans(n_clusters=k_topics, random_state=42,
                       n_init=10).fit_predict(X)
    nmi = normalized_mutual_info_score(louvain_lab, topic_lab)
    ari = adjusted_rand_score(louvain_lab, topic_lab)
    print("\nSTRUCTURE vs TOPIC ALIGNMENT (follow-communities vs text-topics)")
    print(f"  KMeans topical clusters: {k_topics}")
    print(f"  NMI = {nmi:.3f}   ARI = {ari:.3f}   (0 = independent, 1 = identical)")

    # --- (1b) echo-chamber STRENGTH per community (intra-topical cohesion) ---
    print("\nECHO-CHAMBER STRENGTH per community (mean intra-community similarity)")
    cohesion = {}
    for cid, rows in big:
        s = []
        for _ in range(3000):
            a, b = random.choice(rows), random.choice(rows)
            if a != b:
                s.append(float(X[a].multiply(X[b]).sum()))
        cohesion[cid] = float(np.mean(s))
    for cid in sorted(cohesion, key=cohesion.get, reverse=True):
        print(f"  community {cid} ({topic_kw.get(cid, '')}): {cohesion[cid]:.4f}")

    # --- (2) topical homophily: same vs different community ---
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

    # --- homophily on edges: real edges vs non-edges ---
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

    # --- (3) TRACK the echo chamber over account-creation cohorts ---
    # For each generation of accounts, how topically homophilous are the follow
    # edges internal to it (edge similarity / random-pair similarity)?
    created = load_created(data_dir)
    cohort_years = [2023, 2024, 2025]
    cnodes = {y: [d for d in dids if created.get(d) == y] for y in cohort_years}
    cedges = {y: [] for y in cohort_years}
    for u, v in edges:                             # edges already restricted to text-nodes
        cu, cv = created.get(u), created.get(v)
        if cu == cv and cu in cedges:
            cedges[cu].append((u, v))
    print("\nECHO CHAMBER OVER TIME (topical homophily of intra-cohort follow edges)")
    coh_ratio = {}
    for y in cohort_years:
        ce = cedges[y]
        if len(ce) < 200 or len(cnodes[y]) < 50:
            continue
        e_s = cosine_pairs(X, idx, random.sample(ce, min(NP, len(ce))))
        r_pairs = []
        while len(r_pairs) < len(e_s):
            a, b = random.choice(cnodes[y]), random.choice(cnodes[y])
            if a != b:
                r_pairs.append((a, b))
        r_s = cosine_pairs(X, idx, r_pairs)
        ratio = np.mean(e_s) / max(np.mean(r_s), 1e-9)
        coh_ratio[y] = ratio
        print(f"  cohort {y} (n={len(cnodes[y])}, intra-edges={len(ce)}): "
              f"edge sim={np.mean(e_s):.4f} vs random={np.mean(r_s):.4f} "
              f"-> homophily ratio={ratio:.2f}x")

    # --- figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # violin plot of the four similarity distributions
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

    # temporal tracking: homophily ratio per creation cohort
    if coh_ratio:
        ys = sorted(coh_ratio)
        plt.figure(figsize=(7, 5))
        plt.bar([str(y) for y in ys], [coh_ratio[y] for y in ys], color="teal")
        plt.axhline(1.0, color="grey", ls="--", lw=1, label="no homophily (ratio = 1)")
        plt.ylabel("intra-cohort topical homophily  (edges / random pairs)")
        plt.xlabel("account creation cohort")
        plt.title("Tracking topical echo chambers across account generations")
        plt.legend()
        plt.tight_layout()
        out2 = os.path.join(fig_dir, "open_question_temporal.png")
        plt.savefig(out2, dpi=150)
        plt.close()
        print(f"[oq] plot -> {out2}")

    print("\n[oq] Interpretation: if 'same community' and 'real edges' show "
          "markedly higher similarity, the follow network aligns structure and "
          "topics => evidence of topical echo chambers in the Bluesky scientific "
          "community.")


if __name__ == "__main__":
    main()
