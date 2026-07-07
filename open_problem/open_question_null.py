"""
PART 4 - Open question: standalone degree-preserving control.

Tests whether the topical homophily of follow edges survives a degree control.
The main analysis (open_question.py) compares the endpoints of real follow
edges against uniform random non-edges. Uniform non-edges under-sample the
hubs, so that gap could in principle reflect degree rather than topic. Here
real edges are compared against a CONFIGURATION-MODEL null: node pairs whose
endpoints are drawn in proportion to their degree, reproducing the degree of
the real edge endpoints while scrambling who links to whom. If real edges stay
more topically similar, the homophily is not a degree artefact.

Self-contained: it re-loads the data and rebuilds the same TF-IDF as
open_question.py, uses its own random generator, and neither imports the main
pipeline's state nor writes any shared file.

Usage:
    python open_question_null.py
"""

import os
import sys
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "data_collection"))
import config
from open_question import load_graph, load_texts, cosine_pairs


def main():
    rng = random.Random(42)
    data_dir = config.DATA_DIR
    G = load_graph(data_dir)
    texts = load_texts(data_dir)

    dids = [d for d in texts if d in G]
    docs = [texts[d] for d in dids]
    idx = {d: i for i, d in enumerate(dids)}

    # TF-IDF: mirror open_question.py exactly so the topic vectors are comparable
    extra = {"bsky", "social", "com", "www", "org", "net", "http", "https",
             "doi", "new", "just", "like", "people", "really", "today", "via",
             "amp", "don", "ve", "ll", "twitter", "post", "thread"}
    stop = list(ENGLISH_STOP_WORDS.union(extra))
    vec = TfidfVectorizer(max_features=3000, stop_words=stop,
                          min_df=5, token_pattern=r"[a-zA-Z]{3,}")
    X = vec.fit_transform(docs)
    print(f"[null] TF-IDF: {X.shape[0]} documents, {X.shape[1]} terms")

    # sample real follow edges among nodes that carry a topic vector
    edges = [(u, v) for u, v in G.edges() if u in idx and v in idx]
    real = rng.sample(edges, min(5000, len(edges)))

    # degree-preserving (configuration-model) null: draw pairs with endpoints
    # proportional to their degree from a stub list (each node repeated deg times)
    stubs = []
    for d in dids:
        stubs.extend([d] * max(G.degree(d), 1))
    null = []
    while len(null) < len(real):
        u, v = rng.choice(stubs), rng.choice(stubs)
        if u != v:
            null.append((u, v))

    sim_real = cosine_pairs(X, idx, real)
    sim_null = cosine_pairs(X, idx, null)
    _, p = mannwhitneyu(sim_real, sim_null, alternative="greater")

    print("\nDEGREE-PRESERVING NULL (real edges vs configuration-model pairs)")
    print(f"  endpoints of REAL edges:      {np.mean(sim_real):.4f}")
    print(f"  degree-matched null pairs:    {np.mean(sim_null):.4f}")
    print(f"  ratio (real / degree-null):   "
          f"{np.mean(sim_real) / max(np.mean(sim_null), 1e-9):.2f}x")
    print(f"  significance (real > null):   p = {p:.2e}")


if __name__ == "__main__":
    main()
