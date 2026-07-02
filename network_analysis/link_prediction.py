"""
PART 3 (ANALYTICAL cluster) - Link Prediction.

Follows the setup of Liben-Nowell & Kleinberg, "The link prediction problem
for social networks" (CIKM 2003), required by the assignment:

  - the network is split into TRAIN (80% of edges) and TEST (20% of edges);
  - on the train graph the classic unsupervised predictors are computed:
      * Common Neighbors
      * Jaccard Coefficient
      * Adamic-Adar
      * Preferential Attachment
  - each predictor's ability to "recover" the test edges is evaluated
    against a sample of non-edges (negatives), via AUC-ROC and precision@k.

To stay tractable on a network with ~1.6M edges, the evaluation uses a
balanced sample of positive pairs (test edges) and negative pairs (random
non-edges). The sample size is configurable.

Usage:
    python link_prediction.py
    python link_prediction.py --sample 20000
"""

import os
import gzip
import random
import argparse

import networkx as nx
from sklearn.metrics import roc_auc_score

import config

random.seed(42)


def load_graph(data_dir):
    plain = os.path.join(data_dir, "network_undirected.edgelist")
    if os.path.exists(plain):
        return nx.read_edgelist(plain)
    with gzip.open(plain + ".gz", "rt") as f:
        return nx.parse_edgelist(f)


def train_test_split(G, test_frac=0.2):
    edges = list(G.edges())
    random.shuffle(edges)
    n_test = int(len(edges) * test_frac)
    test_edges = edges[:n_test]
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())          # all nodes are kept
    G_train.add_edges_from(edges[n_test:])     # only the train edges
    return G_train, test_edges


def sample_negatives(G, n, nodes):
    """n pairs of nodes NOT connected in the original graph."""
    negs = set()
    nodes = list(nodes)
    while len(negs) < n:
        u, v = random.choice(nodes), random.choice(nodes)
        if u != v and not G.has_edge(u, v) and (u, v) not in negs \
                and (v, u) not in negs:
            negs.add((u, v))
    return list(negs)


# ---- predictors (work on the TRAIN graph) ----
def common_neighbors(G, pairs):
    return [len(list(nx.common_neighbors(G, u, v))) for u, v in pairs]

def jaccard(G, pairs):
    return [p for _, _, p in nx.jaccard_coefficient(G, pairs)]

def adamic_adar(G, pairs):
    return [p for _, _, p in nx.adamic_adar_index(G, pairs)]

def pref_attachment(G, pairs):
    return [p for _, _, p in nx.preferential_attachment(G, pairs)]


PREDICTORS = {
    "Common Neighbors": common_neighbors,
    "Jaccard": jaccard,
    "Adamic-Adar": adamic_adar,
    "Preferential Attachment": pref_attachment,
}


def precision_at_k(scores, labels, k):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    topk = order[:k]
    return sum(labels[i] for i in topk) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000,
                    help="n. positive pairs (and as many negatives) for the evaluation")
    args = ap.parse_args()

    data_dir = config.DATA_DIR
    G = load_graph(data_dir)
    print(f"[lp] network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    G_train, test_edges = train_test_split(G, test_frac=0.2)
    print(f"[lp] train: {G_train.number_of_edges()} edges | "
          f"test: {len(test_edges)} edges")

    # balanced positive/negative sample
    n = min(args.sample, len(test_edges))
    pos = random.sample(test_edges, n)
    neg = sample_negatives(G, n, G.nodes())
    pairs = pos + neg
    labels = [1] * n + [0] * n
    print(f"[lp] evaluation on {n} positives + {n} negatives")

    k = max(1, n // 10)   # precision@k with k = 10% of the positives
    print(f"\n{'predictor':26s} {'AUC':>7s} {'P@'+str(k):>10s}")
    for name, fn in PREDICTORS.items():
        scores = fn(G_train, pairs)
        auc = roc_auc_score(labels, scores)
        pk = precision_at_k(scores, labels, k)
        print(f"{name:26s} {auc:7.3f} {pk:10.3f}")

    print("\n[lp] AUC = prob. that a real edge scores higher than a non-edge "
          "(0.5 = chance). In the report, discuss which predictor wins and why.")


if __name__ == "__main__":
    main()
