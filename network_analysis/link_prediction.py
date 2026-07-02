"""
PART 3 (ANALYTICAL cluster) - Link Prediction.

Follows Liben-Nowell & Kleinberg, "The link prediction problem for social
networks" (CIKM 2003), required by the assignment:

  - the network is split into TRAIN (80% of edges) and TEST (20% of edges);
  - on the TRAIN graph the classic unsupervised predictors are computed:
      * Common Neighbors
      * Jaccard Coefficient
      * Adamic-Adar
      * Preferential Attachment
  - accuracy is measured AS IN THE PAPER. The candidate pairs (the held-out
    TEST edges as positives, plus many random non-edges as negatives, in a
    strongly imbalanced 1:R setting that mimics the paper's realistic scenario
    of few true links among many candidates) are RANKED by each predictor;
    the top-n pairs are taken (n = number of positives) and the number of
    correct predictions is counted. The headline figure, exactly as reported
    in the paper, is the IMPROVEMENT FACTOR over a random predictor
    (precision@n / base-rate). AUC-ROC is also reported as a complementary
    modern metric.

Usage:
    python link_prediction.py
    python link_prediction.py --pos 2000 --ratio 50
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


def top_n_precision(scores, labels, n):
    """Paper-style accuracy: rank the candidate pairs by decreasing score, take
    the top n, and return how many of them are real (held-out) edges, together
    with the resulting precision@n."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    correct = sum(labels[i] for i in order[:n])
    return correct, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=2000,
                    help="n. positive pairs (held-out test edges) used in the ranking")
    ap.add_argument("--ratio", type=int, default=50,
                    help="n. random negatives per positive (imbalance, as in the paper)")
    args = ap.parse_args()

    data_dir = config.DATA_DIR
    G = load_graph(data_dir)
    print(f"[lp] network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    G_train, test_edges = train_test_split(G, test_frac=0.2)
    print(f"[lp] train: {G_train.number_of_edges()} edges | "
          f"test: {len(test_edges)} edges")

    # Imbalanced candidate set: n positives (held-out test edges) + ratio*n random
    # negatives, mimicking the paper's realistic "few real links among many candidates".
    n_pos = min(args.pos, len(test_edges))
    n_neg = n_pos * args.ratio
    pos = random.sample(test_edges, n_pos)
    neg = sample_negatives(G, n_neg, G.nodes())
    pairs = pos + neg
    labels = [1] * n_pos + [0] * n_neg
    prevalence = n_pos / (n_pos + n_neg)      # a random predictor's precision@n
    print(f"[lp] ranking {n_pos} positives + {n_neg} negatives (1:{args.ratio}); "
          f"random-predictor precision@n = {prevalence:.4f}")

    print(f"\n{'predictor':26s} {'top-n':>7s} {'P@n':>7s} {'x random':>9s} {'AUC':>7s}")
    for name, fn in PREDICTORS.items():
        scores = fn(G_train, pairs)
        correct, prec = top_n_precision(scores, labels, n_pos)
        factor = prec / prevalence
        auc = roc_auc_score(labels, scores)
        print(f"{name:26s} {correct:7d} {prec:7.3f} {factor:8.1f}x {auc:7.3f}")

    print(f"\n[lp] top-n precision = fraction of the {n_pos} top-ranked candidate "
          "pairs that are real held-out edges;")
    print("[lp] 'x random' = improvement factor over a random predictor "
          f"(baseline precision@n = {prevalence:.4f}), as reported in the paper;")
    print("[lp] AUC = prob. that a real edge scores higher than a non-edge "
          "(0.5 = chance).")


if __name__ == "__main__":
    main()
