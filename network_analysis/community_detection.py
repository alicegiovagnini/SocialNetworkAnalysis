"""
COMMUNITY DETECTION (required by the assignment).

Identifies, evaluates and compares the modular structure of the network with
THREE community detection algorithms (CDlib), as required:
  1. Louvain           (modularity optimisation)
  2. Label Propagation  (label propagation)
  3. Infomap           (flows / random walk)

For each it computes:
  - number of communities and size distribution;
  - modularity (Newman-Girvan) and average conductance (internal quality);
and compares the partitions pairwise with NMI and ARI (agreement between
algorithms).

Semantic interpretation: for the largest communities it extracts the most
frequent keywords from the accounts' bios/displayName (simple term frequency),
to give the partitions a thematic meaning. Saves the community assignment to
data/communities.csv (reused by the open question).

Usage:
    python community_detection.py
"""

import os
import csv
import json
import gzip
import collections
import re

import networkx as nx
from cdlib import algorithms, evaluation

import config

# Stopwords filtered out of the bios (English + Italian, plus platform tokens).
STOPWORDS = set("""
the a an and or of to in for on with at by from is are be this that it as we i
you he she they them our your my his her their its not no all can will more
una un uno di da il lo la le i gli del della delle dei e che per con su tra fra
non come piu sono ho hai ha siamo are http https www com bsky social org net
""".split())


def load_graph(data_dir):
    plain = os.path.join(data_dir, "network_undirected.edgelist")
    if os.path.exists(plain):
        return nx.read_edgelist(plain)
    with gzip.open(plain + ".gz", "rt") as f:
        return nx.parse_edgelist(f)


def load_attrs(data_dir):
    p = os.path.join(data_dir, "node_attributes.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def keywords_for_community(nodes, attrs, top=8):
    """Most frequent words in the bios/displayName of the community's accounts."""
    counter = collections.Counter()
    for did in nodes:
        a = attrs.get(did, {})
        text = f"{a.get('displayName','')} {a.get('description','')}".lower()
        for w in re.findall(r"[a-zA-Zàèéìòù]{3,}", text):
            if w not in STOPWORDS:
                counter[w] += 1
    return [w for w, _ in counter.most_common(top)]


def summarize(name, clustering, G, attrs):
    comms = clustering.communities
    sizes = sorted((len(c) for c in comms), reverse=True)
    try:
        mod = clustering.newman_girvan_modularity().score
    except Exception:
        mod = float("nan")
    try:
        cond = evaluation.conductance(G, clustering).score
    except Exception:
        cond = float("nan")
    print(f"\n{name}")
    print(f"n. communities:      {len(comms)}")
    print(f"top-10 sizes:        {sizes[:10]}")
    print(f"modularity:          {mod:.4f}")
    print(f"average conductance: {cond:.4f}")
    # interpretation: keywords of the 5 largest communities
    big = sorted(comms, key=len, reverse=True)[:5]
    print("semantic interpretation (keywords from the bios):")
    for i, c in enumerate(big, 1):
        kw = keywords_for_community(c, attrs)
        print(f"  C{i} (n={len(c)}): {', '.join(kw)}")
    return {"name": name, "n_comms": len(comms), "modularity": mod,
            "conductance": cond, "clustering": clustering}


def main():
    data_dir = config.DATA_DIR
    G = load_graph(data_dir)
    attrs = load_attrs(data_dir)
    print(f"[cd] network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    results = []

    print("\n[cd] Louvain...")
    results.append(summarize("LOUVAIN", algorithms.louvain(G), G, attrs))

    print("\n[cd] Label Propagation...")
    results.append(summarize("LABEL PROPAGATION",
                             algorithms.label_propagation(G), G, attrs))

    print("\n[cd] Infomap... (may take a few minutes)")
    try:
        results.append(summarize("INFOMAP", algorithms.infomap(G), G, attrs))
    except Exception as e:
        print(f"[cd] infomap not available ({e}); using greedy modularity.")
        results.append(summarize("GREEDY MODULARITY",
                                 algorithms.greedy_modularity(G), G, attrs))

    # --- pairwise comparison between the partitions ---
    print("\nPARTITION COMPARISON")
    print(f"{'pair':36s} {'NMI':>7s} {'ARI':>7s}")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            try:
                nmi = evaluation.normalized_mutual_information(
                    a["clustering"], b["clustering"]).score
            except Exception:
                nmi = float("nan")
            try:
                ari = evaluation.adjusted_rand_index(
                    a["clustering"], b["clustering"]).score
            except Exception:
                ari = float("nan")
            print(f"{a['name']:>16s} vs {b['name']:<16s} {nmi:7.3f} {ari:7.3f}")

    # --- summary table ---
    print("\nSUMMARY")
    print(f"{'algorithm':20s} {'#comm':>7s} {'modularity':>12s} {'conductance':>12s}")
    for r in results:
        print(f"{r['name']:20s} {r['n_comms']:7d} {r['modularity']:12.4f} "
              f"{r['conductance']:12.4f}")

    # --- save the Louvain assignment (reused by the open question) ---
    louvain = results[0]["clustering"]
    node2comm = {}
    for cid, comm in enumerate(louvain.communities):
        for n in comm:
            node2comm[n] = cid
    out = os.path.join(data_dir, "communities.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["did", "louvain_community"])
        for n in G.nodes():
            w.writerow([n, node2comm.get(n, -1)])
    print(f"\n[cd] community assignment (Louvain) -> {out}")


if __name__ == "__main__":
    main()
