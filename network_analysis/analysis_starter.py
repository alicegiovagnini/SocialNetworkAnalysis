"""
Network analysis

Loads the simple/undirected/unweighted network produced by build_network.py and computes the statistics, comparing them with synthetic ER (Erdos-Renyi) and BA (Barabasi-Albert) graphs with (almost) the
same number of nodes and edges.

Covers:
  - basic statistics (nodes, edges, density, average degree);
  - connected components and giant component;
  - degree distribution in log-log;
  - clustering coefficient and transitivity;
  - path analysis (average length, diameter estimate) on a sample;
  - centrality analysis (degree, sampled betweenness, closeness,
    eigenvector/pagerank);
  - comparison with ER and BA graphs.

Usage:
    python analysis_starter.py
    python analysis_starter.py --betw-k 800   # sample for the betweenness

"""

import os
import csv
import gzip
import argparse
import random

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: save to file
import matplotlib.pyplot as plt

import config

random.seed(42)
np.random.seed(42)

# Loading
def load_graph(data_dir):
    """Load the undirected network, preferring the uncompressed file."""
    plain = os.path.join(data_dir, "network_undirected.edgelist")
    gz = plain + ".gz"
    if os.path.exists(plain):
        G = nx.read_edgelist(plain)
    elif os.path.exists(gz):
        with gzip.open(gz, "rt") as f:
            G = nx.parse_edgelist(f)
    else:
        raise FileNotFoundError(
            "network_undirected.edgelist(.gz) not found: "
            "run build_network.py first"
        )
    # safety: simple, undirected, no self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def load_attributes(data_dir):
    """Node attributes (did -> dict). Returns {} if absent."""
    path = os.path.join(data_dir, "node_attributes.csv")
    gzp = path + ".gz"
    opener = None
    if os.path.exists(path):
        opener = lambda: open(path, newline="", encoding="utf-8")
    elif os.path.exists(gzp):
        opener = lambda: gzip.open(gzp, "rt", newline="", encoding="utf-8")
    if opener is None:
        return {}
    out = {}
    with opener() as f:
        for row in csv.DictReader(f):
            out[row["did"]] = row
    return out


# Statistics
def basic_stats(G, label="network"):
    n, m = G.number_of_nodes(), G.number_of_edges()
    degs = [d for _, d in G.degree()]
    print(f"\n{label}")
    print(f"nodes:           {n}")
    print(f"edges:           {m}")
    print(f"average degree:  {2 * m / n:.3f}" if n else "average degree: n/a")
    print(f"max degree:      {max(degs) if degs else 0}")
    print(f"density:         {nx.density(G):.6f}")
    return {"n": n, "m": m, "degs": degs}


def components_stats(G):
    comps = list(nx.connected_components(G))
    gcc_nodes = max(comps, key=len) if comps else set()
    gcc = G.subgraph(gcc_nodes).copy()
    print(f"connected components: {len(comps)}")
    print(f"giant component:      {gcc.number_of_nodes()} nodes "
          f"({100 * gcc.number_of_nodes() / G.number_of_nodes():.1f}%), "
          f"{gcc.number_of_edges()} edges")
    return gcc


def clustering_stats(G, label="network"):
    avg_c = nx.average_clustering(G)
    trans = nx.transitivity(G)
    print(f"[{label}] average clustering: {avg_c:.4f} | transitivity: {trans:.4f}")
    return avg_c, trans


def assortativity_stats(G, data_dir):
    """Degree assortativity and, if communities.csv is available, the
    community-based mixing (attribute assortativity + intra-community edges)."""
    r = nx.degree_assortativity_coefficient(G)
    print(f"degree assortativity coefficient: {r:.4f}")
    cpath = os.path.join(data_dir, "communities.csv")
    if os.path.exists(cpath):
        comm = {}
        with open(cpath) as f:
            for row in csv.DictReader(f):
                comm[row["did"]] = row["louvain_community"]
        nx.set_node_attributes(G, comm, "community")
        ra = nx.attribute_assortativity_coefficient(G, "community")
        intra = sum(1 for u, v in G.edges() if comm.get(u) == comm.get(v))
        print(f"community attribute assortativity: {ra:.4f}")
        print(f"intra-community edges: {100 * intra / G.number_of_edges():.1f}%")
        return r, ra
    return r, None


def path_stats(gcc, n_samples=500):
    """Estimate average path length and diameter via BFS sampling
    (the exact computation is too expensive on tens of thousands of nodes)."""
    nodes = list(gcc.nodes())
    n_samples = min(n_samples, len(nodes))
    sample = random.sample(nodes, n_samples)
    tot, cnt, ecc_max = 0, 0, 0
    for s in sample:
        lengths = nx.single_source_shortest_path_length(gcc, s)
        if len(lengths) > 1:
            d = list(lengths.values())
            tot += sum(d)
            cnt += len(d) - 1   # exclude the distance from itself (0)
            ecc_max = max(ecc_max, max(d))
    avg_path = tot / cnt if cnt else float("nan")
    print(f"average path length (estimate, {n_samples} sources): {avg_path:.3f}")
    print(f"diameter (lower bound from the sample):              {ecc_max}")
    return avg_path, ecc_max


def centrality_stats(gcc, betw_k=500, top=10, attrs=None):
    """Main centralities on the giant component.
    Betweenness is sampled (k sources) to stay tractable."""
    print("\ncentrality (giant component)")
    deg = nx.degree_centrality(gcc)
    k = min(betw_k, gcc.number_of_nodes())
    print(f"sampled betweenness over k={k} sources...")
    betw = nx.betweenness_centrality(gcc, k=k, seed=42)
    pr = nx.pagerank(gcc)
    try:
        eig = nx.eigenvector_centrality(gcc, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("  (eigenvector did not converge: using pagerank as a proxy)")
        eig = pr

    def name(did):
        if attrs and did in attrs:
            h = attrs[did].get("handle") or ""
            return h or did[:18]
        return did[:18]

    for title, c in [("DEGREE", deg), ("BETWEENNESS", betw),
                     ("PAGERANK", pr), ("EIGENVECTOR", eig)]:
        topn = sorted(c.items(), key=lambda kv: kv[1], reverse=True)[:top]
        print(f"\nTop {top} by {title}:")
        for did, val in topn:
            print(f"  {name(did):32s} {val:.5f}")
    return {"degree": deg, "betweenness": betw, "pagerank": pr, "eigenvector": eig}


# Synthetic comparison graphs (ER and BA)
def make_reference_graphs(n, m):
    print("\ngenerating comparison graphs")
    er = nx.gnm_random_graph(n, m, seed=42)
    m_ba = max(1, round(m / n))   # edges per new node -> ~ n*m_ba total edges
    ba = nx.barabasi_albert_graph(n, m_ba, seed=42)
    print(f"ER: n={er.number_of_nodes()} m={er.number_of_edges()}")
    print(f"BA: n={ba.number_of_nodes()} m={ba.number_of_edges()} (m per node={m_ba})")
    return er, ba


# Figures
def plot_degree_distribution(graphs_degs, out_path):
    """graphs_degs: list of (label, degree_list). CCDF in log-log."""
    plt.figure(figsize=(7, 5))
    for label, degs in graphs_degs:
        degs = np.array([d for d in degs if d > 0])
        if degs.size == 0:
            continue
        vals, counts = np.unique(degs, return_counts=True)
        ccdf = 1.0 - np.cumsum(counts) / counts.sum() + counts / counts.sum()
        plt.loglog(vals, ccdf, marker=".", linestyle="none", label=label, alpha=0.7)
    plt.xlabel("degree k")
    plt.ylabel("P(K >= k)  (CCDF)")
    plt.title("Degree distribution (log-log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] degree distribution -> {out_path}")


def plot_degree_hist(degs, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(degs, bins=60)
    plt.xlabel("degree")
    plt.ylabel("number of nodes")
    plt.title("Degree histogram (real network)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] degree histogram -> {out_path}")


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betw-k", type=int, default=500,
                    help="number of sources for the sampled betweenness")
    ap.add_argument("--path-samples", type=int, default=500,
                    help="number of sources for the path-length estimate")
    args = ap.parse_args()

    data_dir = config.DATA_DIR
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("LOADING")
    G = load_graph(data_dir)
    attrs = load_attributes(data_dir)
    if attrs:
        print(f"attributes loaded for {len(attrs)} nodes")

    print("\nBASIC STATISTICS")
    real = basic_stats(G, "real network (Bluesky)")
    gcc = components_stats(G)

    print("\nCLUSTERING")
    clustering_stats(G, "real network")

    print("\nASSORTATIVITY / MIXING")
    assortativity_stats(G, data_dir)

    print("\nPATHS (giant component)")
    path_stats(gcc, n_samples=args.path_samples)

    centrality_stats(gcc, betw_k=args.betw_k, attrs=attrs)

    # comparison with ER and BA
    print("\nCOMPARISON WITH SYNTHETIC GRAPHS")
    er, ba = make_reference_graphs(real["n"], real["m"])
    clustering_stats(er, "ER")
    clustering_stats(ba, "BA")

    # figures
    print("\nFIGURES")
    plot_degree_hist(real["degs"], os.path.join(fig_dir, "degree_hist.png"))
    plot_degree_distribution(
        [("Bluesky", real["degs"]),
         ("ER", [d for _, d in er.degree()]),
         ("BA", [d for _, d in ba.degree()])],
        os.path.join(fig_dir, "degree_ccdf_compare.png"),
    )

    print("\n[OK] basic analysis complete")


if __name__ == "__main__":
    main()
