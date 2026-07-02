"""
GEPHI EXPORT - prepare the network for visual analysis in Gephi.

Writes two GEXF files (Gephi's native format), with node attributes ready for
visualisation:
  - community : Louvain community id   -> use it to COLOUR the nodes (Partition);
  - degree    : node degree            -> use it to SIZE the nodes (Ranking);
  - handle    : account handle         -> node label.

Outputs (in the data directory):
  - network_full.gexf : the whole 15k-node network (faithful but heavy);
  - network_viz.gexf  : a community-stratified sample (default ~2000 nodes),
                        much lighter and cleaner for a ForceAtlas2 layout.

Usage:
    python export_gephi.py
    python export_gephi.py --sample 3000 --data-dir ../data_collection/data

Suggested Gephi workflow (network_viz.gexf):
  1. File > Open the .gexf;
  2. Layout > ForceAtlas2 (enable "Prevent Overlap", "LinLog mode"), run it;
  3. Appearance > Nodes > Colour > Partition > community;
  4. Appearance > Nodes > Size  > Ranking  > degree;
  5. (optional) show a few labels for the highest-degree hubs;
  6. Preview > export to PNG/PDF.
"""

import os
import csv
import gzip
import random
import argparse
import collections

import networkx as nx

random.seed(42)


def resolve_data_dir(cli):
    if cli:
        return cli
    try:
        import config
        return config.DATA_DIR
    except Exception:
        for cand in ("data", "../data_collection/data", "../data"):
            if os.path.isdir(cand):
                return cand
        return "data"


def load_graph(data_dir):
    plain = os.path.join(data_dir, "network_undirected.edgelist")
    if os.path.exists(plain):
        return nx.read_edgelist(plain)
    with gzip.open(plain + ".gz", "rt") as f:
        return nx.parse_edgelist(f)


def load_communities(data_dir):
    comm = {}
    with open(os.path.join(data_dir, "communities.csv")) as f:
        for r in csv.DictReader(f):
            comm[r["did"]] = int(r["louvain_community"])
    return comm


def load_handles(data_dir):
    handles = {}
    path = os.path.join(data_dir, "node_attributes.csv")
    gzp = path + ".gz"
    if os.path.exists(path):
        fh = open(path, newline="", encoding="utf-8")
    elif os.path.exists(gzp):
        fh = gzip.open(gzp, "rt", newline="", encoding="utf-8")
    else:
        return handles
    for r in csv.DictReader(fh):
        handles[r["did"]] = r.get("handle", "")
    fh.close()
    return handles


def annotate(G, comm, handles):
    """Attach community / degree / handle / label attributes to the nodes."""
    deg = dict(G.degree())
    for n in G.nodes():
        G.nodes[n]["community"] = comm.get(n, -1)
        G.nodes[n]["degree"] = deg[n]
        h = handles.get(n, "")
        G.nodes[n]["handle"] = h
        G.nodes[n]["label"] = h or n[:18]   # Gephi uses 'label' for node labels


def stratified_sample(G, comm, target):
    """Sample ~target nodes keeping each community's proportion, then take the
    induced subgraph and drop isolated nodes."""
    by_comm = collections.defaultdict(list)
    for n in G.nodes():
        by_comm[comm.get(n, -1)].append(n)
    N = G.number_of_nodes()
    keep = []
    for c, members in by_comm.items():
        k = max(1, round(target * len(members) / N))
        keep += random.sample(members, min(k, len(members)))
    H = G.subgraph(keep).copy()
    H.remove_nodes_from(list(nx.isolates(H)))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000,
                    help="approx. number of nodes for the lighter viz export")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print(f"[gephi] data dir: {data_dir}")
    G = load_graph(data_dir)
    comm = load_communities(data_dir)
    handles = load_handles(data_dir)
    print(f"[gephi] network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"{len(set(comm.values()))} communities")

    annotate(G, comm, handles)

    full = os.path.join(data_dir, "network_full.gexf")
    nx.write_gexf(G, full)
    print(f"[gephi] full network -> {full}")

    H = stratified_sample(G, comm, args.sample)
    annotate(H, comm, handles)
    viz = os.path.join(data_dir, "network_viz.gexf")
    nx.write_gexf(H, viz)
    print(f"[gephi] sampled network ({H.number_of_nodes()} nodes, "
          f"{H.number_of_edges()} edges) -> {viz}")
    print("\n[gephi] open network_viz.gexf in Gephi; ForceAtlas2 layout, then "
          "colour by 'community' and size by 'degree'. See the file header for steps.")


if __name__ == "__main__":
    main()
