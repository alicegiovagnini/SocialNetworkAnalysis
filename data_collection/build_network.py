"""
Building the final network and basic statistics

  - builds the directed graph on the induced subgraph of the collected node
    set (keeps only edges u->v with both u and v in the set);
  - attaches node attributes (if available);
  - produces the simple, undirected, unweighted version required for further analysis;
  - exports the edge lists + attribute table
  - prints sanity statistics (nodes, edges, components, giant, density).

Usage:
    python build_network.py
"""

import os
import json
import csv
import gzip

import networkx as nx
import config


def main():
    data_dir = config.DATA_DIR
    with open(os.path.join(data_dir, "nodes.json")) as f:
        nodes = json.load(f)
    node_set = set(nodes.keys())

    #directed graph from the raw dump, only edges internal to the set
    Gd = nx.DiGraph()
    Gd.add_nodes_from(node_set)
    raw = os.path.join(data_dir, "raw_edges.jsonl")
    seen = 0
    with open(raw) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = json.loads(line)
            if u != v and u in node_set and v in node_set:
                Gd.add_edge(u, v)
            seen += 1
    print(f"[build] raw records read={seen}, directed edges={Gd.number_of_edges()}")

    #attributes (handle always; profiles if already downloaded)
    for did, handle in nodes.items():
        if did in Gd:
            Gd.nodes[did]["handle"] = handle
    attr_path = os.path.join(data_dir, "node_attributes.json")
    if os.path.exists(attr_path):
        with open(attr_path) as f:
            attrs = json.load(f)
        for did, a in attrs.items():
            if did in Gd:
                Gd.nodes[did].update(a)
        print(f"[build] attributes attached for {len(attrs)} nodes")
    else:
        print("[build] node_attributes.json not found (run fetch_metadata.py)")

    #simple, undirected, unweighted version
    Gu = Gd.to_undirected()
    Gu.remove_edges_from(nx.selfloop_edges(Gu))

    #export edge lists
    nx.write_edgelist(Gd, os.path.join(data_dir, "network_directed.edgelist"), data=False)
    nx.write_edgelist(Gu, os.path.join(data_dir, "network_undirected.edgelist"), data=False)

    #export attribute table
    cols = ["did", "handle", "displayName", "followersCount",
            "followsCount", "postsCount", "createdAt"]
    with open(os.path.join(data_dir, "node_attributes.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for n in Gu.nodes():
            d = Gu.nodes[n]
            w.writerow([n] + [d.get(c, "") for c in cols[1:]])

    #sanity statistics
    n = Gu.number_of_nodes()
    m = Gu.number_of_edges()
    comps = list(nx.connected_components(Gu))
    gcc = max(comps, key=len) if comps else set()
    print("\nNETWORK SUMMARY (undirected, simple)")
    print(f"nodes:               {n}")
    print(f"edges:               {m}")
    print(f"average degree:      {(2 * m / n):.2f}" if n else "average degree: n/a")
    print(f"density:             {nx.density(Gu):.6f}")
    print(f"connected components:{len(comps)}")
    if n:
        print(f"giant component:     {len(gcc)} nodes ({100 * len(gcc) / n:.1f}%)")
    if n < 10000:
        print("\n[WARNING] fewer than 10k nodes: increase MAX_NODES, add seeds,"
              " or agree on the size with the instructors.")

    #compression for GitHub
    for fname in ["network_undirected.edgelist", "network_directed.edgelist",
                  "node_attributes.csv"]:
        path = os.path.join(data_dir, fname)
        with open(path, "rb") as fin, gzip.open(path + ".gz", "wb") as fout:
            fout.writelines(fin)
    print("\n[build] .gz versions created for the GitHub upload")


if __name__ == "__main__":
    main()
