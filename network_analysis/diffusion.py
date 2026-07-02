"""
PART 3 (MANIPULATION cluster) - Diffusion (NDlib).

Simulates the diffusion models seen in class - SI, SIS, SIR and Threshold -
as required by the assignment, BOTH on the collected data AND on synthetic
graphs (ER and BA) with (almost) the same number of nodes/edges. Analyses the
results varying the model parameters and the initial conditions (infection seeds).

Produces in data/figures/:
  - diffusion_models_real.png   : SI/SIS/SIR/Threshold on the real network;
  - diffusion_sir_compare.png   : SIR on real network vs ER vs BA;
  - diffusion_sir_beta.png      : SIR varying beta (parameter);
  - diffusion_sir_seeds.png     : SIR varying the seed fraction.

Usage:
    python diffusion.py
"""

import os
import gzip

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ndlib.models.epidemics as ep
from ndlib.models.ModelConfig import Configuration

import config

ITERS = 100


def load_graph(data_dir):
    plain = os.path.join(data_dir, "network_undirected.edgelist")
    if os.path.exists(plain):
        return nx.read_edgelist(plain)
    with gzip.open(plain + ".gz", "rt") as f:
        return nx.parse_edgelist(f)


def infected_series(model, iters=ITERS):
    """Fraction of infected nodes (status=1) over time."""
    N = model.graph.number_of_nodes()
    it = model.iteration_bunch(iters)
    trends = model.build_trends(it)
    inf = trends[0]["trends"]["node_count"][1]
    return [x / N for x in inf]


def run_si(G, beta=0.01, frac=0.05):
    m = ep.SIModel(G)
    c = Configuration()
    c.add_model_parameter("beta", beta)
    c.add_model_parameter("fraction_infected", frac)
    m.set_initial_status(c)
    return infected_series(m)


def run_sis(G, beta=0.01, lmbda=0.005, frac=0.05):
    m = ep.SISModel(G)
    c = Configuration()
    c.add_model_parameter("beta", beta)
    c.add_model_parameter("lambda", lmbda)
    c.add_model_parameter("fraction_infected", frac)
    m.set_initial_status(c)
    return infected_series(m)


def run_sir(G, beta=0.01, gamma=0.01, frac=0.05):
    m = ep.SIRModel(G)
    c = Configuration()
    c.add_model_parameter("beta", beta)
    c.add_model_parameter("gamma", gamma)
    c.add_model_parameter("fraction_infected", frac)
    m.set_initial_status(c)
    return infected_series(m)


def run_threshold(G, threshold=0.1, frac=0.05):
    m = ep.ThresholdModel(G)
    c = Configuration()
    for n in G.nodes():
        c.add_node_configuration("threshold", n, threshold)
    c.add_model_parameter("fraction_infected", frac)
    m.set_initial_status(c)
    return infected_series(m)


def lineplot(series_dict, title, path, ylabel="infected fraction"):
    plt.figure(figsize=(7, 5))
    for label, ser in series_dict.items():
        plt.plot(ser, label=label)
    plt.xlabel("iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plot] {path}")


def main():
    data_dir = config.DATA_DIR
    fig = os.path.join(data_dir, "figures")
    os.makedirs(fig, exist_ok=True)

    G = load_graph(data_dir)
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"[diff] real network: {n} nodes, {m} edges")

    print("[diff] generating ER and BA for comparison...")
    ER = nx.gnm_random_graph(n, m, seed=42)
    BA = nx.barabasi_albert_graph(n, max(1, round(m / n)), seed=42)

    # 1) the four models on the real network
    print("[diff] SI/SIS/SIR/Threshold on the real network...")
    lineplot({
        "SI": run_si(G),
        "SIS": run_sis(G),
        "SIR": run_sir(G),
        "Threshold": run_threshold(G),
    }, "Diffusion models - real network (Bluesky)",
       os.path.join(fig, "diffusion_models_real.png"))

    # 2) SIR: real vs ER vs BA
    print("[diff] SIR on real vs ER vs BA...")
    lineplot({
        "Bluesky": run_sir(G),
        "ER": run_sir(ER),
        "BA": run_sir(BA),
    }, "SIR - real network vs synthetic graphs",
       os.path.join(fig, "diffusion_sir_compare.png"))

    # 3) SIR varying beta (model parameter)
    print("[diff] SIR varying beta...")
    lineplot({f"beta={b}": run_sir(G, beta=b)
              for b in (0.002, 0.005, 0.01, 0.02)},
             "SIR (real network) - effect of the infection rate beta",
             os.path.join(fig, "diffusion_sir_beta.png"))

    # 4) SIR varying the seed fraction (initial condition)
    print("[diff] SIR varying the initial seeds...")
    lineplot({f"seed={f:g}": run_sir(G, frac=f)
              for f in (0.01, 0.05, 0.1, 0.2)},
             "SIR (real network) - effect of the initial seed fraction",
             os.path.join(fig, "diffusion_sir_seeds.png"))

    print("\n[diff] done. In the report, discuss: does the real network diffuse "
          "faster than ER (because of hubs and high clustering)? How do peak "
          "and speed change with beta and with the seeds?")


if __name__ == "__main__":
    main()
