"""
PART 3 (extension) - Game-theoretic cascade model (decision-based).

Implements the model seen in class (Class 13, slides 21-24): a two-player
coordination game played on each edge. Each node chooses A or B; the payoff
increases if it chooses like its neighbours. The decision rule reduces to a
THRESHOLD:

    a node v adopts A  <=>  (fraction of neighbours in A) >= q ,   with q = b/(a+b)

where a,b are the coordination-game payoffs. The cascade is DETERMINISTIC:
given the early-adopter set S (hard-wired to A), the threshold q and the
topology, the outcome is unique (unlike the stochastic SI/SIS/SIR models).

Experiments:
  1) critical threshold q* varying the seeding STRATEGY (random / hubs /
     a single community), at equal seed budget;
  2) communities as BARRIERS to the cascade (slide 24): seeding the cascade
     inside one community, how far does it break into the others?

Produces in data/figures/:
  - game_threshold_seeding.png   : cascade size vs q per seeding strategy;
  - game_community_barriers.png  : adoption per community when seeding one.

Usage:
    python game_theoretic.py
"""

import os
import csv
import gzip
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

random.seed(42)


# --------------------------------------------------------------------------
# Loading the network (adjacency list) and communities
# --------------------------------------------------------------------------
def load_adjacency(data_dir):
    """Undirected network -> dict did -> set(neighbours)."""
    path = os.path.join(data_dir, "network_undirected.edgelist")
    if os.path.exists(path):
        fh = open(path)
    else:
        fh = gzip.open(path + ".gz", "rt")
    adj = {}
    for line in fh:
        parts = line.split()
        if len(parts) < 2:
            continue
        u, v = parts[0], parts[1]
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    fh.close()
    return adj


def load_communities(data_dir):
    """did -> community id (Louvain)."""
    comm = {}
    with open(os.path.join(data_dir, "communities.csv")) as f:
        for r in csv.DictReader(f):
            comm[r["did"]] = r["louvain_community"]
    return comm


# --------------------------------------------------------------------------
# Deterministic coordination-game cascade (Linear Threshold)
# --------------------------------------------------------------------------
def cascade(adj, seeds, q):
    """
    DETERMINISTIC threshold cascade.
      adj   : dict did -> set(neighbours)
      seeds : set of early adopters (hard-wired to A)
      q     : coordination threshold = b/(a+b)
    Returns the final set of nodes in A.

    An inactive node v switches to A if the fraction of its neighbours already
    in A is >= q. Only the neighbours of the just-activated nodes (frontier)
    are re-checked, until no new node changes state.
    """
    active = set(seeds)
    frontier = set()
    for s in seeds:
        frontier |= adj.get(s, set())
    frontier -= active

    while frontier:
        newly = set()
        for v in frontier:
            nb = adj.get(v, ())
            if not nb:
                continue
            active_nb = sum(1 for w in nb if w in active)
            if active_nb / len(nb) >= q:
                newly.add(v)
        if not newly:
            break
        active |= newly
        nxt = set()
        for v in newly:
            nxt |= adj[v]
        frontier = nxt - active
    return active


# --------------------------------------------------------------------------
# Custom (ad-hoc) variant: COMMUNITY-AWARE coordination cascade.
# This is the bespoke diffusion model required by the manipulation task: it
# extends the plain cascade by exploiting the external semantic information
# (Louvain communities). Cross-community ties carry a coordination weight
# omega in [0,1], while intra-community ties carry weight 1. A node adopts A iff
#     (A_in + omega*A_out) / (D_in + omega*D_out) >= q
# where A_in/A_out (D_in/D_out) are the active (total) neighbours inside/outside
# the node's community. omega=1 recovers the plain model; omega<1 models
# homophilous coordination (people coordinate more with same-community peers).
# --------------------------------------------------------------------------
def cascade_community_aware(adj, comm, seeds, q, omega):
    active = set(seeds)
    frontier = set()
    for s in seeds:
        frontier |= adj.get(s, set())
    frontier -= active
    while frontier:
        newly = set()
        for v in frontier:
            nb = adj.get(v, ())
            if not nb:
                continue
            cv = comm.get(v)
            a_in = a_out = d_in = d_out = 0
            for w in nb:
                same = (comm.get(w) == cv)
                if same:
                    d_in += 1
                    if w in active:
                        a_in += 1
                else:
                    d_out += 1
                    if w in active:
                        a_out += 1
            den = d_in + omega * d_out
            if den <= 0:
                continue
            if (a_in + omega * a_out) / den >= q:
                newly.add(v)
        if not newly:
            break
        active |= newly
        nxt = set()
        for v in newly:
            nxt |= adj[v]
        frontier = nxt - active
    return active


# --------------------------------------------------------------------------
# Seeding strategies (at equal budget k)
# --------------------------------------------------------------------------
def seeds_random(adj, k):
    return set(random.sample(list(adj.keys()), k))


def seeds_top_degree(adj, k):
    return set(sorted(adj, key=lambda n: len(adj[n]), reverse=True)[:k])


def seeds_one_community(adj, comm, k):
    """k (high-degree) nodes taken from the largest community."""
    from collections import Counter
    biggest = Counter(comm.values()).most_common(1)[0][0]
    members = [n for n in adj if comm.get(n) == biggest]
    members.sort(key=lambda n: len(adj[n]), reverse=True)
    return set(members[:k]), biggest


# --------------------------------------------------------------------------
# Experiment 1: critical threshold vs seeding strategy
# --------------------------------------------------------------------------
def exp_threshold_seeding(adj, comm, fig_dir, k=150,
                          qs=(0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)):
    N = len(adj)
    s_rand = seeds_random(adj, k)
    s_hub = seeds_top_degree(adj, k)
    s_comm, biggest = seeds_one_community(adj, comm, k)
    strategies = {"random": s_rand, "hubs (high degree)": s_hub,
                  f"one community (#{biggest})": s_comm}

    print(f"\n[E1] critical threshold vs seeding (budget k={k} nodes, {100*k/N:.1f}% of net)")
    results = {}
    for label, S in strategies.items():
        sizes = []
        for q in qs:
            final = cascade(adj, S, q)
            sizes.append(len(final) / N)
        results[label] = sizes
        # q* = highest threshold that still yields a "global" cascade (>50%)
        qstar = max([q for q, s in zip(qs, sizes) if s > 0.5], default=None)
        print(f"   {label:24s}  q*≈{qstar}   (size@q_min={sizes[0]:.2f}, "
              f"size@q_max={sizes[-1]:.3f})")

    plt.figure(figsize=(7, 5))
    for label, sizes in results.items():
        plt.plot(qs, sizes, marker="o", label=label)
    plt.xlabel("coordination threshold  q = b/(a+b)")
    plt.ylabel("final fraction of A-adopters")
    plt.title(f"Game-theoretic cascade: effect of threshold and seeds (k={k})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(fig_dir, "game_threshold_seeding.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   [plot] {out}")


# --------------------------------------------------------------------------
# Experiment 2: communities as barriers to the cascade
# --------------------------------------------------------------------------
def exp_community_barriers(adj, comm, fig_dir,
                           qs=(0.20, 0.30, 0.40, 0.42, 0.44, 0.45, 0.47, 0.50)):
    """
    Seed an ENTIRE community and measure the "spill-over" into the others as q
    varies. At low q the cascade floods everything; as q grows it stays confined
    to the seeded community -> dense communities act as barriers (slide 24).
    """
    from collections import Counter, defaultdict
    sizes = Counter(comm.values())
    seed_comm = sizes.most_common(1)[0][0]
    seeds = {n for n in adj if comm.get(n) == seed_comm}
    others_total = len(adj) - len(seeds)

    print(f"\n[E2] communities as barriers: seeding the WHOLE community "
          f"#{seed_comm} ({len(seeds)} nodes) varying q")
    spillover = []          # fraction of nodes OUTSIDE the seed that adopt A
    per_comm_at = {}        # per-community detail at an intermediate q
    target_q = 0.45
    for q in qs:
        final = cascade(adj, seeds, q)
        out_nodes = sum(1 for n in final if comm.get(n) != seed_comm)
        spillover.append(out_nodes / others_total)
        if abs(q - target_q) < 1e-9:
            adopt = defaultdict(int)
            for n in final:
                adopt[comm.get(n)] += 1
            per_comm_at = {c: adopt[c] / sizes[c] for c in sizes}
        print(f"   q={q:.2f}  spill-over into the other communities = "
              f"{spillover[-1]*100:5.1f}%")

    # Figure 1: spill-over vs q (how well the barriers hold)
    plt.figure(figsize=(7, 5))
    plt.plot(qs, spillover, marker="o", color="tab:purple")
    plt.xlabel("coordination threshold  q = b/(a+b)")
    plt.ylabel("fraction of adopters OUTSIDE the seeded community")
    plt.title(f"Cascade spill-over outside community #{seed_comm}")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(fig_dir, "game_community_barriers.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   [plot] {out}")

    # Figure 2: per-community detail at the intermediate threshold target_q
    if per_comm_at:
        labels, fracs, colors = [], [], []
        for c in sorted(sizes, key=lambda x: -sizes[x]):
            labels.append(f"#{c}\n(n={sizes[c]})")
            fracs.append(per_comm_at[c])
            colors.append("tab:red" if c == seed_comm else "tab:blue")
        print(f"   detail at q={target_q}: " +
              ", ".join(f"#{c}={per_comm_at[c]*100:.0f}%"
                        for c in sorted(sizes, key=lambda x: -sizes[x])))
        plt.figure(figsize=(8, 5))
        plt.bar(labels, fracs, color=colors)
        plt.ylabel("fraction of nodes adopting A")
        plt.title(f"Adoption per community (seed = #{seed_comm}, q={target_q})")
        plt.ylim(0, 1)
        plt.tight_layout()
        out2 = os.path.join(fig_dir, "game_community_barriers_detail.png")
        plt.savefig(out2, dpi=150)
        plt.close()
        print(f"   [plot] {out2}")


# --------------------------------------------------------------------------
# Experiment 3 (custom model): homophilous coordination weight omega.
# Seed one community and vary the cross-community weight omega: how much does
# the cascade spill over into the other communities?
# --------------------------------------------------------------------------
def exp_custom_homophily(adj, comm, fig_dir, q=0.40,
                         omegas=(1.0, 0.9, 0.8, 0.75, 0.5, 0.25, 0.1)):
    from collections import Counter
    sizes = Counter(comm.values())
    seed_comm = sizes.most_common(1)[0][0]
    seeds = {n for n in adj if comm.get(n) == seed_comm}
    others = len(adj) - len(seeds)

    print(f"\n[E3] custom community-aware model: seed community #{seed_comm} "
          f"({len(seeds)} nodes), q={q}, varying omega")
    spill, total = [], []
    for w in omegas:
        final = cascade_community_aware(adj, comm, seeds, q, w)
        spill.append(sum(1 for n in final if comm.get(n) != seed_comm) / others)
        total.append(len(final) / len(adj))
        print(f"   omega={w:<4}  spill-over={spill[-1]*100:5.1f}%  "
              f"total adopters={total[-1]*100:5.1f}%")

    plt.figure(figsize=(7, 5))
    plt.plot(omegas, [s*100 for s in spill], marker="o", label="spill-over (other communities)")
    plt.plot(omegas, [t*100 for t in total], marker="s", label="total adopters")
    plt.xlabel("cross-community coordination weight  $\\omega$")
    plt.ylabel("adopters (%)")
    plt.title(f"Community-aware cascade: effect of homophilous coordination (q={q})")
    plt.gca().invert_xaxis()   # from standard (omega=1) to strongly homophilous
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(fig_dir, "game_custom_omega.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"   [plot] {out}")


def main():
    data_dir = config.DATA_DIR
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("[game] loading network and communities...")
    adj = load_adjacency(data_dir)
    comm = load_communities(data_dir)
    print(f"[game] network: {len(adj)} nodes; communities: {len(set(comm.values()))} (Louvain)")

    exp_threshold_seeding(adj, comm, fig_dir)
    exp_community_barriers(adj, comm, fig_dir)
    exp_custom_homophily(adj, comm, fig_dir)

    print("\n[game] done. For the report: (1) hubs trigger global cascades at "
          "higher q than random seeds; (2) a dense community, once seeded, "
          "struggles to break into the others -> communities act as barriers "
          "(cluster of density > 1-q, slide 24).")


if __name__ == "__main__":
    main()
