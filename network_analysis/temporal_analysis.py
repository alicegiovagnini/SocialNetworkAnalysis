"""
TEMPORAL ANALYSIS - account-creation dynamics and community composition.

Uses the account-creation dates (node_attributes) together with the Louvain
communities to produce two figures (in data/figures/):
  - account_creation.png   : accounts created per month (the two adoption waves);
  - cohort_composition.png : disciplinary (community) mix of the two waves --
                             the 2023 founding cohort vs the Nov-2024 migration.

Finding: the November-2024 migration (post-US-election exodus from X) is more
science-heavy than the 2023 founding cohort -- chemistry/biology roughly doubles
its share while politics/news recedes.

Usage:
    python temporal_analysis.py
"""

import os
import csv
import gzip
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Louvain community id -> topic (from the TF-IDF keyword analysis).
TOPICS = {"0": "politics/\nnews", "1": "climate/\nenergy", "2": "chemistry/\nbiology",
          "3": "neuro-\nscience", "4": "conserv./\necology", "5": "space"}
ORDER = ["0", "1", "2", "3", "4", "5"]


def load_created(data_dir):
    """did -> 'YYYY-MM' creation month (plausible dates only, 2023-2026)."""
    out = {}
    p = os.path.join(data_dir, "node_attributes.csv")
    fh = open(p, newline="", encoding="utf-8") if os.path.exists(p) \
        else gzip.open(p + ".gz", "rt", newline="", encoding="utf-8")
    for r in csv.DictReader(fh):
        c = r.get("createdAt", "")
        if len(c) >= 7 and c[:4].isdigit() and 2023 <= int(c[:4]) <= 2026:
            out[r["did"]] = c[:7]
    fh.close()
    return out


def load_comm(data_dir):
    comm = {}
    with open(os.path.join(data_dir, "communities.csv")) as f:
        for r in csv.DictReader(f):
            comm[r["did"]] = r["louvain_community"]
    return comm


def cohort(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    if y == 2023:
        return "2023 founding wave"
    if (y == 2024 and m >= 10) or (y == 2025 and m <= 1):
        return "Nov-2024 migration"
    return None


def plot_creation_timeline(created, fig_dir):
    months = collections.Counter(created.values())
    keys = sorted(months)
    vals = [months[k] for k in keys]
    plt.figure(figsize=(8, 4.2))
    plt.bar(range(len(keys)), vals, color="tab:blue")
    step = max(1, len(keys) // 12)
    plt.xticks(range(0, len(keys), step), [keys[i] for i in range(0, len(keys), step)],
               rotation=45, ha="right")
    plt.ylabel("accounts created")
    plt.xlabel("month")
    plt.title("Account creation over time (collected network)")
    plt.tight_layout()
    out = os.path.join(fig_dir, "account_creation.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[plot] {out}  (peak month: {max(months, key=months.get)})")


def plot_cohort_composition(created, comm, fig_dir):
    agg = collections.defaultdict(collections.Counter)
    for did, ym in created.items():
        co, cm = cohort(ym), comm.get(did)
        if co and cm is not None:
            agg[co][cm] += 1
    waves = ["2023 founding wave", "Nov-2024 migration"]
    shares = {w: [100 * agg[w].get(c, 0) / sum(agg[w].values()) for c in ORDER]
              for w in waves}
    x = np.arange(len(ORDER))
    w = 0.38
    plt.figure(figsize=(8, 4.6))
    plt.bar(x - w / 2, shares[waves[0]], w, label=waves[0], color="tab:blue")
    plt.bar(x + w / 2, shares[waves[1]], w, label=waves[1], color="tab:orange")
    plt.xticks(x, [TOPICS[c] for c in ORDER])
    plt.ylabel("share of the cohort (%)")
    plt.title("Disciplinary composition of the two adoption waves")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(fig_dir, "cohort_composition.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[plot] {out}")
    for wv in waves:
        print(f"  {wv}: " + ", ".join(
            f"{TOPICS[c].replace(chr(10), '')} {shares[wv][i]:.0f}%"
            for i, c in enumerate(ORDER)))


def main():
    data_dir = config.DATA_DIR
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    created = load_created(data_dir)
    comm = load_comm(data_dir)
    print(f"[temporal] accounts with a plausible creation date: {len(created)}")
    plot_creation_timeline(created, fig_dir)
    plot_cohort_composition(created, comm, fig_dir)


if __name__ == "__main__":
    main()
