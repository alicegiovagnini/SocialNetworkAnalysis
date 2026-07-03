# Plots — Network Analysis (Part 2 & Part 3)

Figures produced by the scripts in `network_analysis/` (saved to
`../data_collection/data/figures/` and copied here for the repo).

## Part 2 — Network characterization
- `degree_hist.png` — degree distribution (histogram) — `analysis_starter.py`
- `degree_ccdf_compare.png` — degree CCDF vs ER / BA models — `analysis_starter.py`
- `network_communities.png` — community structure (Louvain), rendered in Gephi
  (ForceAtlas2) from the GEXF written by `export_gephi.py`
- `account_creation.png` — accounts created per month (adoption waves) — `temporal_analysis.py`
- `cohort_composition.png` — community mix of the 2023 vs Nov-2024 cohorts — `temporal_analysis.py`

## Part 3 — Network manipulation & analysis
- `diffusion_models_real.png` — diffusion models on the real network — `diffusion.py`
- `diffusion_sir_beta.png` — SIR sensitivity to β — `diffusion.py`
- `diffusion_sir_seeds.png` — SIR sensitivity to seed set — `diffusion.py`
- `diffusion_sir_compare.png` — SIR comparison — `diffusion.py`
- `game_threshold_seeding.png` — cascade size vs threshold q per seeding strategy — `game_theoretic.py`
- `game_community_barriers.png` — cascade barriers between communities — `game_theoretic.py`
- `game_community_barriers_detail.png` — barriers (detail) — `game_theoretic.py`
- `game_custom_omega.png` — custom semantic model: topical spill-over vs ω — `game_theoretic.py`
