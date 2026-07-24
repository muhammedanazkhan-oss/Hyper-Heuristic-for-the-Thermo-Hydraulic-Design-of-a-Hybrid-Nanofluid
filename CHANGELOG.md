# Changelog

## v1.0.3 (2026-07-23) - corrections to the computational pipeline

Six issues raised in peer review were corrected in the code and all affected
experiments re-run. The deterministic methods reproduce their v1.0.2 values exactly
(DE on Sphere: mean = 9.61208156969011e-06 before and after).

1. **LNS operator** (`hyperheuristic.py`, `k == 3`): now `j = rng.choice(self.d);
   c[j] = self.best_x[j]` (a single shared index) instead of independent source and
   destination indices.
2. **CMA-ES** (`hyperheuristic.py`, `cma_es`): terminates exactly at the evaluation
   budget and records its convergence history in true evaluation order (running best),
   not sorted order.
3. **Wilcoxon** (`stats_analysis.py`): computed on within-problem ranks
   (scale-invariant), with a tie-adjusted variance in the normal approximation.
4. **Friedman** (`stats_analysis.py`): tie correction applied to the statistic
   (all-problem chi-square = 147.62; the result remains highly significant).
5. **Figure 6** (`make_figures.py`, `fig_stats`): the pairwise Wilcoxon heatmap now
   uses within-problem ranks, not the raw objective matrix.
6. **Figure 2** (`make_figures.py`, `fig_convergence`): collector confidence bands use
   30 runs and sd/sqrt(30); Figures 3 and 4 x-axis relabelled "operator invocations";
   `make_figures.py` `main()` restored (the v1.0.2 file was truncated).

### Packaging and metadata
- `run_experiments.py`: the multiprocessing start method now falls back to `spawn`
  when `fork` is unavailable (Windows), so the driver runs on all platforms.
- `README.md`: the reproduction steps no longer reference per-round `revision*.py`
  scripts (not part of the release); `revision_results.json` is shipped precomputed
  and `append_revision.py` regenerates the appended analyses.
- Added `.zenodo.json`, `CITATION.cff` (version 1.0.3), `LICENSE` (MIT),
  `LICENSE-DATA.txt` (CC BY 4.0), `Dockerfile`, `REPRODUCIBILITY.md`,
  `verify_reproduction.py` and `set_doi.sh`.
- `__pycache__`/`*.pyc` removed from the archive.

### Table 10 (collector verification)
The reconstructed-model column of Table 10 was reconciled with the shipped
`compile_tables.py` output: efficiency point 1 = 0.75964, criterion point 1 =
1.00992, criterion point 2 = 1.01172; the reconstructed model agrees with the
surrogate to within 0.29 percent. Headline optima are unchanged (efficiency 0.760,
grid 0.762, maximum grid gap 0.40 percent).

### Reproducibility note
Exact bit-for-bit reproduction of `results.csv` requires the pinned numerical stack
(NumPy 2.4.4 and its bundled BLAS/LAPACK microkernels), not the NumPy version alone.
HH-UCB (via libm `log`) and CMA-ES (via LAPACK `eigh`) are the only stack-dependent
methods; the deterministic controls reproduce everywhere and all reported rankings,
significance tests and collector optima are invariant to the residual last-bit
divergence. See `REPRODUCIBILITY.md`.

### Headline results (all 12 problems, tie-tolerant Friedman ranks)
DE 3.833, VNS 3.833, HH-Random 4.500, CMA-ES 4.583, HH-UCB 4.667.
HH-UCB vs HH-Random: rank-based Wilcoxon p = 0.615 (Holm 1.0).
HH-UCB significantly beats 6 operators (PSO and the five problem-specific moves) under
Holm correction.
