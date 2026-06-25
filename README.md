# Sensitivity-Guided Selection Hyper-Heuristic for the Thermo-Hydraulic Design of a Hybrid-Nanofluid Evacuated-Tube Collector

Reproducibility package: code, dataset, trained surrogate models, derived results and figures.

**Author:** Muhammed Anaz Khan
**ORCID:** https://orcid.org/0000-0002-8837-9865
**Affiliation:** Department of Mechanical Engineering, College of Engineering, University of Bisha, Bisha 61922, P.O. Box 551, Saudi Arabia
**Contact:** mkhan@ub.edu.sa

**DOI:** 10.5281/zenodo.XXXXXXXX  *(replace with the DOI minted by Zenodo on deposit; it is also cited in the article's Data Availability statement)*

---

## 1. Overview

This package reproduces, end to end, a fully computational and surrogate-assisted optimization study of a closed-loop hybrid-nanofluid evacuated-tube solar collector. A variance-based sensitivity screening reduces the design space; histogram gradient-boosted regression-tree surrogates emulate a reduced-order collector model; and a **selection hyper-heuristic** (upper-confidence-bound credit assignment with late-acceptance hill-climbing over a twelve-operator portfolio) solves the resulting constrained single-objective problems, benchmarked against a random-selection ablation, differential evolution, CMA-ES and each constituent operator.

The entire pipeline is **pure NumPy**: the gradient-boosted trees and every statistical test (Friedman, Nemenyi critical difference, Holm-corrected Wilcoxon) are implemented from first principles. **SciPy and scikit-learn are not required.**

## 2. Repository structure

```
ETC_HyperHeuristic_Zenodo/
├── README.md                 this file
├── LICENSE                   MIT (source code)
├── LICENSE-DATA.txt          CC BY 4.0 (dataset, models, figures, results)
├── CITATION.cff              machine-readable citation metadata
├── .zenodo.json              Zenodo deposit metadata
├── requirements.txt          Python dependencies
├── data/
│   └── ETC_Hybrid_Nanofluid_Sweep.xlsx   full-factorial dataset (sheet "Simulation Data", 54,432 runs)
└── code/                     working directory: run all scripts from here
    ├── etc_common.py             reduced-order ETC simulator + recovered constants loader
    ├── surrogate.py              pure-NumPy histogram gradient-boosted trees
    ├── train_surrogates.py       trains one surrogate per response -> surr_<name>.pkl
    ├── benchmarks.py             classical + constrained engineering benchmark problems
    ├── hyperheuristic.py         selection hyper-heuristic (UCB + LAHC) + DE / CMA-ES baselines
    ├── etc_problems.py           constrained-efficiency and PEC-maximisation problems
    ├── run_experiments.py        30-trial driver (resumable) -> results.csv, conv/, traces/
    ├── sensitivity.py            exact variance decomposition + Morris screening
    ├── stats_analysis.py         Friedman / Nemenyi / Holm-Wilcoxon
    ├── compile_tables.py         assembles all manuscript tables
    ├── make_figures.py           regenerates all figures
    ├── revision*.py, append_revision.py   supplementary analyses (blocked CV, closure
    │                              sensitivity, conditional-optimum stability, encoding
    │                              invariance, top-region PEC ranking) -> revision_results.json
    ├── solid_props.json, basefluid_props.json, sim_const.json   constants recovered from the dataset
    ├── surr_eta.pkl, surr_Wp.pkl, surr_Re.pkl, surr_To.pkl, surr_PEC.pkl   trained surrogates
    ├── results.csv, stats_results.json, sensitivity_results.json,
    │   revision_results.json, etc_truth.json, *_metrics.json   precomputed results
    ├── results_tables.xlsx        all result tables in one workbook
    ├── figs/                      figures (PNG at 300 dpi + vector PDF)
    ├── conv/                      precomputed convergence arrays (so figures regenerate without rerunning)
    └── traces/                    precomputed search-trajectory arrays
```

## 3. Requirements

Python >= 3.10 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

(`numpy`, `pandas`, `openpyxl`, `matplotlib`.)

## 4. Quick start / reproduction

All scripts are run from inside `code/` and read the dataset from `../data/`. Trained models and all derived results are already provided, so any individual step can be run on its own.

```
cd code

# (optional) retrain the five surrogates from the dataset      -> surr_<name>.pkl + *_metrics.json
python train_surrogates.py

# exact variance-based indices + Morris screening              -> sensitivity_results.json
python sensitivity.py

# 30-trial benchmark + ETC experiments (long; resumable;
# uses the surrogate models)                                   -> results.csv, conv/, traces/
python run_experiments.py

# non-parametric statistical comparison                        -> stats_results.json
python stats_analysis.py

# supplementary review-round analyses                          -> revision_results.json
python revision1.py && python revision2.py && python revision3.py && \
python revision4.py && python revision4a.py && python revision4b.py && \
python revision5a.py && python revision6.py && python revision7.py && \
python revision8a.py && python revision8b.py && python append_revision.py

# assemble every result table                                  -> table_*.csv + results_tables.xlsx
python compile_tables.py

# regenerate every figure                                      -> figs/*.png, figs/*.pdf
python make_figures.py
```

`run_experiments.py` is the only long step (it executes 12 problems x 16 methods x 30 seeds). It is **resumable** and **seeded from a fixed master seed** (see `SEED_BASE` in the script) so results are deterministic. Because `results.csv`, `conv/`, `traces/` and the surrogate models are included, the statistics, tables and figures can be regenerated immediately without rerunning it.

## 5. Dataset

`data/ETC_Hybrid_Nanofluid_Sweep.xlsx`, sheet **"Simulation Data"**, 54,432 rows (a balanced full factorial). Columns:

- **Design factors:** `hybrid_pair`, `component_1`, `component_2`, `base_fluid`, `w_pct` (total weight fraction), `comp1_share_pct`, `dp_nm` (particle diameter), `V_Lmin` (loop flow rate), `Ti_C` (inlet temp.), `Is_Wm2` (irradiance), `Ta_C` (ambient temp.).
- **Hybrid-nanofluid properties:** `phi1`, `phi2`, `phi_total`, `rho_hnf_kgm3`, `cp_hnf_JkgK`, `k_hnf_WmK`, `mu_hnf_mPas`, `Tm_mean_C`; base-fluid counterparts `rho_bf_kgm3`, `cp_bf_JkgK`, `k_bf_WmK`, `mu_bf_mPas`.
- **Hydraulics / heat transfer:** `Vel_ms`, `mdot_kgs`, `Re`, `Pr`, `flow_regime`, `Nu`, `f_friction`, `h_Wm2K`, `F_prime`, `FR`, `S_absorbed_Wm2`, `Qu_W`, `To_C`, `eta_thermal`, `dP_Pa`, `Wp_W`.
- **Baseline and criterion:** `Nu_bf`, `f_bf`, `eta_bf`, `Wp_bf_W`, `eps_h_Nu_ratio`, `eps_f_f_ratio`, `PEC`.

The collector adopts a single-representative-tube convention; absolute thermal quantities are dataset-convention values, as described in the article.

## 6. Notes on reproducibility

- Pure NumPy; no SciPy / scikit-learn. The surrogate (`surrogate.py`) and statistics (`stats_analysis.py`) are self-contained.
- The reconstructed simulator (`etc_common.py`) reproduces the supplied dataset to a median relative error near 0.005 percent; the recovered conductivity coefficients are effective blending coefficients of the property closures, not physical material constants (see `solid_props.json` and the article's Table 1).
- Results are conditional on the supplied reduced-order model, its closures and the sampled envelope; the search is interpolation-only within the sampled grid.

## 7. Licensing

- **Source code** (`code/*.py`): MIT License (`LICENSE`).
- **Dataset, trained models, figures and derived results**: Creative Commons Attribution 4.0 International, CC BY 4.0 (`LICENSE-DATA.txt`).

## 8. How to cite

If you use this material, please cite the Zenodo record (see `CITATION.cff`; insert the DOI after deposit) and the associated journal article once it is published.

## 9. Contact

Muhammed Anaz Khan — mkhan@ub.edu.sa — ORCID 0000-0002-8837-9865
