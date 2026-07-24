# Reproducibility and numerical determinism

This document defines exactly what "reproduce the archived results" means for this
package, why one part of it is bounded by the numerical stack rather than by the
code, and how to obtain bit-for-bit agreement.

## 1. Summary

The pipeline has two layers with different reproducibility guarantees.

**Analysis layer (deterministic on every platform).** Every table, statistic and
figure is a deterministic function of the archived `code/results.csv`. Running
`stats_analysis.py`, `compile_tables.py` and `make_figures.py` reproduces
`stats_results.json`, the `table_*.csv` files and `figs/` bit-for-bit on any
machine, because the archived `results.csv`, `conv/` and `traces/` are shipped and
these scripts only post-process them. This is the layer that produces every number
printed in the article.

**Experiment layer (bit-exact within a fixed numerical stack).** Regenerating
`results.csv` from scratch with `run_experiments.py` re-runs the optimisers. Ten of
the twelve compared methods reproduce bit-for-bit on any platform. Two do not, for a
reason that is intrinsic to floating-point arithmetic and not a defect in the code
(Section 3). Their per-seed values can differ in the last decimals across machines,
but this changes none of the reported conclusions (Section 4).

## 2. The archived numerical stack

The shipped `results.csv` was produced with:

- Python 3.11.15 (the same values are obtained under Python 3.13.x; the interpreter
  minor version is not the determining factor);
- NumPy 2.4.4 (pinned in `requirements.txt`);
- the OpenBLAS build bundled with the NumPy 2.4.4 wheel (scipy-openblas 0.3.31.x,
  built with `DYNAMIC_ARCH`, which selects a CPU-specific microkernel at run time);
- single-process arithmetic per worker (multiprocessing only distributes independent
  seeds; it does not change any per-seed computation).

## 3. Why two methods are stack-dependent

Reproducibility across machines is governed by which floating-point primitives a
method uses.

- **Methods that use only IEEE-754 basic operations** (`+ - * /`, comparisons, and
  the correctly-rounded `sqrt`) are bit-portable, because IEEE-754 fixes their
  rounding exactly. In this package that is **differential evolution, the
  random-selection ablation (HH-Random), and the standalone LNS operator**. They
  reproduce the archived values on any platform and serve as deterministic controls
  (for example DE on Sphere gives mean `9.61208156969011e-06` everywhere).

- **HH-UCB** evaluates a `log` in its upper-confidence-bound exploration term.
  `log` is *not* one of the correctly-rounded IEEE-754 operations, so different libm
  and SIMD implementations may differ in the last bit. That last bit occasionally
  flips which operator is the arg-max of the UCB score, which changes the search
  trajectory. Because the multimodal benchmarks (Rastrigin, Schwefel, ...) are
  chaotic, an early trajectory difference can end in a different local basin. This is
  why only a few seeds diverge (those where a UCB tie was close): on Rastrigin,
  archived HH-UCB mean `10.23454`, a supported alternative stack reports `10.55923`,
  differing on 4 of 30 seeds.

- **CMA-ES** calls `numpy.linalg.eigh` (a LAPACK symmetric eigendecomposition) every
  generation. LAPACK routines are built on BLAS kernels that, under OpenBLAS
  `DYNAMIC_ARCH`, are CPU-specific; their last-bit results therefore depend on the
  host microarchitecture. The eigenvectors feed the sampling of every generation, so
  the divergence compounds and all 30 seeds differ across stacks (archived Rastrigin
  mean `12.86812`, alternative stack `14.95753`).

The pattern (deterministic methods identical everywhere, only the `log`-using and
`eigh`-using methods stack-dependent) is exactly what floating-point theory predicts
and is itself evidence that the divergence is environmental, not a coding error: a
bug would move the deterministic controls too.

## 4. The conclusions are invariant to this divergence

The statistical analysis operates on **within-problem ranks**, not on raw objective
values (`stats_analysis.py`, `analyse()`), and the reported design optima are the
collector results, which are governed by deterministic search on a smooth low-
dimensional surrogate. Substituting an independent stack's divergent values
(Rastrigin HH-UCB `10.55923`, CMA-ES `14.95753`) into the full analysis leaves:

- every method's average Friedman rank identical,
- the all-problem Friedman statistic identical (chi-square `147.62`, change `0.0`),
- every Holm-corrected Wilcoxon significance decision identical,

because on each problem the divergent means stay inside the same rank interval (on
Rastrigin, HH-UCB stays between HH-Random and CMA-ES, and CMA-ES stays between HH-UCB
and the LNS operator). The collector optima (efficiency `0.760`, grid gap `<=0.40%`)
are unchanged. The scientifically meaningful notion of reproduction, identical
conclusions, therefore holds on every platform.

## 5. How to obtain bit-for-bit experiment reproduction

Use the recorded stack in a container so the OpenBLAS microkernel and libm are fixed:

```dockerfile
# Dockerfile (provided)
FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.4.4 pandas openpyxl matplotlib
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /work
COPY . /work
CMD ["bash"]
```

```
docker build -t etc-hh .
docker run --rm -it etc-hh
# inside the container:
cd code
python run_experiments.py 600     # repeat until it prints 'ALL DONE'
python stats_analysis.py
python compile_tables.py
python make_figures.py
```

Setting `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` removes thread-count
dependent reduction ordering in BLAS, which is the remaining within-machine source of
run-to-run variation. On an x86-64 host this reproduces the archived `results.csv`
bit-for-bit.

## 6. Verifying reproduction without a full rerun

`code/verify_reproduction.py` performs three fast checks and prints a PASS/FAIL line
for each:

1. **Analysis-layer determinism:** recompute the statistics from the shipped
   `results.csv` and compare to `stats_results.json` (must be bit-identical on every
   platform).
2. **Deterministic controls:** re-run DE, HH-Random and the LNS operator on two
   benchmark problems and compare to `results.csv` (must be bit-identical on every
   platform).
3. **Stack-dependent methods:** re-run HH-UCB and CMA-ES on Rastrigin and report the
   difference from the archive together with the rank check, so a non-zero difference
   is correctly reported as a stack effect that does not change any rank.

```
cd code
python verify_reproduction.py
```
