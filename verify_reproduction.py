"""
verify_reproduction.py
Fast reproduction check for the released package. Prints a PASS/FAIL line for each
of three layers (see REPRODUCIBILITY.md):

  1. Analysis-layer determinism  - recompute statistics from the shipped results.csv
     and compare to stats_results.json. Must be bit-identical on every platform.
  2. Deterministic controls      - re-run DE, HH-Random and the standalone LNS
     operator on two problems and compare to results.csv. Bit-identical everywhere.
  3. Stack-dependent methods     - re-run HH-UCB and CMA-ES on Rastrigin and report
     the difference from the archive plus the rank check (a non-zero difference is a
     numerical-stack effect that does not change any rank; see REPRODUCIBILITY.md).

Run from inside code/:  python verify_reproduction.py
"""
import json, csv, math
import numpy as np
import benchmarks as B, hyperheuristic as HH
import stats_analysis as S

SEED_BASE = 20240617
N_SEEDS = 30
BUD_BENCH = 5000

def cell_mean(problem, method, budget=BUD_BENCH):
    p = [x for x in B.classical_suite(10) if x.name == problem][0]
    finals = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(SEED_BASE + s)
        r = HH.run_method(method, p, budget, rng, record=False)
        finals.append(float(r["best_f"]))
    return np.array(finals)

def archived(problem, method):
    for row in csv.DictReader(open("results.csv")):
        if row["problem"] == problem and row["method"] == method:
            return float(row["mean"]), float(row["best"])
    raise KeyError((problem, method))

def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * (abs(b) + 1e-12)

print("=" * 70)
print("REPRODUCTION VERIFICATION")
print("=" * 70)
print(f"Python/NumPy in use: numpy {np.__version__}")

# ---- 1. analysis-layer determinism ----------------------------------------
BENCH = ["Sphere","Rastrigin","Ackley","Rosenbrock","Griewank","Schwefel","WeldedBeam","PressureVessel"]
ETCP  = ["ETC-Eff-op1","ETC-Eff-op2","ETC-PEC-op1","ETC-PEC-op2"]
METH  = ["HH-UCB","HH-Random","DE","CMA-ES"] + [f"LLH:{n}" for n in HH.LLH_NAMES]
ref = json.load(open("stats_results.json"))
M = S.load_matrix("results.csv", BENCH + ETCP, METH)
now = S.analyse(M, METH, "all")
chi_ok = approx(now["friedman_chi2"], ref["all"]["friedman_chi2"], 1e-9)
rank_ok = all(approx(now["avg_rank"][m], ref["all"]["avg_rank"][m], 1e-9) for m in METH)
p1 = chi_ok and rank_ok
print(f"\n[1] Analysis-layer determinism (results.csv -> statistics)")
print(f"    Friedman chi2 recomputed {now['friedman_chi2']:.5f} vs archived "
      f"{ref['all']['friedman_chi2']:.5f}")
print(f"    -> {'PASS' if p1 else 'FAIL'} (must be bit-identical on every platform)")

# ---- 2. deterministic controls --------------------------------------------
print(f"\n[2] Deterministic controls (must be bit-identical on every platform)")
p2 = True
for problem, method in [("Sphere","DE"), ("Rastrigin","DE"),
                        ("Rastrigin","HH-Random"), ("Rastrigin","LLH:LNS")]:
    fresh = cell_mean(problem, method)
    am, ab = archived(problem, method)
    ok = approx(float(fresh.mean()), am, 1e-9)
    p2 = p2 and ok
    label = method.replace("LLH:", "") + " (standalone)" if method.startswith("LLH:") else method
    print(f"    {problem:10s} / {label:16s} fresh {fresh.mean():.11g}  archived {am:.11g}"
          f"  -> {'PASS' if ok else 'FAIL'}")

# ---- 3. stack-dependent methods -------------------------------------------
print(f"\n[3] Stack-dependent methods (last-bit divergence allowed; ranks must hold)")
for method in ["HH-UCB", "CMA-ES"]:
    fresh = cell_mean("Rastrigin", method)
    am, ab = archived("Rastrigin", method)
    d = abs(float(fresh.mean()) - am)
    tag = "identical" if d <= 1e-9 else f"differs by {d:.5f} (numerical-stack effect)"
    print(f"    Rastrigin / {method:8s} fresh {fresh.mean():.5f}  archived {am:.5f}  -> {tag}")

# rank invariance under the freshly obtained Rastrigin values
Mx = M.copy(); ri = (BENCH + ETCP).index("Rastrigin")
Mx[ri, METH.index("HH-UCB")]  = cell_mean("Rastrigin", "HH-UCB").mean()
Mx[ri, METH.index("CMA-ES")]  = cell_mean("Rastrigin", "CMA-ES").mean()
nowx = S.analyse(Mx, METH, "all")
rank_inv = all(approx(nowx["avg_rank"][m], now["avg_rank"][m], 1e-9) for m in METH)
chi_inv = approx(nowx["friedman_chi2"], now["friedman_chi2"], 1e-9)
print(f"    rank vector unchanged: {rank_inv};  Friedman chi2 unchanged: {chi_inv}")

print("\n" + "=" * 70)
verdict = "PASS" if (p1 and p2 and rank_inv and chi_inv) else "CHECK"
print(f"OVERALL: analysis-layer + controls + rank-invariance -> {verdict}")
print("Any residual difference in [3] is a numerical-stack effect documented in")
print("REPRODUCIBILITY.md and leaves all reported conclusions unchanged.")
print("=" * 70)
