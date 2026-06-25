"""
stats_analysis.py
Non-parametric statistical comparison of optimisers across problem blocks, in
pure numpy (no scipy). Implements the Friedman test (chi-square and Iman-Davenport
F forms), the Nemenyi post-hoc critical difference, and pairwise Wilcoxon
signed-rank tests with Holm correction. Chi-square and normal tail probabilities
are computed from incomplete-gamma / error-function series.
"""
import numpy as np, pandas as pd, json, math, csv

# ---- special functions (no scipy) -----------------------------------------
def _gammaincc(a, x):
    if x < 0 or a <= 0: return 1.0
    if x < a + 1:                       # series for P, then Q=1-P
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-12: break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c = x + 1 - a, 1e30              # continued fraction for Q
    d = 1.0 / b; h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-30 if abs(d) < 1e-30 else d
        c = b + an / c; c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-12: break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))

def chi2_sf(x, k):   return _gammaincc(k / 2.0, x / 2.0)
def norm_sf(z):      return 0.5 * math.erfc(z / math.sqrt(2))

# Nemenyi q_alpha (alpha=0.05), studentised range / sqrt(2), index by #methods
Q05 = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164,
       11:3.219,12:3.268,13:3.313,14:3.354,15:3.391,16:3.426,17:3.458,18:3.489,
       19:3.517,20:3.544}

def wilcoxon_signed_rank(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    n = len(d)
    if n == 0: return 0.0, 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    # average ties
    order = np.argsort(np.abs(d)); ad = np.abs(d)[order]
    ranks = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and ad[j + 1] == ad[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    rr = np.empty(n); rr[order] = ranks
    Wp = np.sum(rr[d > 0]); Wm = np.sum(rr[d < 0])
    W = min(Wp, Wm)
    mu = n * (n + 1) / 4.0; sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu + 0.5) / sig if sig > 0 else 0.0
    return float(W), float(2 * norm_sf(abs(z)))

def holm(pairs):
    items = sorted(pairs.items(), key=lambda kv: kv[1])
    m = len(items); out = {}; mx = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p); mx = max(mx, adj); out[k] = mx
    return out

def load_matrix(results_csv, problems, methods, metric="mean"):
    rows = {(r["problem"], r["method"]): r for r in csv.DictReader(open(results_csv))}
    M = np.full((len(problems), len(methods)), np.nan)
    for i, p in enumerate(problems):
        for j, m in enumerate(methods):
            if (p, m) in rows:
                M[i, j] = float(rows[(p, m)][metric])
    return M

def _rank_tol(v, tol=1e-4):
    v=np.asarray(v,float); k=len(v); order=np.argsort(v); r=np.zeros(k); j=0
    while j<k:
        j2=j
        while j2+1<k and abs(v[order[j2+1]]-v[order[j]])<=tol*(abs(v[order[j]])+1e-12): j2+=1
        r[order[j:j2+1]]=(j+j2)/2.0+1; j=j2+1
    return r

def analyse(M, methods, tag, hh="HH-UCB"):
    # rank methods within each problem (ascending = better, minimisation)
    N, k = M.shape
    ranks = np.array([_rank_tol(M[i]) for i in range(N)])
    avg = ranks.mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4.0)
    p_chi = chi2_sf(chi2, k - 1)
    Fid = (N - 1) * chi2 / (N * (k - 1) - chi2) if (N * (k - 1) - chi2) != 0 else np.inf
    cd = Q05.get(k, 3.5) * math.sqrt(k * (k + 1) / (6.0 * N))
    # pairwise Wilcoxon: hh vs each other across problems
    hi = methods.index(hh); praw = {}
    for j, m in enumerate(methods):
        if m == hh: continue
        _, p = wilcoxon_signed_rank(M[:, hi], M[:, j]); praw[m] = p
    padj = holm(praw)
    return dict(tag=tag, methods=methods, avg_rank={m: float(avg[j]) for j, m in enumerate(methods)},
                friedman_chi2=float(chi2), friedman_df=k - 1, p_value=float(p_chi),
                iman_davenport_F=float(Fid), N_blocks=N, CD=float(cd),
                wilcoxon_raw={k_: float(v) for k_, v in praw.items()},
                wilcoxon_holm={k_: float(v) for k_, v in padj.items()})

def main():
    BENCH = ["Sphere","Rastrigin","Ackley","Rosenbrock","Griewank","Schwefel","WeldedBeam","PressureVessel"]
    ETCP = ["ETC-Eff-op1","ETC-Eff-op2","ETC-PEC-op1","ETC-PEC-op2"]
    import hyperheuristic as HH
    METHODS = ["HH-UCB","HH-Random","DE","CMA-ES"] + [f"LLH:{n}" for n in HH.LLH_NAMES]
    out = {}
    Mall = load_matrix("results.csv", BENCH + ETCP, METHODS)
    out["all"] = analyse(Mall, METHODS, "all 12 problems")
    out["bench"] = analyse(load_matrix("results.csv", BENCH, METHODS), METHODS, "8 benchmarks")
    out["etc"] = analyse(load_matrix("results.csv", ETCP, METHODS), METHODS, "4 ETC problems")
    json.dump(out, open("stats_results.json", "w"), indent=2)
    for tag in out:
        o = out[tag]
        print(f"\n[{o['tag']}] Friedman chi2={o['friedman_chi2']:.2f} (df={o['friedman_df']}) "
              f"p={o['p_value']:.2e}  CD={o['CD']:.3f}")
        ar = sorted(o["avg_rank"].items(), key=lambda kv: kv[1])
        print("  best avg ranks: " + ", ".join(f"{m} {r:.2f}" for m, r in ar[:5]))

if __name__ == "__main__":
    main()
