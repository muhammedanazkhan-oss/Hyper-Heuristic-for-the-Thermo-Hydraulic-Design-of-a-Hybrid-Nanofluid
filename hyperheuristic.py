"""
hyperheuristic.py
Selection hyper-heuristic with a mixed low-level-heuristic (LLH) portfolio,
UCB1 online credit assignment (recency-weighted, tiered success reward),
late-acceptance hill-climbing (LAHC) acceptance, and restart intensification.
Also: uniform-random-selector ablation, single-LLH runners, and standalone DE and
CMA-ES baselines. Pure numpy. One function evaluation is charged per proposed
candidate, so every method shares an identical budget axis.
"""
import numpy as np

LLH_NAMES = ["SA", "Tabu", "VNS", "LNS", "DE/rand/1", "DE/best/1", "PSO",
             "perturb-V", "perturb-w", "perturb-share", "swap-hybrid", "swap-fluid"]
N_LLH = len(LLH_NAMES)

def _clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


class HyperHeuristic:
    def __init__(self, problem, rng, controller='ucb', NP=16, lahc_len=50,
                 ucb_c=0.10, alpha=0.30, record=False):
        self.p = problem; self.rng = rng
        self.lo, self.hi = problem.lo, problem.hi; self.d = problem.dim
        self.span = self.hi - self.lo
        self.controller = controller; self.NP = NP; self.lahc_len = lahc_len
        self.ucb_c = ucb_c; self.alpha = alpha; self.record = record
        self.named = getattr(problem, "named_coords", {})
        self.cat = getattr(problem, "cat_dims", {})

    def _ev(self, x):
        self.nev += 1
        f = self.p.eval(x)
        if f < self.best_f:
            self.best_f = f; self.best_x = x.copy()
        if self.nev <= self.budget:
            self.hist[self.nev - 1] = self.best_f
        return f

    # ---------------- low-level heuristics
    def _llh(self, k):
        s = self.cur.copy(); rng = self.rng
        a = 0.20 + 0.80 * self.temp
        if k == 0:
            sig = 0.18 * self.span * self.temp
            return _clip(s + rng.normal(0, 1, self.d) * sig, self.lo, self.hi)
        if k == 1:
            best, bf, bkey = None, np.inf, None
            for _ in range(6):
                c = _clip(s + rng.normal(0, 1, self.d) * 0.10 * self.span * a, self.lo, self.hi)
                key = tuple(np.round((c - self.lo) / self.span * 24).astype(int))
                if key in self.tabu:
                    continue
                fc = self._ev(c)
                if fc < bf:
                    bf, best, bkey = fc, c, key
                if self.nev >= self.budget:
                    break
            if best is None:
                return _clip(s + rng.normal(0, 1, self.d) * 0.10 * self.span, self.lo, self.hi)
            self.tabu.append(bkey)
            if len(self.tabu) > 25:
                self.tabu.pop(0)
            self._cand_pre_evaluated = (best, bf)
            return best
        if k == 2:
            self.vns_k = (self.vns_k % 4) + 1
            mag = 0.06 * self.vns_k * self.span * a
            j = rng.integers(0, self.d, size=min(self.vns_k, self.d))
            c = s.copy(); c[j] = s[j] + rng.normal(0, 1, len(j)) * mag[j]
            return _clip(c, self.lo, self.hi)
        if k == 3:
            ndes = max(1, int(0.4 * self.d))
            j = rng.choice(self.d, ndes, replace=False)
            c = s.copy(); c[j] = self.lo[j] + rng.random(ndes) * self.span[j]
            j = rng.choice(self.d); c[j] = self.best_x[j]
            return _clip(c, self.lo, self.hi)
        if k in (4, 5):
            idx = rng.choice(self.NP, 3, replace=False)
            aa, bb, cc = self.pop[idx]
            F = 0.5 + 0.3 * rng.random()
            base = self.pop[np.argmin(self.popf)] if k == 5 else aa
            mut = base + F * (bb - cc)
            mask = rng.random(self.d) < 0.9; mask[rng.integers(0, self.d)] = True
            return _clip(np.where(mask, mut, s), self.lo, self.hi)
        if k == 6:
            w, c1, c2 = 0.6, 1.5, 1.5
            r1, r2 = rng.random(self.d), rng.random(self.d)
            self.vel = w * self.vel + c1 * r1 * (self.pbest - s) + c2 * r2 * (self.best_x - s)
            self.vel = _clip(self.vel, -0.5 * self.span, 0.5 * self.span)
            return _clip(s + self.vel, self.lo, self.hi)
        if k in (7, 8, 9):
            role = {7: 'V', 8: 'w', 9: 'share'}[k]
            j = self.named.get(role, (k * 7 + 3) % self.d)
            c = s.copy(); c[j] = s[j] + rng.normal(0, 1) * 0.15 * self.span[j] * a
            return _clip(c, self.lo, self.hi)
        if k in (10, 11):
            role = {10: 'hybrid', 11: 'fluid'}[k]
            j = (0 if self.cat else (1 % self.d)) if role == 'hybrid' \
                else (1 if len(self.cat) > 1 else (2 % self.d))
            c = s.copy()
            if j in self.cat:
                n = self.cat[j]; cur = int(min(n - 1, max(0, np.floor(s[j]))))
                ch = [v for v in range(n) if v != cur] or [cur]
                c[j] = rng.choice(ch) + 0.5
            else:
                c[j] = self.lo[j] + rng.random() * self.span[j]
            return _clip(c, self.lo, self.hi)
        raise ValueError(k)

    # ---------------- UCB1 selection (recency-weighted value)
    def _select(self):
        if self.controller == 'random':
            return self.rng.integers(0, N_LLH)
        if isinstance(self.controller, tuple):
            return self.controller[1]
        for k in range(N_LLH):
            if self.n[k] == 0:
                return k
        bonus = self.ucb_c * np.sqrt(np.log(max(2, self.t)) / self.n)
        return int(np.argmax(self.val + bonus))

    def _credit(self, k, best_before, parent_f, cand_f):
        reward = 1.0 if cand_f < best_before - 1e-12 else 0.0   # rewards new global-best moves
        self.val[k] += self.alpha * (reward - self.val[k])      # recency-weighted value (EMA)
        self.n[k] += 1
        if self.record:
            self.sel_log.append(k); self.rew_log.append(reward)

    # ---------------- main loop
    def run(self, budget):
        rng = self.rng; p = self.p; self.budget = budget
        self.hist = np.full(budget, np.inf); self.nev = 0
        self.best_f = np.inf; self.best_x = None
        self.pop = np.array([p.random(rng) for _ in range(self.NP)])
        self.popf = np.array([self._ev(self.pop[i]) for i in range(self.NP)])
        b = int(np.argmin(self.popf))
        self.cur = self.pop[b].copy(); self.cur_f = self.popf[b]
        self.pbest = self.cur.copy(); self.pbest_f = self.cur_f
        self.vel = np.zeros(self.d); self.temp = 1.0
        self.tabu = []; self.vns_k = 0
        self.n = np.zeros(N_LLH, int); self.val = np.zeros(N_LLH)
        self.t = 0; self.lahc = np.full(self.lahc_len, self.cur_f)
        if self.record:
            self.sel_log = []; self.rew_log = []; self.pos_log = []; self.bestf_log = []
        while self.nev < budget:
            self.t += 1
            self.temp = max(0.02, 1.0 - self.nev / budget)
            best_before = self.best_f
            k = self._select()
            self._cand_pre_evaluated = None
            cand = self._llh(k)
            if self.nev >= budget:
                break
            if self._cand_pre_evaluated is not None:
                cand, cand_f = self._cand_pre_evaluated
            else:
                cand_f = self._ev(cand)
            self._credit(k, best_before, self.cur_f, cand_f)
            v = self.t % self.lahc_len
            if cand_f <= self.lahc[v] or cand_f <= self.cur_f:
                self.cur = cand; self.cur_f = cand_f
                if cand_f < self.pbest_f:
                    self.pbest = cand.copy(); self.pbest_f = cand_f
            self.lahc[v] = self.cur_f
            w = int(np.argmax(self.popf))
            if cand_f < self.popf[w]:
                self.pop[w] = cand; self.popf[w] = cand_f
            if self.record:
                self.pos_log.append(self.cur.copy()); self.bestf_log.append(self.best_f)
        self.hist[self.nev:] = self.best_f
        out = dict(best_f=self.best_f, best_x=self.best_x, history=self.hist.copy(), nev=self.nev)
        if self.record:
            out["sel_log"] = np.array(self.sel_log); out["rew_log"] = np.array(self.rew_log)
            out["pos_log"] = np.array(self.pos_log); out["bestf_log"] = np.array(self.bestf_log)
        return out


# ============================================================ external baselines
def canonical_de(problem, budget, rng, NP=30, F=0.6, CR=0.9):
    p = problem; d = p.dim
    pop = np.array([p.random(rng) for _ in range(NP)])
    fit = np.array([p.eval(x) for x in pop]); nev = NP
    hist = np.full(budget, np.inf); hist[:nev] = fit.min()
    while nev < budget:
        for i in range(NP):
            idx = rng.choice([j for j in range(NP) if j != i], 3, replace=False)
            a, b, c = pop[idx]
            mut = _clip(a + F * (b - c), p.lo, p.hi)
            mask = rng.random(d) < CR; mask[rng.integers(0, d)] = True
            trial = np.where(mask, mut, pop[i])
            ft = p.eval(trial); nev += 1
            if ft <= fit[i]:
                pop[i] = trial; fit[i] = ft
            if nev <= budget:
                hist[nev - 1] = fit.min()
            if nev >= budget:
                break
    hist[nev:] = fit.min(); bi = int(np.argmin(fit))
    return dict(best_f=float(fit[bi]), best_x=pop[bi].copy(), history=hist, nev=nev)


def cma_es(problem, budget, rng, sigma0=0.3):
    p = problem; n = p.dim
    xmean = p.lo + 0.5 * (p.hi - p.lo); sigma = sigma0 * np.mean(p.hi - p.lo)
    lam = 4 + int(3 * np.log(n)); mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n); cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    pc = np.zeros(n); ps = np.zeros(n); C = np.eye(n)
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    nev = 0; hist = np.full(budget, np.inf); best_f = np.inf; best_x = None
    while nev < budget:
        try:
            D2, B = np.linalg.eigh(C); D = np.sqrt(np.maximum(D2, 1e-20))
        except np.linalg.LinAlgError:
            C = np.eye(n); D2, B = np.linalg.eigh(C); D = np.sqrt(D2)
        Z = rng.normal(0, 1, (lam, n)); Y = Z @ (B * D).T
        X = _clip(xmean + sigma * Y, p.lo, p.hi)
        take = min(lam, budget - nev)                       # respect the total budget exactly
        f = np.array([p.eval(x) for x in X[:take]])
        for e in range(take):                               # history in true evaluation order (running best)
            ix = nev + e
            if f[e] < best_f:
                best_f = float(f[e]); best_x = X[e].copy()
            if ix < budget:
                hist[ix] = best_f
        nev += take
        if take < lam:                                      # final truncated generation: stop at budget
            break
        order = np.argsort(f)
        xold = xmean; sel = X[order[:mu]]; xmean = w @ sel; ymean = (xmean - xold) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ ymean) / np.maximum(D, 1e-12)))
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * nev / lam)) / chiN) < (1.4 + 2 / (n + 1))
        pc = (1 - cc) * pc + (hsig * np.sqrt(cc * (2 - cc) * mueff)) * ymean
        artmp = (sel - xold) / sigma
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (artmp.T * w) @ artmp)
        C = np.triu(C) + np.triu(C, 1).T
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, np.mean(p.hi - p.lo))
    hist[nev:] = best_f if nev < budget else hist[budget - 1]
    if best_x is None:
        best_x = xmean
    return dict(best_f=best_f, best_x=best_x, history=hist[:budget], nev=nev)


def run_method(method, problem, budget, rng, record=False):
    if method == "HH-UCB":
        return HyperHeuristic(problem, rng, controller='ucb', record=record).run(budget)
    if method == "HH-Random":
        return HyperHeuristic(problem, rng, controller='random', record=record).run(budget)
    if method == "DE":
        return canonical_de(problem, budget, rng)
    if method == "CMA-ES":
        return cma_es(problem, budget, rng)
    if method.startswith("LLH:"):
        return HyperHeuristic(problem, rng, controller=('fixed', LLH_NAMES.index(method.split(":", 1)[1]))).run(budget)
    raise ValueError(method)
