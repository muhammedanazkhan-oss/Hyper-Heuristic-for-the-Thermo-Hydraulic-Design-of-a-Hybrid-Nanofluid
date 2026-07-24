"""
surrogate.py
Histogram-based gradient-boosted regression-tree surrogate, implemented in pure
numpy (no scikit-learn / xgboost available in this sandbox). The eight ETC design
inputs are low-cardinality grid factors, so a histogram split finder over the few
distinct values per feature trains in seconds and predicts quickly.

  HistGBT.fit(X, y)            squared-error gradient boosting with shrinkage
  HistGBT.predict(X)          vectorised batch prediction (accuracy reporting)
  HistGBT.predict_one(x)      tree-vectorised single-point prediction (search loop)

Trees are stored as padded numpy arrays so single-point prediction descends all
trees simultaneously, which keeps the optimiser's inner loop fast.
"""
import numpy as np


class _Tree:
    __slots__ = ("feat", "thr", "left", "right", "val", "nnode")
    def __init__(self, cap):
        self.feat = np.zeros(cap, np.int32)
        self.thr = np.zeros(cap, np.float64)
        self.left = -np.ones(cap, np.int32)
        self.right = -np.ones(cap, np.int32)
        self.val = np.zeros(cap, np.float64)
        self.nnode = 0
    def add(self):
        i = self.nnode; self.nnode += 1; return i


class HistGBT:
    def __init__(self, n_estimators=300, learning_rate=0.1, max_depth=6,
                 min_leaf=20, max_bins=64, subsample=1.0, seed=0):
        self.M = n_estimators; self.lr = learning_rate; self.max_depth = max_depth
        self.min_leaf = min_leaf; self.max_bins = max_bins
        self.subsample = subsample; self.seed = seed

    # ---- binning
    def _bin_fit(self, X):
        self.edges = []
        Xb = np.zeros_like(X, dtype=np.int16)
        for j in range(X.shape[1]):
            vals = np.unique(X[:, j])
            if len(vals) > self.max_bins:
                qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                ed = np.quantile(vals, qs)
            else:
                ed = (vals[:-1] + vals[1:]) / 2.0
            self.edges.append(ed)
            Xb[:, j] = np.searchsorted(ed, X[:, j]).astype(np.int16)
        return Xb

    def _bin_transform(self, X):
        Xb = np.zeros_like(X, dtype=np.int16)
        for j in range(X.shape[1]):
            Xb[:, j] = np.searchsorted(self.edges[j], X[:, j]).astype(np.int16)
        return Xb

    # ---- single regression tree on residual g
    def _grow(self, Xb, g):
        n, d = Xb.shape
        cap = 2 ** (self.max_depth + 1)
        T = _Tree(cap)
        nbins = [int(Xb[:, j].max()) + 1 for j in range(d)]
        stack = [(np.arange(n), 0, T.add())]
        while stack:
            idx, depth, node = stack.pop()
            gi = g[idx]
            T.val[node] = gi.mean()
            if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
                T.feat[node] = -1; continue
            S = gi.sum(); N = len(idx)
            best_gain = 1e-12; best = None
            for j in range(d):
                b = Xb[idx, j]
                nb = nbins[j]
                if nb < 2:
                    continue
                sg = np.bincount(b, weights=gi, minlength=nb)
                cn = np.bincount(b, minlength=nb).astype(np.float64)
                csg = np.cumsum(sg); ccn = np.cumsum(cn)
                NL = ccn[:-1]; SL = csg[:-1]
                NR = N - NL; SR = S - SL
                ok = (NL >= self.min_leaf) & (NR >= self.min_leaf)
                if not np.any(ok):
                    continue
                gain = np.where(ok, SL * SL / np.maximum(NL, 1) +
                                SR * SR / np.maximum(NR, 1) - S * S / N, -1)
                t = int(np.argmax(gain))
                if gain[t] > best_gain:
                    best_gain = gain[t]; best = (j, t)
            if best is None:
                T.feat[node] = -1; continue
            j, t = best
            mask = Xb[idx, j] <= t
            li, ri = idx[mask], idx[~mask]
            if len(li) < self.min_leaf or len(ri) < self.min_leaf:
                T.feat[node] = -1; continue
            T.feat[node] = j
            T.thr[node] = 0.5  # bin threshold (compare binned <= t); stored separately
            T.thr[node] = t + 0.5
            ln = T.add(); rn = T.add()
            T.left[node] = ln; T.right[node] = rn
            stack.append((li, depth + 1, ln))
            stack.append((ri, depth + 1, rn))
        return T

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xb = self._bin_fit(X)
        self.base = y.mean()
        pred = np.full(len(y), self.base)
        self.trees = []
        rng = np.random.default_rng(self.seed)
        for m in range(self.M):
            g = y - pred                      # negative gradient of squared loss
            if self.subsample < 1.0:
                sel = rng.random(len(y)) < self.subsample
                T = self._grow(Xb[sel], g[sel])
            else:
                T = self._grow(Xb, g)
            self.trees.append(T)
            pred += self.lr * self._tree_pred_binned(T, Xb)
        self._pack()
        return self

    def _tree_pred_binned(self, T, Xb):
        n = Xb.shape[0]
        node = np.zeros(n, np.int32)
        for _ in range(self.max_depth + 1):
            f = T.feat[node]
            leaf = f < 0
            if leaf.all():
                break
            j = np.where(leaf, 0, f)
            xb = Xb[np.arange(n), j]
            go_left = xb <= T.thr[node]
            child = np.where(go_left, T.left[node], T.right[node])
            node = np.where(leaf, node, child)
        return T.val[node]

    # ---- pack trees into padded arrays for fast prediction
    def _pack(self):
        T = self.M
        maxn = max(t.nnode for t in self.trees)
        self.P_feat = np.full((T, maxn), -1, np.int32)
        self.P_thr = np.zeros((T, maxn), np.float64)
        self.P_left = np.zeros((T, maxn), np.int32)
        self.P_right = np.zeros((T, maxn), np.int32)
        self.P_val = np.zeros((T, maxn), np.float64)
        for m, t in enumerate(self.trees):
            k = t.nnode
            self.P_feat[m, :k] = t.feat[:k]
            self.P_thr[m, :k] = t.thr[:k]
            self.P_left[m, :k] = np.maximum(t.left[:k], 0)
            self.P_right[m, :k] = np.maximum(t.right[:k], 0)
            self.P_val[m, :k] = t.val[:k]
        self.maxdepth = self.max_depth

    def predict(self, X):
        X = np.asarray(X, float)
        Xb = self._bin_transform(X).astype(np.float64)
        n = X.shape[0]
        out = np.full(n, self.base)
        for m in range(self.M):
            feat = self.P_feat[m]; thr = self.P_thr[m]
            left = self.P_left[m]; right = self.P_right[m]; val = self.P_val[m]
            node = np.zeros(n, np.int32)
            for _ in range(self.maxdepth + 1):
                f = feat[node]; leaf = f < 0
                if leaf.all():
                    break
                j = np.where(leaf, 0, f)
                go_left = Xb[np.arange(n), j] <= thr[node]
                child = np.where(go_left, left[node], right[node])
                node = np.where(leaf, node, child)
            out += self.lr * val[node]
        return out

    def predict_one(self, x):
        """tree-vectorised single point: descend all trees simultaneously."""
        x = np.asarray(x, float)
        xb = np.array([np.searchsorted(self.edges[j], x[j]) for j in range(len(x))],
                      dtype=np.float64)
        T = self.M
        node = np.zeros(T, np.int32)
        ar = np.arange(T)
        for _ in range(self.maxdepth + 1):
            f = self.P_feat[ar, node]
            leaf = f < 0
            if leaf.all():
                break
            j = np.where(leaf, 0, f)
            go_left = xb[j] <= self.P_thr[ar, node]
            child = np.where(go_left, self.P_left[ar, node], self.P_right[ar, node])
            node = np.where(leaf, node, child)
        return self.base + self.lr * self.P_val[ar, node].sum()


def metrics(y, yh):
    y = np.asarray(y); yh = np.asarray(yh)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - yh) ** 2) / ss if ss > 0 else 1.0
    rmse = np.sqrt(np.mean((y - yh) ** 2))
    mae = np.mean(np.abs(y - yh))
    nz = np.abs(y) > 1e-12
    mape = 100 * np.mean(np.abs((y[nz] - yh[nz]) / y[nz]))
    return dict(R2=r2, RMSE=rmse, MAE=mae, MAPE=mape)
