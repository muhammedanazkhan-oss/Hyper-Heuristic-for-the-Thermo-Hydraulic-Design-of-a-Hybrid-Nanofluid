"""ETC single-objective problems on the GB surrogate evaluators (Problem interface)."""
import numpy as np, pickle
HYBRIDS = ["Al2O3-Cu", "MWCNT-Fe3O4", "Graphene-TiO2"]
FLUIDS  = ["Distilled water", "EG/water 60:40", "Synthetic HTF oil"]
RE_LO, RE_HI = 47.0, 45840.0
TO_MAX = 158.0
_SURR = {}
def load_surrogates(keys=("eta","Wp","Re","To","PEC")):
    for k in keys:
        if k not in _SURR:
            _SURR[k] = pickle.load(open(f"surr_{k}.pkl","rb"))
    return _SURR
class _ETCBase:
    def __init__(self, Ti, Is, Ta, keys, penalty=50.0, name=""):
        self.Ti, self.Is, self.Ta = Ti, Is, Ta
        self.dim = 5
        self.lo = np.array([0.0,0.0,0.25,25.0,0.5]); self.hi = np.array([3.0,3.0,3.0,75.0,6.0])
        self.penalty = penalty; self.name = name
        self.cat_dims = {0:3,1:3}; self.named_coords = {'w':2,'share':3,'V':4}
        self.keys = keys; self.S = load_surrogates()
    def decode(self, x):
        return (int(min(2,max(0,np.floor(x[0])))), int(min(2,max(0,np.floor(x[1])))),
                float(x[2]), float(x[3]), float(x[4]))
    def _pred(self, x, keys):
        h,f,w,s,V = self.decode(x)
        feat = np.array([h,f,w,s,V,self.Ti,self.Is,self.Ta], float)
        out = {}
        for k in keys:
            v = self.S[k].predict_one(feat)
            out[k] = np.exp(v) if k == "Wp" else v
        out["cfg"] = (h,f,w,s,V); return out
    def raw(self, x):
        return self._pred(x, ("eta","Wp","Re","To","PEC"))
    def random(self, rng):
        return self.lo + rng.random(self.dim)*(self.hi - self.lo)
class ConstrainedEfficiency(_ETCBase):
    def __init__(self, Ti=40, Is=1200, Ta=25, Wp_cap=0.5, **kw):
        super().__init__(Ti,Is,Ta,("eta","Wp"),name=f"ETC-Eff(Ti{Ti},Is{Is},Ta{Ta},Wcap{Wp_cap})",**kw)
        self.Wp_cap = Wp_cap
    def _viol(self, r):
        return max(0.0,(r["Wp"]-self.Wp_cap)/self.Wp_cap)   # Re/single-phase verified non-binding
    def eval(self, x):
        x = np.clip(x,self.lo,self.hi); r = self._pred(x,("eta","Wp"))
        return float(-r["eta"] + self.penalty*self._viol(r)**2)
    def objective(self, x): return self._pred(x,("eta",))["eta"]
    def feasible(self, x): return self._viol(self._pred(x,("eta","Wp"))) <= 1e-9
class PECmax(_ETCBase):
    def __init__(self, Ti=40, Is=1200, Ta=25, **kw):
        super().__init__(Ti,Is,Ta,("PEC",),name=f"ETC-PEC(Ti{Ti},Is{Is},Ta{Ta})",**kw)
    def eval(self, x):
        x = np.clip(x,self.lo,self.hi)
        return float(-self._pred(x,("PEC",))["PEC"])      # Re/single-phase verified non-binding
    def objective(self, x): return self._pred(x,("PEC",))["PEC"]
    def feasible(self, x): return True
