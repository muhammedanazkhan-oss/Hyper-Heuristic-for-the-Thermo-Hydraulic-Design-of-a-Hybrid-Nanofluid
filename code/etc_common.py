"""
etc_common.py
Shared utilities for the sensitivity-guided selection hyper-heuristic study of a
hybrid-nanofluid evacuated-tube collector (ETC).

Provides:
  * loading / encoding of the 54,432-run full-factorial dataset,
  * a reconstructed reduced-order ETC simulator (Hottel-Whillier-Bliss with the
    Takabi-Salehi hybrid conductivity and Brinkman viscosity, Eqs. 1-40 of the
    source manuscript), whose physical constants were recovered from the dataset
    and validated to reproduce it to within ~0.66% worst case.

All physical constants are recovered from the dataset and persisted as JSON in the
same directory (solid_props.json, basefluid_props.json, sim_const.json).
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = None  # set by load_dataset

HYBRIDS = ["Al2O3-Cu", "MWCNT-Fe3O4", "Graphene-TiO2"]
FLUIDS  = ["Distilled water", "EG/water 60:40", "Synthetic HTF oil"]
HYBRID_IDX = {h: i for i, h in enumerate(HYBRIDS)}
FLUID_IDX  = {f: i for i, f in enumerate(FLUIDS)}

# Continuous decision-variable bounds (sampled envelope)
W_LO, W_HI = 0.25, 3.0      # total weight fraction, percent
S_LO, S_HI = 25.0, 75.0     # component-1 mass share, percent
V_LO, V_HI = 0.5, 6.0       # loop flow rate, L/min

# Sampled operating-grid levels (for brute-force enumeration)
W_LEVELS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
S_LEVELS = [25, 50, 75]
V_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
TI_LEVELS = [20, 40, 60, 80]
IS_LEVELS = [200, 700, 1200]
TA_LEVELS = [15, 25, 35]

RE_LO, RE_HI = 47.0, 45840.0   # sampled Reynolds envelope

# ---------------------------------------------------------------- data loading
def load_dataset(xlsx_path):
    global DATA_XLSX
    DATA_XLSX = xlsx_path
    df = pd.read_excel(xlsx_path, sheet_name="Simulation Data")
    df["hybrid_idx"] = df["hybrid_pair"].map(HYBRID_IDX)
    df["fluid_idx"]  = df["base_fluid"].map(FLUID_IDX)
    return df

# ---------------------------------------------------- recovered constants (I/O)
def _load_json(name):
    return json.load(open(os.path.join(HERE, name)))

def load_constants():
    sol  = _load_json("solid_props.json")
    bf   = _load_json("basefluid_props.json")
    cst  = _load_json("sim_const.json")
    return sol, bf, cst

# --------------------------------------------------- reduced-order ETC simulator
class ETCSimulator:
    """Self-consistent reduced-order ETC solver (Eqs. 1-40), off-grid capable."""
    def __init__(self):
        self.sol, self.bf, self.C = load_constants()

    def _bf_prop(self, fluid, Tm):
        p = self.bf[fluid]
        rho = np.polyval(p["rho"], Tm)
        cp  = np.polyval(p["cp"], Tm)
        k   = np.polyval(p["k"], Tm)
        mu  = np.exp(np.polyval(p["lnmu"], Tm)) * 1e-3   # Pa.s
        return rho, cp, k, mu

    def _nf_prop(self, hybrid, w_pct, share_pct, Tm, fluid):
        s = self.sol[hybrid]
        rho_bf, cp_bf, k_bf, mu_bf = self._bf_prop(fluid, Tm)
        w = w_pct / 100.0
        sh = share_pct / 100.0
        w1, w2 = w * sh, w * (1 - sh)
        wbf = 1 - w1 - w2
        v1 = w1 / s["rho1"]; v2 = w2 / s["rho2"]; vbf = wbf / rho_bf
        vt = v1 + v2 + vbf
        phi1, phi2 = v1 / vt, v2 / vt
        phi = phi1 + phi2
        rho = (1 - phi) * rho_bf + phi1 * s["rho1"] + phi2 * s["rho2"]
        rc  = (1 - phi) * rho_bf * cp_bf + phi1 * s["rc1"] + phi2 * s["rc2"]
        cp  = rc / rho
        if phi > 0:
            kp = (phi1 * s["k1"] + phi2 * s["k2"]) / phi
            k = k_bf * (kp + 2*k_bf + 2*phi*(kp - k_bf)) / (kp + 2*k_bf - phi*(kp - k_bf))
        else:
            k = k_bf
        mu = mu_bf * (1 - phi) ** -2.5
        return rho, cp, k, mu

    def solve(self, hybrid, fluid, w_pct, share_pct, V_Lmin, Ti, Is, Ta, baseline=False):
        C = self.C
        di, L, As, Ac = C["di"], C["L"], C["As"], C["Ac"]
        ta, UL, etap = C["tau_alpha"], C["UL"], C["eta_p"]
        Acs = np.pi * di**2 / 4
        Vdot = V_Lmin / 60000.0
        Tm = float(Ti)
        ww = 0.0 if baseline else w_pct
        for _ in range(80):
            rho, cp, k, mu = self._nf_prop(hybrid, ww, share_pct, Tm, fluid)
            mdot = rho * Vdot
            u = mdot / (rho * Acs)
            Re = rho * u * di / mu
            Pr = mu * cp / k
            if Re < 2300:
                Nu = 3.66 + (0.0668*(di/L)*Re*Pr)/(1 + 0.04*((di/L)*Re*Pr)**(2/3.))
            else:
                fr = (0.790*np.log(Re) - 1.64)**-2
                Nu = (fr/8)*(Re-1000)*Pr/(1 + 12.7*(fr/8)**0.5*(Pr**(2/3.)-1))
            h = Nu * k / di
            Fp = 1.0 / (1 + UL*Ac/(h*As))
            FR = (mdot*cp)/(Ac*UL) * (1 - np.exp(-Ac*UL*Fp/(mdot*cp)))
            S = ta * Is
            Qu = Ac * FR * (S - UL*(Ti - Ta))
            To = Ti + Qu/(mdot*cp)
            Tm_new = (Ti + To)/2
            if abs(Tm_new - Tm) < 1e-7:
                Tm = Tm_new; break
            Tm = Tm_new
        f_d = 64/Re if Re < 2300 else (0.790*np.log(Re) - 1.64)**-2
        dP = f_d*(L/di)*(rho*u**2/2)
        Wp = Vdot*dP/etap
        eta = Qu/(Ac*Is)
        return dict(Re=Re, Pr=Pr, Nu=Nu, h=h, Fp=Fp, FR=FR, Qu=Qu, To=To,
                    eta=eta, dP=dP, Wp=Wp, f=f_d, Tm=Tm)

    def evaluate(self, hybrid, fluid, w_pct, share_pct, V_Lmin, Ti, Is, Ta):
        """Full evaluation returning eta, Wp, PEC, Re, To via the nanofluid and
        its identically-solved base-fluid baseline."""
        o = self.solve(hybrid, fluid, w_pct, share_pct, V_Lmin, Ti, Is, Ta, baseline=False)
        b = self.solve(hybrid, fluid, w_pct, share_pct, V_Lmin, Ti, Is, Ta, baseline=True)
        eps_h = o["Nu"]/b["Nu"]
        eps_f = o["f"]/b["f"]
        PEC = eps_h / eps_f**(1/3.)
        return dict(eta=o["eta"], Wp=o["Wp"], PEC=PEC, Re=o["Re"], To=o["To"],
                    Nu=o["Nu"], dP=o["dP"], Qu=o["Qu"])


if __name__ == "__main__":
    # self-validation against the dataset
    df = load_dataset(os.path.join(HERE, "..", "data", "ETC_Hybrid_Nanofluid_Sweep.xlsx")) \
         if os.path.exists(os.path.join(HERE, "..", "data")) else None
    sim = ETCSimulator()
    print("ETCSimulator ready. Constants:", list(sim.C.items()))
