"""
Resumable, time-boxed, parallel experiment driver.
Job unit = (problem, method): 30 independent seeds. Persists incrementally so it
survives the sandbox's per-call time limit. Run repeatedly until 'ALL DONE'.
"""
import os, sys, json, time, csv, hashlib
import numpy as np
import multiprocessing as mp
import benchmarks as B, hyperheuristic as HH, etc_problems as EP

SEED_BASE = 20240617
N_SEEDS = 30
BUD_BENCH = 5000
BUD_ETC = 700
METHODS = ["HH-UCB", "HH-Random", "DE", "CMA-ES"] + [f"LLH:{n}" for n in HH.LLH_NAMES]
BENCH = ["Sphere","Rastrigin","Ackley","Rosenbrock","Griewank","Schwefel","WeldedBeam","PressureVessel"]
ETCP  = ["ETC-Eff-op1","ETC-Eff-op2","ETC-PEC-op1","ETC-PEC-op2"]
PROBLEMS = BENCH + ETCP
TRUTH = json.load(open("etc_truth.json"))
RESULTS = "results.csv"
CONV = "conv"; TRACE = "traces"
os.makedirs(CONV, exist_ok=True); os.makedirs(TRACE, exist_ok=True)

def build(spec):
    if spec in ("WeldedBeam","PressureVessel"):
        p = {"WeldedBeam":B.welded_beam,"PressureVessel":B.pressure_vessel}[spec]()
        return p, BUD_BENCH, p.target, True
    if spec in BENCH:
        p = [x for x in B.classical_suite(10) if x.name==spec][0]
        return p, BUD_BENCH, p.target, True
    kind, op = spec.split("-")[1], spec.split("-")[2]
    Ti = 40 if op=="op1" else 60
    if kind == "Eff":
        p = EP.ConstrainedEfficiency(Ti,1200,25,Wp_cap=0.5)
    else:
        p = EP.PECmax(Ti,1200,25)
    tf = -(0.99*TRUTH[spec]["value"])      # success threshold in minimised space
    return p, BUD_ETC, tf, False

def run_one(args):
    spec, method, seed, record = args
    p, bud, target_f, _ = build(spec)
    rng = np.random.default_rng(SEED_BASE + seed)
    t0 = time.time()
    r = HH.run_method(method, p, bud, rng, record=record)
    wall = time.time() - t0
    hist = r["history"]
    hit = np.where(hist <= target_f)[0]
    ett = int(hit[0]) + 1 if len(hit) else -1
    out = dict(final=float(r["best_f"]), ett=ett, wall=wall, hist=hist.astype(np.float32))
    # ETC raw objective + feasibility + config of best
    if hasattr(p, "objective"):
        bx = r["best_x"]
        out["obj"] = float(p.objective(bx)); out["feas"] = bool(p.feasible(bx))
        out["cfg"] = list(p.decode(bx))
    if record and "sel_log" in r:
        out["sel"] = r["sel_log"].astype(np.int16); out["rew"] = r["rew_log"].astype(np.float32)
    return out

def safe(s): return s.replace("/","_").replace(":","_")

def done_set():
    d = set()
    if os.path.exists(RESULTS):
        for row in csv.DictReader(open(RESULTS)):
            d.add((row["problem"], row["method"]))
    return d

def main(deadline=38):
    t0 = time.time()
    done = done_set()
    units = [(pr, m) for pr in PROBLEMS for m in METHODS if (pr, m) not in done]
    if not units:
        print("ALL DONE"); return
    new = os.path.exists(RESULTS)
    fcsv = open(RESULTS, "a", newline="")
    fields = ["problem","method","best","mean","median","std","worst",
              "succ_rate","mean_ett","mean_wall","finals","obj_mean","obj_best","feas_rate","cfg_best"]
    w = csv.DictWriter(fcsv, fieldnames=fields)
    if not new: w.writeheader()
    pool = mp.Pool(2)
    n_done = 0
    for (spec, method) in units:
        if time.time() - t0 > deadline:
            break
        record_seed = 0 if method == "HH-UCB" else -1
        jobs = [(spec, method, s, (s == record_seed)) for s in range(N_SEEDS)]
        res = pool.map(run_one, jobs)
        finals = np.array([r["final"] for r in res])
        etts = np.array([r["ett"] for r in res], float)
        walls = np.array([r["wall"] for r in res])
        H = np.array([r["hist"] for r in res])
        succ = np.mean(etts > 0)
        mean_ett = float(np.mean(etts[etts > 0])) if np.any(etts > 0) else -1
        row = dict(problem=spec, method=method,
                   best=float(finals.min()), mean=float(finals.mean()),
                   median=float(np.median(finals)), std=float(finals.std()),
                   worst=float(finals.max()), succ_rate=float(succ),
                   mean_ett=mean_ett, mean_wall=float(walls.mean()),
                   finals=json.dumps([round(float(x),6) for x in finals]),
                   obj_mean="", obj_best="", feas_rate="", cfg_best="")
        if "obj" in res[0]:
            objs = np.array([r["obj"] for r in res]); feas = np.array([r["feas"] for r in res])
            row["obj_mean"] = float(objs.mean()); row["obj_best"] = float(objs.max())
            row["feas_rate"] = float(feas.mean())
            row["cfg_best"] = json.dumps(res[int(np.argmax(objs))]["cfg"])
        w.writerow(row); fcsv.flush()
        np.savez_compressed(f"{CONV}/{safe(spec)}__{safe(method)}.npz",
                            mean=H.mean(0), std=H.std(0), finals=finals)
        if method == "HH-UCB" and "sel" in res[record_seed]:
            np.savez_compressed(f"{TRACE}/{safe(spec)}.npz",
                                sel=res[record_seed]["sel"], rew=res[record_seed]["rew"])
        n_done += 1
        print(f"done {spec} / {method}  ({time.time()-t0:.1f}s)")
    pool.close(); pool.join(); fcsv.close()
    rem = len(units) - n_done
    print(f"chunk complete: {n_done} units this call, {rem} remaining")

if __name__ == "__main__":
    # 'fork' is fastest on Linux/macOS but is unavailable on Windows; fall back to
    # the platform default ('spawn' on Windows) so the driver runs everywhere.
    try:
        mp.set_start_method("fork")
    except (RuntimeError, ValueError):
        pass
    main(deadline=float(sys.argv[1]) if len(sys.argv) > 1 else 38)
