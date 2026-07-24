"""Compile all results tables (CSV + multi-sheet xlsx) and ETC ground-truth verification."""
import csv, json, numpy as np, pandas as pd
import hyperheuristic as HH, etc_problems as EP
from etc_common import ETCSimulator
HN=EP.HYBRIDS; FN=EP.FLUIDS
rows={(r['problem'],r['method']):r for r in csv.DictReader(open('results.csv'))}
BEN=["Sphere","Rastrigin","Ackley","Rosenbrock","Griewank","Schwefel","WeldedBeam","PressureVessel"]
ETC=["ETC-Eff-op1","ETC-Eff-op2","ETC-PEC-op1","ETC-PEC-op2"]
METH=["HH-UCB","HH-Random","DE","CMA-ES"]+[f"LLH:{n}" for n in HH.LLH_NAMES]
truth=json.load(open('etc_truth.json'))

# 1 surrogate accuracy
sur=[]
for t in ["eta","Wp","Re","To","PEC"]:
    sur.append(json.load(open(f"surr_{t}_metrics.json")))
T_sur=pd.DataFrame(sur)[["target","n_estimators","max_depth","logspace","R2","RMSE","MAE","MAPE","n_test"]]

# 2 sensitivity
s=json.load(open("sensitivity_results.json")); fac=["hybrid_pair","base_fluid","w_pct","comp1_share_pct","V_Lmin","Ti_C","Is_Wm2","Ta_C"]
LAB={"hybrid_pair":"hybrid pair","base_fluid":"base fluid","w_pct":"weight fraction","comp1_share_pct":"component share","V_Lmin":"flow rate","Ti_C":"inlet temp","Is_Wm2":"irradiance","Ta_C":"ambient temp"}
sens=[]
for f in fac:
    sens.append(dict(factor=LAB[f],PEC_S1=round(s["PEC"]["first_order"][f],4),eta_S1=round(s["eta_thermal"]["first_order"][f],4),
                     PEC_mu_star=round(s["PEC"]["morris"][f]["mu_star"],4),PEC_sigma=round(s["PEC"]["morris"][f]["sigma"],4)))
T_sens=pd.DataFrame(sens)
T_group=pd.DataFrame([dict(response="PEC",**{k:round(v,4) for k,v in s["PEC"]["grouped"].items()}),
                      dict(response="efficiency",**{k:round(v,4) for k,v in s["eta_thermal"]["grouped"].items()})])

# 3 benchmark results
ben=[]
for p in BEN:
    for m in METH:
        r=rows[(p,m)]
        ben.append(dict(problem=p,method=m.replace("LLH:",""),best=float(r['best']),mean=float(r['mean']),
                        median=float(r['median']),std=float(r['std']),worst=float(r['worst']),
                        success=float(r['succ_rate']),mean_FE_to_target=float(r['mean_ett']),mean_wall_s=round(float(r['mean_wall']),3)))
T_ben=pd.DataFrame(ben)

# 4 ETC results per method
etc=[]
for p in ETC:
    for m in METH:
        r=rows[(p,m)]
        cfg=json.loads(r['cfg_best']) if r['cfg_best'] else None
        cfgs=f"{HN[cfg[0]]}/{FN[cfg[1]]} w={cfg[2]:.2f} s={cfg[3]:.1f} V={cfg[4]:.2f}" if cfg else ""
        etc.append(dict(problem=p,method=m.replace("LLH:",""),obj_best=round(float(r['obj_best']),5),
                        obj_mean=round(float(r['obj_mean']),5),feasible_rate=float(r['feas_rate']),best_config=cfgs))
T_etc=pd.DataFrame(etc)

# 5 stats
st=json.load(open("stats_results.json"))
strow=[]
for tag in ["all","bench","etc"]:
    o=st[tag]
    for m in METH:
        strow.append(dict(block=o['tag'],method=m.replace("LLH:",""),avg_rank=round(o['avg_rank'][m],3),
                          wilcoxon_vs_HHUCB_holm=round(o['wilcoxon_holm'].get(m,float('nan')),4) if m!="HH-UCB" else 0.0))
T_stats=pd.DataFrame(strow)
T_fried=pd.DataFrame([dict(block=st[t]['tag'],friedman_chi2=round(st[t]['friedman_chi2'],3),df=st[t]['friedman_df'],
                           p_value=st[t]['p_value'],iman_davenport_F=round(st[t]['iman_davenport_F'],3),
                           N_blocks=st[t]['N_blocks'],CD_0p05=round(st[t]['CD'],3)) for t in ["all","bench","etc"]])

# 6 ETC ground-truth verification: surrogate vs reconstructed simulator vs exact grid
sim=ETCSimulator()
ver=[]
for p in ETC:
    r=rows[(p,'HH-UCB')]; cfg=json.loads(r['cfg_best']); h,f,w,sh,V=cfg
    Ti=40 if p.endswith("op1") else 60
    o=sim.evaluate(HN[h],FN[f],w,sh,V,Ti,1200,25)
    surr_obj=float(r['obj_best']); sim_obj=o['PEC'] if 'PEC' in p else o['eta']
    g=truth[p]
    ver.append(dict(problem=p,metric=('PEC' if 'PEC' in p else 'efficiency'),
                    HH_config=f"{HN[h]}/{FN[f]} w={w:.2f} s={sh:.1f} V={V:.2f}",
                    surrogate=round(surr_obj,5),simulator=round(sim_obj,5),
                    grid_truth=round(g['value'],5),grid_config=f"{g['hybrid']}/{g['fluid']} w={g['w']} s={g['share']} V={g['V']}",
                    gap_to_grid_pct=round(100*(g['value']-surr_obj)/g['value'],3)))
T_ver=pd.DataFrame(ver)

# write CSVs + xlsx
tabs={"surrogate_accuracy":T_sur,"sensitivity_indices":T_sens,"grouped_partition":T_group,
      "benchmark_results":T_ben,"etc_results":T_etc,"avg_ranks_wilcoxon":T_stats,
      "friedman_cd":T_fried,"etc_verification":T_ver}
for nm,df in tabs.items(): df.to_csv(f"table_{nm}.csv",index=False)
with pd.ExcelWriter("results_tables.xlsx") as xl:
    for nm,df in tabs.items(): df.to_excel(xl,sheet_name=nm[:31],index=False)
print("TABLES WRITTEN.")
print("\n=== ETC ground-truth verification ==="); print(T_ver.to_string(index=False))
print("\n=== Friedman/CD ==="); print(T_fried.to_string(index=False))
