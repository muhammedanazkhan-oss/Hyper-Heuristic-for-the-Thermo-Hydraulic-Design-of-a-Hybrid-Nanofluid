import json,pandas as pd
RV=json.load(open("revision_results.json"))
tabs={}
te=RV["total_effect"]
tabs["total_effect"]=pd.DataFrame([dict(response=r,**{k:te[r][k] for k in te[r]}) for r in te])
co=RV["conditional_optimum"]
tabs["conditional_optimum"]=pd.DataFrame([
 dict(objective="PEC",optimal_hybrid_counts=str(co["PEC_hybrid"]),optimal_fluid_counts=str(co["PEC_fluid"])),
 dict(objective="efficiency",optimal_hybrid_counts=str(co["eff_hybrid"]),optimal_fluid_counts=str(co["eff_fluid"]))])
tabs["blocked_cv"]=pd.DataFrame([dict(group=k,**{kk:vv for kk,vv in v.items()}) for k,v in RV["blocked_cv"].items()])
tabs["pec_top_region"]=pd.DataFrame([RV["pec_top_region"]])
tabs["wpcap_sensitivity"]=pd.DataFrame([dict(cap=c,**RV["wpcap_sensitivity"][c]) for c in RV["wpcap_sensitivity"]])
tabs["friedman_ranks_tol"]=pd.DataFrame([dict(method=m,avg_rank=r) for m,r in RV["friedman_ranks_tol"].items()])
tabs["optima_verification"]=pd.DataFrame(RV["optima_verification"])
tabs["closure_sensitivity"]=pd.DataFrame([dict(closure=k,**v) for k,v in RV["closure_sensitivity"].items()])
tabs["encoding_check"]=pd.DataFrame([RV["encoding_check"]])
pp=RV["particle_props"];tabs["particle_props"]=pd.DataFrame([dict(particle=k,**v) for k,v in pp.items()])
for nm,df in tabs.items(): df.to_csv(f"table_rev_{nm}.csv",index=False)
with pd.ExcelWriter("results_tables.xlsx",mode="a",engine="openpyxl",if_sheet_exists="replace") as xl:
    for nm,df in tabs.items(): df.to_excel(xl,sheet_name=("rev_"+nm)[:31],index=False)
print("appended",len(tabs),"revision sheets")
