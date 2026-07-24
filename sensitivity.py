"""Sensitivity-guided search-space reduction, exact on the balanced full factorial."""
import numpy as np, pandas as pd, json
UP="../data/ETC_Hybrid_Nanofluid_Sweep.xlsx"
FACTORS=["hybrid_pair","base_fluid","w_pct","comp1_share_pct","V_Lmin","Ti_C","Is_Wm2","Ta_C"]
COMP=["hybrid_pair","base_fluid","w_pct","comp1_share_pct"]; OPER=["V_Lmin","Ti_C","Is_Wm2","Ta_C"]
LABEL={"hybrid_pair":"hybrid pair","base_fluid":"base fluid","w_pct":"weight fraction",
       "comp1_share_pct":"component share","V_Lmin":"flow rate","Ti_C":"inlet temp",
       "Is_Wm2":"irradiance","Ta_C":"ambient temp"}
def first_order(df,resp):
    y=df[resp].values;gm=y.mean();sst=np.sum((y-gm)**2);out={}
    for f in FACTORS:
        out[f]=float(sum(len(g)*(g[resp].mean()-gm)**2 for _,g in df.groupby(f))/sst)
    return out
def grouped(df,resp):
    y=df[resp].values;gm=y.mean();var=y.var()
    mC=df.groupby(COMP)[resp].transform("mean");mO=df.groupby(OPER)[resp].transform("mean")
    SC=float(((mC-gm)**2).mean()/var);SO=float(((mO-gm)**2).mean()/var)
    return dict(composition=SC,operating=SO,interaction=float(1-SC-SO))
def morris(df,resp):
    cards=[df[f].nunique() for f in FACTORS]
    s=df.sort_values(FACTORS);Y=s[resp].values.reshape(cards);out={}
    for i,f in enumerate(FACTORS):
        step=1.0/(cards[i]-1);ee=np.diff(Y,axis=i)/step
        out[f]=dict(mu_star=float(np.mean(np.abs(ee))),sigma=float(np.std(ee)))
    return out
def fixed_op(df,resp,Ti,Is,Ta):
    op=df[(df.Ti_C==Ti)&(df.Is_Wm2==Is)&(df.Ta_C==Ta)];y=op[resp].values;gm=y.mean();sst=np.sum((y-gm)**2)
    return {f:(float(sum(len(g)*(g[resp].mean()-gm)**2 for _,g in op.groupby(f))/sst) if sst>0 else 0.0)
            for f in ["hybrid_pair","base_fluid","w_pct","comp1_share_pct","V_Lmin"]}
def main():
    df=pd.read_excel(UP,sheet_name="Simulation Data");res={}
    for resp in ["PEC","eta_thermal"]:
        res[resp]=dict(first_order=first_order(df,resp),grouped=grouped(df,resp),morris=morris(df,resp),
                       fixed_op1=fixed_op(df,resp,40,1200,25),fixed_op2=fixed_op(df,resp,60,1200,25))
    json.dump(res,open("sensitivity_results.json","w"),indent=2)
    for resp in res:
        g=res[resp]["grouped"];print(f"\n=== {resp} ===")
        print(f"  grouped: composition={g['composition']*100:.1f}% operating={g['operating']*100:.1f}% interaction={g['interaction']*100:.1f}%")
        fo=sorted(res[resp]["first_order"].items(),key=lambda kv:-kv[1])
        print("  first-order top: "+", ".join(f"{LABEL[k]} {v:.3f}" for k,v in fo[:4]))
        fop=sorted(res[resp]["fixed_op1"].items(),key=lambda kv:-kv[1])
        print("  fixed-op1 decision vars: "+", ".join(f"{LABEL[k]} {v:.3f}" for k,v in fop))
if __name__=="__main__": main()
