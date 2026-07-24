"""Train numpy GB surrogates, ONE file per target (surr_<name>.pkl) to avoid races."""
import numpy as np, pandas as pd, pickle, json, time, sys, os
import surrogate as S
UP="../data/ETC_Hybrid_Nanofluid_Sweep.xlsx"
HMAP={'Al2O3-Cu':0,'MWCNT-Fe3O4':1,'Graphene-TiO2':2};FMAP={'Distilled water':0,'EG/water 60:40':1,'Synthetic HTF oil':2}
TARGETS={"eta":("eta_thermal",False,300,6),"Wp":("Wp_W",True,300,6),
         "Re":("Re",False,300,6),"To":("To_C",False,300,6),"PEC":("PEC",False,350,8)}
def main(which):
    df=pd.read_excel(UP,sheet_name="Simulation Data")
    X=np.c_[df.hybrid_pair.map(HMAP),df.base_fluid.map(FMAP),df.w_pct,df.comp1_share_pct,
            df.V_Lmin,df.Ti_C,df.Is_Wm2,df.Ta_C].astype(float)
    rng=np.random.default_rng(42);idx=rng.permutation(len(X));ntr=int(0.85*len(X));tr,te=idx[:ntr],idx[ntr:]
    for name in which:
        col,logsp,M,dep=TARGETS[name]
        y=df[col].values.astype(float);yt=np.log(y) if logsp else y
        t=time.time()
        m=S.HistGBT(n_estimators=M,learning_rate=0.08,max_depth=dep,min_leaf=20).fit(X[tr],yt[tr])
        yh=m.predict(X[te])
        met=S.metrics(np.exp(yt[te]),np.exp(yh)) if logsp else S.metrics(y[te],yh)
        met2=S.metrics(yt[te],yh);m.logspace=logsp
        rec=dict(target=name,n_estimators=M,max_depth=dep,logspace=logsp,R2=round(met2["R2"],6),
                 RMSE=round(met["RMSE"],5),MAE=round(met["MAE"],5),MAPE=round(met["MAPE"],4),
                 fit_s=round(time.time()-t,1),n_test=len(te))
        pickle.dump(m,open(f"surr_{name}.pkl","wb"))
        json.dump(rec,open(f"surr_{name}_metrics.json","w"),indent=2)
        print(f"{name}: R2={met2['R2']:.5f} MAPE={met['MAPE']:.3f}% ({time.time()-t:.1f}s) -> surr_{name}.pkl")
if __name__=="__main__": main(sys.argv[1:] if len(sys.argv)>1 else list(TARGETS))
