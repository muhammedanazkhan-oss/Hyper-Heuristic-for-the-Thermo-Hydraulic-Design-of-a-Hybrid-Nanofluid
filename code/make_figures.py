"""Publication figure set (legibility-improved). Colour-blind-safe, 300 dpi PNG + vector PDF.
Larger fonts throughout; legends placed off the plotting area."""
import os, json, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import benchmarks as B, hyperheuristic as HH, etc_problems as EP
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.titlesize":15,
                     "axes.labelsize":13,"xtick.labelsize":11,"ytick.labelsize":11,
                     "legend.fontsize":11,"legend.framealpha":0.95,"axes.grid":True,
                     "grid.alpha":0.25,"axes.axisbelow":True,"savefig.bbox":"tight","savefig.dpi":300})
FIG="figs"; os.makedirs(FIG, exist_ok=True)
OI=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#F0E442','#999999','#117733','#882255','#44AA99','#332288']
LLH=HH.LLH_NAMES
MCOL={"HH-UCB":'#D55E00',"HH-Random":'#0072B2',"DE":'#009E73',"CMA-ES":'#CC79A7',"LLH:VNS":'#882255'}
METHODS=["HH-UCB","HH-Random","DE","CMA-ES"]+[f"LLH:{n}" for n in LLH]
BENCH=["Sphere","Rastrigin","Ackley","Rosenbrock","Griewank","Schwefel","WeldedBeam","PressureVessel"]
ETCP=["ETC-Eff-op1","ETC-Eff-op2","ETC-PEC-op1","ETC-PEC-op2"]
def safe(s): return s.replace("/","_").replace(":","_")
def conv(p,m): d=np.load(f"conv/{safe(p)}__{safe(m)}.npz"); return d["mean"],d["std"],d["finals"]

class TEff(EP.ConstrainedEfficiency):
    def eval(self,x):
        v=super().eval(x); r=self._pred(x,("eta","Wp")); self._tr.append((r["eta"],self._viol(r)<=1e-9)); return v
class TPEC(EP.PECmax):
    def eval(self,x):
        v=super().eval(x); self._tr.append((self._pred(x,("PEC",))["PEC"],True)); return v
def obj_curve(make,method,budget,seeds=12):
    H=[]
    for s in range(seeds):
        p=make(); p._tr=[]
        HH.run_method(method,p,budget,np.random.default_rng(20240617+s))
        obj=np.array([o for o,f in p._tr]); feas=np.array([f for o,f in p._tr])
        best=np.full(len(obj),np.nan); cur=-np.inf
        for i in range(len(obj)):
            if feas[i] and obj[i]>cur: cur=obj[i]
            best[i]=cur
        best=best[:budget]; best=np.pad(best,(0,budget-len(best)),constant_values=best[-1])
        H.append(best)
    H=np.array(H); return np.nanmean(H,0), np.nanstd(H,0)

def fig_convergence():
    fig,axes=plt.subplots(1,4,figsize=(20,5.2)); show=["HH-UCB","HH-Random","DE","CMA-ES","LLH:VNS"]
    for ax,p in [(axes[0],"Rastrigin"),(axes[1],"Schwefel")]:
        for m in show:
            try: mu,sd,_=conv(p,m)
            except: continue
            x=np.arange(1,len(mu)+1)
            ax.plot(x,mu,color=MCOL.get(m,'#444'),lw=2.2,label=m.replace("LLH:",""))
            ax.fill_between(x,mu-sd/np.sqrt(30)*1.96,mu+sd/np.sqrt(30)*1.96,color=MCOL.get(m,'#444'),alpha=0.16,lw=0)
        ax.set_title(p); ax.set_xlabel("function evaluations"); ax.set_ylabel("best objective"); ax.set_yscale("log")
    for ax,(p,make,lab,yl) in zip(axes[2:],[("ETC-Eff-op1",lambda:TEff(40,1200,25,Wp_cap=0.5),"thermal efficiency",(0.40,0.78)),
                                            ("ETC-PEC-op1",lambda:TPEC(40,1200,25),"PEC",(0.97,1.013))]):
        for m in show:
            mu,sd=obj_curve(make,m,700,seeds=12); x=np.arange(1,len(mu)+1)
            ax.plot(x,mu,color=MCOL.get(m,'#444'),lw=2.2,label=m.replace("LLH:",""))
            ax.fill_between(x,mu-sd/np.sqrt(12)*1.96,mu+sd/np.sqrt(12)*1.96,color=MCOL.get(m,'#444'),alpha=0.16,lw=0)
        ax.set_title(p); ax.set_xlabel("function evaluations"); ax.set_ylabel(lab); ax.set_ylim(*yl)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=5,fontsize=13,frameon=True,bbox_to_anchor=(0.5,-0.02))
    fig.suptitle("Convergence of best feasible objective (mean of independent runs, 95% CI band)",y=1.04,fontsize=15)
    fig.subplots_adjust(bottom=0.20,wspace=0.30)
    fig.savefig(f"{FIG}/fig_convergence.png"); fig.savefig(f"{FIG}/fig_convergence.pdf"); plt.close(fig)

def fig_selection_and_credit():
    p=[x for x in B.classical_suite(10) if x.name=="Rastrigin"][0]
    r=HH.HyperHeuristic(p,np.random.default_rng(20240617),controller='ucb',record=True).run(5000)
    sel=r["sel_log"]; rew=r["rew_log"]; n=len(sel); win=max(40,n//50)
    centers=np.arange(win//2,n-win//2,max(1,win//2)); frac=np.zeros((len(centers),len(LLH)))
    for i,c in enumerate(centers):
        w=sel[max(0,c-win//2):c+win//2]
        for k in range(len(LLH)): frac[i,k]=np.mean(w==k) if len(w) else 0
    fig,ax=plt.subplots(figsize=(11,5.8))
    ax.stackplot(centers,frac.T,labels=LLH,colors=OI,alpha=0.9)
    ax.set_xlim(centers.min(),centers.max()); ax.set_ylim(0,1)
    ax.set_xlabel("function evaluations"); ax.set_ylabel("selection fraction (sliding window)")
    ax.set_title("Heuristic-selection map over a genuine search (Rastrigin, 10-D)")
    ax.legend(ncol=6,fontsize=10.5,loc="upper center",bbox_to_anchor=(0.5,-0.13),frameon=True)
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(f"{FIG}/fig_selection_map.png"); fig.savefig(f"{FIG}/fig_selection_map.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.8))
    for k in range(len(LLH)):
        idx=np.where(sel==k)[0]
        if len(idx)<3: continue
        ax.plot(idx,np.cumsum(rew[idx])/np.arange(1,len(idx)+1),color=OI[k],lw=2.0,label=LLH[k])
    ax.set_xlabel("function evaluations"); ax.set_ylabel("running mean reward (new-best rate)")
    ax.set_title("Credit (reward) evolution per low-level heuristic (Rastrigin)")
    ax.legend(ncol=6,fontsize=10.5,loc="upper center",bbox_to_anchor=(0.5,-0.13),frameon=True)
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(f"{FIG}/fig_credit_evolution.png"); fig.savefig(f"{FIG}/fig_credit_evolution.pdf"); plt.close(fig)

def fig_boxplots():
    fig,axes=plt.subplots(1,2,figsize=(17,5.8))
    for ax,p in zip(axes,["Rastrigin","ETC-Eff-op1"]):
        data=[];labs=[]
        for m in METHODS:
            try: _,_,fin=conv(p,m)
            except: continue
            v=-fin if p.startswith("ETC") else fin; data.append(v); labs.append(m.replace("LLH:",""))
        vp=ax.violinplot(data,showmedians=True,widths=0.85)
        for i,b in enumerate(vp['bodies']): b.set_facecolor(OI[i%len(OI)]); b.set_alpha(0.6)
        ax.set_xticks(range(1,len(labs)+1)); ax.set_xticklabels(labs,rotation=45,ha="right",fontsize=11)
        ax.set_title(f"Final solution quality across 30 runs: {p}")
        ax.set_ylabel("thermal efficiency" if p.startswith("ETC") else "best objective")
        if p=="Rastrigin": ax.set_yscale("log")
    fig.subplots_adjust(bottom=0.20,wspace=0.18)
    fig.savefig(f"{FIG}/fig_boxplots.png"); fig.savefig(f"{FIG}/fig_boxplots.pdf"); plt.close(fig)

def fig_stats():
    import stats_analysis as SA
    M=SA.load_matrix("results.csv",BENCH+ETCP,METHODS); k=len(METHODS); P=np.ones((k,k))
    for i in range(k):
        for j in range(k):
            if i!=j: _,pp=SA.wilcoxon_signed_rank(M[:,i],M[:,j]); P[i,j]=pp
    fig,ax=plt.subplots(figsize=(9.8,8.6))
    im=ax.imshow(np.log10(P+1e-4),cmap="viridis",vmin=-4,vmax=0)
    labs=[m.replace("LLH:","") for m in METHODS]
    ax.set_xticks(range(k)); ax.set_xticklabels(labs,rotation=45,ha="right",fontsize=10.5)
    ax.set_yticks(range(k)); ax.set_yticklabels(labs,fontsize=10.5)
    for i in range(k):
        for j in range(k):
            if i!=j and P[i,j]<0.05: ax.text(j,i,"*",ha="center",va="center",color="w",fontsize=11)
    cb=fig.colorbar(im,ax=ax,shrink=0.85); cb.set_label("log10( pairwise Wilcoxon p ),  more negative = more significant",fontsize=12)
    ax.set_title("Pairwise Wilcoxon signed-rank p-values\n(asterisks: uncorrected p<0.05, descriptive only)",fontsize=14)
    fig.savefig(f"{FIG}/fig_wilcoxon_heatmap.png"); fig.savefig(f"{FIG}/fig_wilcoxon_heatmap.pdf"); plt.close(fig)
    o=json.load(open("stats_results.json"))["all"]; ranks=o["avg_rank"]; cd=o["CD"]
    items=sorted(ranks.items(),key=lambda kv:kv[1]); names=[a.replace("LLH:","") for a,_ in items]; rv=[b for _,b in items]
    k=len(METHODS); nL=(k+1)//2; axis_y=0.86
    fig,ax=plt.subplots(figsize=(13,5.2)); ax.set_xlim(0,k+1); ax.set_ylim(0,1); ax.axis("off")
    ax.plot([1,k],[axis_y,axis_y],"k",lw=1.3)
    for t in range(1,k+1):
        ax.plot([t,t],[axis_y,axis_y+0.028],"k",lw=1.1); ax.text(t,axis_y+0.062,str(t),ha="center",fontsize=10)
    ytop=0.72; dy=0.075
    for i,(nm,rk) in enumerate(zip(names,rv)):
        col=OI[i%12]
        if i<nL:
            row=i; y=ytop-row*dy; xend=0.6; xtxt=0.45; ha="right"
        else:
            row=k-1-i; y=ytop-row*dy; xend=k+0.4; xtxt=k+0.55; ha="left"
        ax.plot([rk,rk],[axis_y,y],color=col,lw=1.7)
        ax.plot([rk,xend],[y,y],color=col,lw=1.7)
        ax.text(xtxt,y,f"{nm} ({rk:.2f})",ha=ha,va="center",fontsize=10.5)
    ax.plot([1,1+cd],[axis_y+0.082,axis_y+0.082],"k",lw=4); ax.text(1+cd/2,axis_y+0.112,f"CD = {cd:.2f}",ha="center",fontsize=11.5)
    ax.set_title("Critical-difference diagram (Friedman + Nemenyi, all 12 problems)",fontsize=14,y=1.02)
    fig.savefig(f"{FIG}/fig_cd_diagram.png"); fig.savefig(f"{FIG}/fig_cd_diagram.pdf"); plt.close(fig)

def fig_radar():
    res={(r['problem'],r['method']):r for r in csv.DictReader(open("results.csv"))}
    o=json.load(open("stats_results.json"))["all"]; methods=["HH-UCB","HH-Random","DE","CMA-ES","LLH:VNS"]
    qual={m:o["avg_rank"][m] for m in methods}
    rel={m:np.mean([float(res[(p,m)]["succ_rate"]) for p in BENCH+ETCP]) for m in methods}
    rob={m:np.std([float(res[(p,m)]["mean"]) for p in BENCH+ETCP]) for m in methods}
    eff={m:np.mean([float(res[(p,m)]["mean_ett"]) for p in BENCH+ETCP if float(res[(p,m)]["mean_ett"])>0] or [1e9]) for m in methods}
    def norm(d,inv=False):
        v=np.array([d[m] for m in methods]); v=-v if inv else v; return {m:(v[i]-v.min())/(np.ptp(v)+1e-9) for i,m in enumerate(methods)}
    Q,Rl,E,Rb=norm(qual,True),norm(rel),norm(eff,True),norm(rob,True)
    cats=["Quality\n(rank)","Reliability\n(success)","Efficiency\n(speed)","Robustness\n(low spread)"]
    ang=np.linspace(0,2*np.pi,len(cats),endpoint=False); ang=np.r_[ang,ang[0]]
    fig,ax=plt.subplots(figsize=(7.6,7.6),subplot_kw=dict(polar=True))
    for m in methods:
        vals=[Q[m],Rl[m],E[m],Rb[m]]; vals=np.r_[vals,vals[0]]
        ax.plot(ang,vals,color=MCOL.get(m,'#444'),lw=2.4,label=m.replace("LLH:","")); ax.fill(ang,vals,color=MCOL.get(m,'#444'),alpha=0.07)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats,fontsize=12); ax.set_yticklabels([])
    ax.set_title("Method profile across quality, reliability, efficiency, robustness\n(descriptive only)",y=1.10,fontsize=14)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.08),ncol=5,fontsize=11,frameon=True)
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(f"{FIG}/fig_radar.png"); fig.savefig(f"{FIG}/fig_radar.pdf"); plt.close(fig)

def fig_sensitivity():
    s=json.load(open("sensitivity_results.json"))
    LAB={"hybrid_pair":"hybrid pair","base_fluid":"base fluid","w_pct":"weight frac.","comp1_share_pct":"share","V_Lmin":"flow rate","Ti_C":"inlet temp","Is_Wm2":"irradiance","Ta_C":"ambient"}
    order=["hybrid_pair","base_fluid","w_pct","comp1_share_pct","V_Lmin","Ti_C","Is_Wm2","Ta_C"]
    numeric=["w_pct","comp1_share_pct","V_Lmin","Ti_C","Is_Wm2","Ta_C"]
    fig,axes=plt.subplots(1,3,figsize=(18,5.2))
    ax=axes[0]; x=np.arange(len(order)); wd=0.38
    ax.bar(x-wd/2,[s["PEC"]["first_order"][f] for f in order],wd,color=OI[3],label="PEC")
    ax.bar(x+wd/2,[s["eta_thermal"]["first_order"][f] for f in order],wd,color=OI[0],label="efficiency")
    ax.set_xticks(x); ax.set_xticklabels([LAB[f] for f in order],rotation=40,ha="right",fontsize=11)
    ax.set_ylabel("first-order variance share"); ax.set_title("(a) First-order indices"); ax.legend(fontsize=12)
    ax=axes[1]
    for i,resp in enumerate(["PEC","eta_thermal"]):
        g=s[resp]["grouped"]; b=0
        for key,col in [("composition",OI[2]),("operating",OI[1]),("interaction",OI[4])]:
            ax.bar(i,g[key],bottom=b,color=col,label=key if i==0 else None); b+=g[key]
    ax.set_xticks([0,1]); ax.set_xticklabels(["PEC","efficiency"]); ax.set_ylabel("variance share")
    ax.set_ylim(0,1.42)
    ax.set_title("(b) Composition / operating / interaction")
    ax.legend(loc="upper center",ncol=3,fontsize=9.5,frameon=True,handlelength=1.2,columnspacing=1.0,borderaxespad=0.3)
    ax=axes[2]; mo=s["PEC"]["morris"]
    for f in numeric:
        ax.scatter(mo[f]["mu_star"],mo[f]["sigma"],color=OI[order.index(f)],s=90)
        ax.annotate(LAB[f],(mo[f]["mu_star"],mo[f]["sigma"]),fontsize=10.5,xytext=(4,4),textcoords="offset points")
    ax.set_xlabel("mu* (mean |elementary effect|)"); ax.set_ylabel("sigma"); ax.set_title("(c) Morris screening, ordinal factors (PEC)")
    fig.suptitle("Sensitivity-guided reduction: composition is involved in 95% of PEC variance; operating has a large total effect (0.75) via interaction",y=1.03,fontsize=13)
    fig.subplots_adjust(wspace=0.26,bottom=0.16)
    fig.savefig(f"{FIG}/fig_sensitivity_reduction.png"); fig.savefig(f"{FIG}/fig_sensitivity_reduction.pdf"); plt.close(fig)

def fig_landscape():
    p=EP.PECmax(40,1200,25); ws=np.linspace(0.25,3.0,60); Vs=np.linspace(0.5,6.0,60); Z=np.zeros((len(Vs),len(ws)))
    for i,V in enumerate(Vs):
        for j,w in enumerate(ws): Z[i,j]=p._pred([0.5,1.5,w,25,V],("PEC",))["PEC"]
    r=HH.HyperHeuristic(p,np.random.default_rng(3),controller='ucb',record=True).run(1200); pos=r["pos_log"]
    fig,ax=plt.subplots(figsize=(8.4,6.2)); cf=ax.contourf(ws,Vs,Z,levels=22,cmap="viridis")
    ax.plot(pos[:,2],pos[:,4],color="w",lw=1.2,alpha=0.6); ax.scatter(pos[:,2],pos[:,4],c=np.arange(len(pos)),cmap="autumn",s=14)
    t=json.load(open("etc_truth.json"))["ETC-PEC-op1"]; ax.scatter([t["w"]],[t["V"]],marker="*",s=460,color="red",edgecolor="w",zorder=6)
    ax.annotate("grid optimum (w=3%, V=2 L/min)",xy=(t["w"],t["V"]),xytext=(1.0,4.3),fontsize=12,color="white",fontweight="bold",
        arrowprops=dict(arrowstyle="->",color="white",lw=1.8),zorder=7)
    cb=fig.colorbar(cf,ax=ax); cb.set_label("PEC (surrogate)",fontsize=12)
    ax.set_xlabel("weight fraction w (%)"); ax.set_ylabel("flow rate V (L/min)")
    ax.set_title("PEC landscape (Al2O3-Cu, EG/water, share=25%, Ti=40C, Is=1200, Ta=25C)",fontsize=13)
    fig.savefig(f"{FIG}/fig_landscape_trajectory.png"); fig.savefig(f"{FIG}/fig_landscape_trajectory.pdf"); plt.close(fig)

def main():
    for fn in [fig_convergence,fig_selection_and_c