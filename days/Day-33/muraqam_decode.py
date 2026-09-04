import numpy as np, pandas as pd

NPY  = "/kaggle/input/datasets/tahaalshatiri/oof-try-correct-illegals/dl"
DATA = "/kaggle/input/competitions/muraqqamchallenge"
CATDIR = "/kaggle/input/muraqam-categories"   # <-- dataset holding train_categories.csv / test_categories.csv
S1B, S1L = "s1_base_softmax_94d83ed7", "s1_large_softmax_1d001f87"
REF = ["ref_base_mask_8e1145d0_33686967","ref_large_mask_8e1145d0_d25ff28b","ref_modern_mask_8e1145d0_80c6a50a"]
S3  = "s3_base_mask_8e1145d0_11f91656_33686967"
L = lambda stem,sp: np.load(f"{NPY}/{stem}_{sp}.npy")

W_LARGE = 0.50
ALPHA   = 0.9
RAW_W   = np.array([0.32,0.32,0.16,0.20], np.float32)     # [rb,rl,rm,s3]
THRESH0 = np.array([0.6,0.5,0.7,0.65,0.8,0.6,0.65], np.float32)

MARKS=[".","،","؟","!",":","؛","-"]; MARKSET=set(MARKS)
CLASSES=["",".","،","؟","!",":","؛","-","؟!"]
CLS_MH=np.zeros((9,7),np.int8)
for ci,c in enumerate(CLASSES):
    for ch in c: CLS_MH[ci,MARKS.index(ch)]=1

def parse_gaps(text, final):
    w=text.split(); g=[""]*len(w); wi=ci=0
    for ch in final:
        if wi<len(w) and ci<len(w[wi]) and ch==w[wi][ci]:
            ci+=1
            if ci==len(w[wi]): wi+=1; ci=0
        elif ch in MARKSET and wi>0: g[wi-1]+=ch
    return w,g
def gap_mh(g):
    v=np.zeros(7,np.int8)
    for ch in g:
        if ch in MARKSET: v[MARKS.index(ch)]=1
    return v

train=pd.read_csv(f"{DATA}/train.csv").dropna(subset=["text","final_text"]).reset_index(drop=True)
test =pd.read_csv(f"{DATA}/test.csv")
docs=[]
for di,r in enumerate(train.itertuples()):
    w,g=parse_gaps(r.text,r.final_text); docs.append(dict(di=di,words=w,n=len(w),mh=np.stack([gap_mh(x) for x in g])))
test_docs=[dict(di=ti,words=str(r.text).split(),id=r.id,n=len(str(r.text).split())) for ti,r in enumerate(test.itertuples())]
gap_base=np.cumsum([0]+[d["n"] for d in docs]); n_gaps=int(gap_base[-1])
_tbase=np.cumsum([0]+[d["n"] for d in test_docs])
gold_mh=np.concatenate([d["mh"] for d in docs])
doc_of_gap=np.concatenate([[d["di"]]*d["n"] for d in docs])
assert n_gaps==233334, n_gaps

def s1(sp):   return (1-W_LARGE)*L(S1B,sp)+W_LARGE*L(S1L,sp)
def marg(P):  return P@CLS_MH.astype(np.float32)
def s1arg(P): return CLS_MH[P.argmax(1)].astype(np.int8)
def load_models(sp):
    refs=[L(s,sp) for s in REF]; s3=L(S3,sp)
    mask=~np.isnan(refs[0][:,0])
    for r in refs[1:]: mask&=~np.isnan(r[:,0])
    mask&=~np.isnan(s3[:,0])
    return [np.nan_to_num(x) for x in refs+[s3]], mask

def macro_f1(pred, gold):
    p=pred.astype(bool); g=gold.astype(bool)
    tp=(p&g).sum(0).astype(float); fp=(p&~g).sum(0); fn=(~p&g).sum(0)
    f1=2*tp/np.maximum(2*tp+fp+fn,1e-9); return float(f1.mean()), f1

def wgrid4(step=0.2):
    k=round(1/step); out=[]
    for a in range(k+1):
        for b in range(k+1-a):
            for c in range(k+1-a-b):
                out.append(np.array([a,b,c,k-a-b-c],np.float32)/k)
    return out
WG=wgrid4(0.2)

def best_cut(sc, gc, tp0, fp0, P):
    order=np.argsort(-sc,kind="stable"); gs=gc[order].astype(np.int64)
    csum=np.cumsum(gs); k=np.arange(1,len(sc)+1)
    tp=tp0+csum; fp=fp0+(k-csum); fn=P-tp
    f1=2*tp/np.maximum(2*tp+fp+fn,1e-9)
    f0=2*tp0/max(2*tp0+fp0+(P-tp0),1e-9)
    bi=int(np.argmax(f1))
    return (f0,np.inf) if f0>=f1[bi] else (float(f1[bi]), float(sc[order][bi]))

def build_S(models, mg, mask, W):
    idx=np.where(mask)[0]; S=np.zeros((len(mg),7),np.float32)
    for c in range(7):
        Mc=np.stack([M[idx,c] for M in models])
        S[idx,c]=ALPHA*(W[c]@Mc)+(1-ALPHA)*mg[idx,c]
    return S
def decode_S(S, mask, base, thr):
    pred=base.copy(); idx=np.where(mask)[0]
    for c in range(7): pred[idx,c]=(S[idx,c]>=thr[c]).astype(np.int8)
    return pred

# ---- fitters operate on a chosen subset of gap indices (for held-out eval) ----
def base_stats(base, gold, nonc_idx, all_idx):
    TP0=np.array([ (base[nonc_idx,c].astype(bool)& gold[nonc_idx,c].astype(bool)).sum() for c in range(7)])
    FP0=np.array([ (base[nonc_idx,c].astype(bool)&~gold[nonc_idx,c].astype(bool)).sum() for c in range(7)])
    Pc =np.array([ gold[all_idx,c].sum() for c in range(7)])
    return TP0,FP0,Pc

def fit_thresh(models,mg,base,gold,cand,nonc,allidx,W):
    TP0,FP0,Pc=base_stats(base,gold,nonc,allidx)
    S=build_S(models,mg,mask_from(cand,len(mg)),W)
    thr=np.zeros(7,np.float32)
    for c in range(7):
        _,t=best_cut(S[cand,c], gold[cand,c].astype(np.int64), TP0[c],FP0[c],Pc[c]); thr[c]=t
    return thr
def fit_perclass(models,mg,base,gold,cand,nonc,allidx):
    TP0,FP0,Pc=base_stats(base,gold,nonc,allidx)
    W=np.zeros((7,4),np.float32); thr=np.zeros(7,np.float32)
    Mstack=[np.stack([M[cand,c] for M in models]) for c in range(7)]; mc=[mg[cand,c] for c in range(7)]
    for c in range(7):
        bestf=-1
        for w in WG:
            sc=ALPHA*(w@Mstack[c])+(1-ALPHA)*mc[c]
            f,t=best_cut(sc, gold[cand,c].astype(np.int64), TP0[c],FP0[c],Pc[c])
            if f>bestf: bestf,bw,bt=f,w,t
        W[c]=bw; thr[c]=bt
    return W,thr
def mask_from(idx,n):
    m=np.zeros(n,bool); m[idx]=True; return m

# =======================================================================
# OOF context
# =======================================================================
Po=s1("oof"); mgo=marg(Po); baseo=s1arg(Po); Mo,masko=load_models("oof")
idxo=np.where(masko)[0]; nonco=np.where(~masko)[0]; alli=np.arange(n_gaps)
RAWW7=np.tile(RAW_W,(7,1))

# ---- 1) full-OOF fits (what you'll actually deploy) ----
THR_T  = fit_thresh(Mo,mgo,baseo,gold_mh,idxo,nonco,alli,RAWW7)
W_PC,THR_PC = fit_perclass(Mo,mgo,baseo,gold_mh,idxo,nonco,alli)

VAR={"raw":(RAWW7,THRESH0), "thresh":(RAWW7,THR_T), "perclass":(W_PC,THR_PC)}
print(f"{'variant':<10}{'macro':>8} | {' '.join(MARKS)}")
oof={}
for name,(W,thr) in VAR.items():
    S=build_S(Mo,mgo,masko,W); pred=decode_S(S,masko,baseo,thr)
    m,pc=macro_f1(pred,gold_mh); oof[name]=m
    print(f"{name:<10}{m:>8.4f} | "+" ".join(f"{pc[k]:.3f}" for k in range(7))
          + ("   (base)" if name=="raw" else f"   Δ{m-oof['raw']:+.4f}"))

# =======================================================================
# HONEST CHECK — fit on half the docs, score the other half (is the gain real?)
# =======================================================================
rng=np.random.RandomState(0); dperm=rng.permutation(len(docs))
dA=set(dperm[:len(docs)//2].tolist())
inA=np.array([doc_of_gap[j] in dA for j in range(n_gaps)])
print("\nheld-out (fit on half the docs, score other half):")
for label,(fitA,fitB) in [("A->B",(inA,~inA)),("B->A",(~inA,inA))]:
    for name,fitter in [("thresh","t"),("perclass","p")]:
        candF=np.where(masko&fitA)[0]; noncF=np.where(~masko&fitA)[0]; allF=np.where(fitA)[0]
        if fitter=="t": W,thr=RAWW7, fit_thresh(Mo,mgo,baseo,gold_mh,candF,noncF,allF,RAWW7)
        else:           W,thr=fit_perclass(Mo,mgo,baseo,gold_mh,candF,noncF,allF)
        ev=np.where(fitB)[0]
        S=build_S(Mo,mgo,masko,W); pred=decode_S(S,masko,baseo,thr)
        m,_=macro_f1(pred[ev],gold_mh[ev])
        mr,_=macro_f1(decode_S(build_S(Mo,mgo,masko,RAWW7),masko,baseo,THRESH0)[ev],gold_mh[ev])
        print(f"  {label} {name:<9} eval-half macro {m:.4f}  (raw {mr:.4f}, Δ{m-mr:+.4f})")

# =======================================================================
# PER-CLUSTER FULL FIT  —  the whole perclass tuning (WEIGHTS + thresholds)
# re-run independently on each big cluster's gaps. tiny clusters keep the
# global (W_PC, THR_PC) plus a couple of rules.
# =======================================================================
CLUSTERS = ["hadith_seerah","adab","kids","news","letter","poem"]
CIDX     = {c:i for i,c in enumerate(CLUSTERS)}; K=len(CLUSTERS)
FIT_CL   = [CIDX["hadith_seerah"], CIDX["adab"], CIDX["kids"]]
MIN_GAPS = 1500     # below this many candidate gaps, don't fit — keep global

tr_cat = pd.read_csv(f"{CATDIR}/train_categories.csv").set_index("id")["category"]
te_cat = pd.read_csv(f"{CATDIR}/test_categories.csv").set_index("id")["category"]
cl_of_gap  = np.array([CIDX[tr_cat[di]] for di in doc_of_gap], np.int16)
cl_of_tdoc = np.array([CIDX[te_cat[d["di"]]] for d in test_docs], np.int16)
cl_of_tgap = np.concatenate([[cl_of_tdoc[d["di"]]]*d["n"] for d in test_docs]).astype(np.int16)

def fit_clusters(models, mg, base, gold, cl, mask, scope, W_base, thr_base):
    # scope = boolean over all gaps limiting which docs are visible (for held-out).
    # each cluster runs the SAME fit_perclass -> its own weights AND thresholds.
    Wbyk = [W_base.copy() for _ in range(K)]
    THR  = np.tile(thr_base.reshape(7,1), (1,K)).astype(np.float32)
    for k in FIT_CL:
        cand = np.where(mask & (cl==k) & scope)[0]
        if len(cand) < MIN_GAPS: continue
        nonc   = np.where((~mask) & (cl==k) & scope)[0]
        allidx = np.where((cl==k) & scope)[0]
        W, thr = fit_perclass(models, mg, base, gold, cand, nonc, allidx)
        Wbyk[k] = W; THR[:,k] = thr
    THR[5, CIDX["letter"]] = np.inf                     # rule: drop ؛ in letters
    THR[5, CIDX["news"]]   = np.inf                     # rule: drop ؛ in news
    for c in [1,2,4,5,6]: THR[c, CIDX["poem"]] = np.inf # rule: poem keeps only ! and .
    return Wbyk, THR

def build_S_pc(models, mg, mask, cl, Wbyk):
    # per-cluster weights: each gap scored with its cluster's W. clusters are disjoint -> sum.
    S = np.zeros((len(mg),7), np.float32)
    for k in range(K):
        S += build_S(models, mg, mask & (cl==k), Wbyk[k])
    return S

def decode_pc(S, mask, base, THR, cl):
    pred = base.copy(); idx = np.where(mask)[0]
    pred[idx] = (S[idx] >= THR[:, cl[idx]].T).astype(np.int8)
    return pred

def class_f1(pred, gold, c):
    p=pred[:,c].astype(bool); g=gold[:,c].astype(bool)
    tp=(p&g).sum(); fp=(p&~g).sum(); fn=(~p&g).sum()
    return float(2*tp/max(2*tp+fp+fn,1e-9))
def cofire(pred, a=1, b=5): return int((pred[:,a].astype(bool)&pred[:,b].astype(bool)).sum())

allmask = np.ones(n_gaps, bool)
So      = build_S(Mo, mgo, masko, W_PC)                 # global perclass score (baseline)
Wbyk, THR_PCL = fit_clusters(Mo, mgo, baseo, gold_mh, cl_of_gap, masko, allmask, W_PC, THR_PC)
So_pc   = build_S_pc(Mo, mgo, masko, cl_of_gap, Wbyk)

pred_g  = decode_S(So, masko, baseo, THR_PC)
pred_pc = decode_pc(So_pc, masko, baseo, THR_PCL, cl_of_gap)
mg_,_=macro_f1(pred_g,gold_mh); mp_,_=macro_f1(pred_pc,gold_mh)
print("\nper-class(global) OOF macro %.4f  !=%.3f ؛=%.3f  {،؛}=%d"
      % (mg_, class_f1(pred_g,gold_mh,3), class_f1(pred_g,gold_mh,5), cofire(pred_g)))
print("per-cluster full  OOF macro %.4f  !=%.3f ؛=%.3f  {،؛}=%d   Δ%+.4f"
      % (mp_, class_f1(pred_pc,gold_mh,3), class_f1(pred_pc,gold_mh,5), cofire(pred_pc), mp_-mg_))
print("per-cluster ! / ؛ thresholds (cols=%s):" % " ".join(CLUSTERS))
for c in [3,5]:
    print("  %s  "%MARKS[c] + "  ".join(("inf" if np.isinf(THR_PCL[c,k]) else "%.3f"%THR_PCL[c,k]) for k in range(K)))

# ---- honest held-out: fit per-cluster (weights+thr) on half the docs, score other half ----
print("\nheld-out per-cluster full fit (fit half / score other half):")
dm=d3=d5=0.0
for label,(fitm,evm) in [("A->B",(inA,~inA)),("B->A",(~inA,inA))]:
    Wbyk_h, THR_h = fit_clusters(Mo,mgo,baseo,gold_mh,cl_of_gap,masko,fitm,W_PC,THR_PC)
    S_h = build_S_pc(Mo,mgo,masko,cl_of_gap,Wbyk_h)     # weights from fit half, applied to all gaps
    ev=np.where(evm)[0]; g=gold_mh[ev]
    pg=decode_S(So,masko,baseo,THR_PC)[ev]
    pp=decode_pc(S_h,masko,baseo,THR_h,cl_of_gap)[ev]
    dmac=macro_f1(pp,g)[0]-macro_f1(pg,g)[0]; d_ex=class_f1(pp,g,3)-class_f1(pg,g,3); d_se=class_f1(pp,g,5)-class_f1(pg,g,5)
    dm+=dmac; d3+=d_ex; d5+=d_se
    print("  %s  macro Δ%+.4f   ! Δ%+.4f   ؛ Δ%+.4f" % (label,dmac,d_ex,d_se))
print("  held-out sum: macroΣ%+.4f  !Σ%+.4f  ؛Σ%+.4f  ->  %s"
      % (dm,d3,d5,"generalizes" if dm>=0 else "OOF-only gain (ship on LB judgement)"))

# =======================================================================
# decode + write test
# =======================================================================
Pt=s1("test"); mgt=marg(Pt); baset=s1arg(Pt); Mt,maskt=load_models("test")
def write(pred,fname):
    rows=[]
    for d in test_docs:
        out=[w+"".join(MARKS[k] for k in range(7) if pred[_tbase[d["di"]]+j][k]) for j,w in enumerate(d["words"])]
        rows.append(dict(id=d["id"],final_text=" ".join(out)))
    sub=pd.DataFrame(rows); bad=0
    for (_,rr),(_,tt) in zip(sub.iterrows(),test.iterrows()):
        s=rr.final_text
        for mk in MARKSET: s=s.replace(mk," ")
        bad+= " ".join(s.split())!=" ".join(str(tt.text).split())
    sub.to_csv(fname,index=False)
    print(f"{fname:<28} mism {bad} | hedges {int((pred.sum(1)>=2).sum())} | "
          f"marks {{{', '.join(f'{MARKS[i]}:{int(pred[:,i].sum())}' for i in range(7))}}}")

print()
for name,(W,thr) in VAR.items():
    St=build_S(Mt,mgt,maskt,W)
    write(decode_S(St,maskt,baset,thr), f"submission-{name}.csv")

# ---- per-cluster full-fit test decode (per-cluster weights + per-cluster thresholds) ----
St_pc = build_S_pc(Mt, mgt, maskt, cl_of_tgap, Wbyk)
write(decode_pc(St_pc, maskt, baset, THR_PCL, cl_of_tgap), "submission-percluster.csv")
# NOTE: poem rule generalises your 101/198 override to all poem-cluster test docs; re-apply on top if yours differs.
