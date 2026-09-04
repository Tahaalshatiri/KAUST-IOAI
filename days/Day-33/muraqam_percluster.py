# =======================================================================
# PER-CLUSTER (per-class) THRESHOLD TUNING
# Slots in AFTER the OOF-context block and the test-array loads:
#   requires: Mo,mgo,baseo,masko, gold_mh, idxo,nonco, doc_of_gap, n_gaps,
#             Mt,mgt,baset,maskt, W_PC,THR_PC (from fit_perclass),
#             build_S, best_cut, macro_f1, decode_S, MARKS, test_docs,_tbase,test
# Replaces the final `for name,(W,thr) in VAR.items(): write(...)` loop.
# =======================================================================
import numpy as np, pandas as pd

CLUSTERS = ["hadith_seerah","adab","kids","news","letter","poem"]
CIDX     = {c:i for i,c in enumerate(CLUSTERS)}
K        = len(CLUSTERS)
FIT_CL   = [CIDX["hadith_seerah"], CIDX["adab"], CIDX["kids"]]   # big enough to sweep
FIT_CLS  = [3, 5]                                                 # ! and ؛ only
MIN_POS  = 40           # need at least this many positive gaps in a cluster to fit a class
ITERS    = 3           # coordinate-ascent passes

CATDIR = "/kaggle/input/..."   # <-- point at train_categories.csv / test_categories.csv
tr_cat = pd.read_csv(f"{CATDIR}/train_categories.csv").set_index("id")["category"]
te_cat = pd.read_csv(f"{CATDIR}/test_categories.csv").set_index("id")["category"]

# cluster id per OOF gap (via its doc) and per test doc
cl_of_gap  = np.array([CIDX[tr_cat[di]] for di in doc_of_gap], np.int16)
cl_of_tdoc = np.array([CIDX[te_cat[d["di"]]] for d in test_docs], np.int16)

# ---------------------------------------------------------------------
# fit per-cluster cutoffs for one class, coordinate ascent on GLOBAL class-c F1.
# base_tp0/fp0 = fixed (non-candidate argmax) contribution for this class.
# other clusters' candidate contribution is recomputed each pass.
# ---------------------------------------------------------------------
def percluster_class(sc, gc, cl, tp0, fp0, P, thr_global, fit_cl, min_pos, iters):
    thr = np.full(K, thr_global, np.float32)
    cm  = [cl == k for k in range(K)]
    for _ in range(iters):
        for k in fit_cl:
            if int(gc[cm[k]].sum()) < min_pos:      # too few positives -> leave global
                continue
            tp, fp = tp0, fp0                        # fold in every OTHER cluster at current thr
            for j in range(K):
                if j == k: continue
                fire = cm[j] & (sc >= thr[j])
                tp += int((gc[fire] == 1).sum())
                fp += int((gc[fire] == 0).sum())
            f, t = best_cut(sc[cm[k]], gc[cm[k]].astype(np.int64), tp, fp, P)
            thr[k] = t
    return thr

def build_THR(models, mg, mask, base, gold, cl, W, thr_base, fit_cl, fit_cls, min_pos, iters):
    S   = build_S(models, mg, mask, W)
    idx = np.where(mask)[0]
    nonc= np.where(~mask)[0]
    THR = np.tile(thr_base.reshape(7,1), (1, K)).astype(np.float32)   # (7,K) start = per-class global
    for c in fit_cls:
        sc  = S[idx, c]; gc = gold[idx, c].astype(np.int64); cl_c = cl[idx]
        tp0 = int((base[nonc, c].astype(bool) &  gold[nonc, c].astype(bool)).sum())
        fp0 = int((base[nonc, c].astype(bool) & ~gold[nonc, c].astype(bool)).sum())
        P   = int(gold[:, c].sum())
        THR[c] = percluster_class(sc, gc, cl_c, tp0, fp0, P, thr_base[c], fit_cl, min_pos, iters)
    # ---- tiny-cluster RULES (folded in as threshold overrides) ----
    THR[5, CIDX["letter"]] = np.inf                       # drop ؛ in letters (rare/noise)
    THR[5, CIDX["news"]]   = np.inf                       # drop ؛ in news
    for c in [1,2,4,5,6]:                                 # poem: keep only ! and . (…) 
        THR[c, CIDX["poem"]] = np.inf
    return THR, S

def decode_pc(S, mask, base, THR, cl):
    pred = base.copy(); idx = np.where(mask)[0]
    thr_gap = THR[:, cl[idx]].T                           # (len(idx),7)
    pred[idx] = (S[idx] >= thr_gap).astype(np.int8)
    return pred

def cofire(pred, a=1, b=5):                               # {،؛} artifact count
    return int((pred[:, a].astype(bool) & pred[:, b].astype(bool)).sum())

def class_f1(pred, gold, c):
    p = pred[:, c].astype(bool); g = gold[:, c].astype(bool)
    tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
    return float(2*tp / max(2*tp + fp + fn, 1e-9))

# =======================================================================
# OOF: fit full per-cluster THR, compare to global per-class baseline
# =======================================================================
THR_PCL, So = build_THR(Mo, mgo, masko, baseo, gold_mh, cl_of_gap,
                        W_PC, THR_PC, FIT_CL, FIT_CLS, MIN_POS, ITERS)

pred_g  = decode_S(So, masko, baseo, THR_PC)             # global per-class
pred_pc = decode_pc(So, masko, baseo, THR_PCL, cl_of_gap)
mg_, _  = macro_f1(pred_g,  gold_mh)
mp_, _  = macro_f1(pred_pc, gold_mh)
print("OOF  per-class(global)  macro %.4f   !=%.3f ؛=%.3f  {،؛}=%d"
      % (mg_, class_f1(pred_g,gold_mh,3), class_f1(pred_g,gold_mh,5), cofire(pred_g)))
print("OOF  per-cluster        macro %.4f   !=%.3f ؛=%.3f  {،؛}=%d   Δ%+.4f"
      % (mp_, class_f1(pred_pc,gold_mh,3), class_f1(pred_pc,gold_mh,5), cofire(pred_pc), mp_-mg_))
print("per-cluster THR  (rows=marks, cols=%s):" % " ".join(CLUSTERS))
for c in FIT_CLS:
    print("  %s  " % MARKS[c] + "  ".join(("inf" if np.isinf(THR_PCL[c,k]) else "%.3f"%THR_PCL[c,k]) for k in range(K)))

# =======================================================================
# HONEST HELD-OUT: fit on half the docs, score other half. Gate on ! and ؛.
# =======================================================================
rng = np.random.RandomState(0); dperm = rng.permutation(len(docs))
dA  = set(dperm[:len(docs)//2].tolist())
inA = np.array([doc_of_gap[j] in dA for j in range(n_gaps)])
print("\nheld-out (fit half docs / score other half):")
d3 = d5 = dm = 0.0
for label, (fitm, evm) in [("A->B",(inA,~inA)), ("B->A",(~inA,inA))]:
    fit_mask = masko & fitm
    THR_h, _ = build_THR(Mo, mgo, fit_mask, baseo, gold_mh, cl_of_gap,
                         W_PC, THR_PC, FIT_CL, FIT_CLS, MIN_POS, ITERS)
    ev = np.where(evm)[0]
    pg = decode_S(So, masko, baseo, THR_PC)[ev]
    pp = decode_pc(So, masko, baseo, THR_h, cl_of_gap)[ev]
    g  = gold_mh[ev]
    dmac = (macro_f1(pp,g)[0] - macro_f1(pg,g)[0])
    d_ex = class_f1(pp,g,3) - class_f1(pg,g,3)
    d_se = class_f1(pp,g,5) - class_f1(pg,g,5)
    d3+=d_ex; d5+=d_se; dm+=dmac
    print("  %s  macro Δ%+.4f   ! Δ%+.4f   ؛ Δ%+.4f" % (label, dmac, d_ex, d_se))
GATE = (d3 >= 0) and (d5 >= 0)     # non-negative out-of-fold on both weak classes
print("  gate:  ! Σ%+.4f  ؛ Σ%+.4f  macro Σ%+.4f  ->  %s"
      % (d3, d5, dm, "SHIP per-cluster" if GATE else "KEEP global per-class"))

# =======================================================================
# WRITE test: per-cluster if gate passes else global per-class (+ tiny rules)
# =======================================================================
def write(pred, fname):
    rows=[]
    for d in test_docs:
        out=[w+"".join(MARKS[k] for k in range(7) if pred[_tbase[d["di"]]+j][k]) for j,w in enumerate(d["words"])]
        rows.append(dict(id=d["id"], final_text=" ".join(out)))
    sub=pd.DataFrame(rows); bad=0
    for (_,rr),(_,tt) in zip(sub.iterrows(), test.iterrows()):
        s=rr.final_text
        for mk in set(MARKS): s=s.replace(mk," ")
        bad += " ".join(s.split()) != " ".join(str(tt.text).split())
    sub.to_csv(fname, index=False)
    print("%-28s mism %d | hedges %d | marks {%s}"
          % (fname, bad, int((pred.sum(1)>=2).sum()),
             ", ".join("%s:%d"%(MARKS[i],int(pred[:,i].sum())) for i in range(7))))

# test THR uses SAME fitted cutoffs; expand cluster->gap via per-doc cluster
cl_of_tgap = np.concatenate([[cl_of_tdoc[d["di"]]]*d["n"] for d in test_docs]).astype(np.int16)
St = build_S(Mt, mgt, maskt, W_PC)
print()
write(decode_S(St, maskt, baset, THR_PC), "submission-perclass.csv")
THR_ship = THR_PCL if GATE else np.tile(THR_PC.reshape(7,1),(1,K)).astype(np.float32)
# keep tiny-cluster rules even when gate fails on the fitted classes:
THR_ship[5, CIDX["letter"]] = np.inf; THR_ship[5, CIDX["news"]] = np.inf
for c in [1,2,4,5,6]: THR_ship[c, CIDX["poem"]] = np.inf
write(decode_pc(St, maskt, baset, THR_ship, cl_of_tgap), "submission-percluster.csv")
# NOTE: reconcile poem rule with your existing 101/198 override — this generalizes
# it to *all* poem-cluster test docs (2 of them). Re-apply your override on top if it differs.
