# Historical experiment script (see ../EXPERIMENTS.md for the commit it was run against).
# REPO = repository root; OUT = where the pickles / jsonl go (default: solution/cache/experiments).
import os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("MTAH_EXP_DIR") or os.path.join(REPO, "solution", "cache", "experiments")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
import sys, json, pickle; sys.path.insert(0,os.path.join(REPO,'solution'))
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from possession import calibrate, gaps
P=pickle.load(open(os.path.join(OUT,'P.pkl'),'rb')); sets=gaps.training_set(P)
P2=[]
for i,(G,f,gt,gl) in enumerate(P):
    X,y=sets[1-i]; h=HGB(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,random_state=0).fit(X,y)
    idx,Xt=gaps.gap_features(f,G['ball'],calibrate.blocks_of(gt)); pr=np.clip(h.predict_proba(Xt)[:,1],1e-3,1-1e-3)
    z=np.zeros(G['n_frames'],np.float32); p0=y.mean(); z[idx]=np.clip(np.log(pr/(1-pr))-np.log(p0/(1-p0)),-4,4)
    P2.append((G,f,gt,z))
R=[json.loads(l) for l in open(os.path.join(OUT,'paired.jsonl'))]
top=sorted(R,key=lambda r:-min(r['off']))[:20]
def run(r):
    cfg=calibrate.make_cfg(r['d']['switch'],r['d']['min_pos'],r['d']['min_dead'])
    return [(k, calibrate.score(P2,dict(r['p'],k_gap=k),cfg)[1]) for k in (0.15,0.4)]
with Pool(2) as pool: out=pool.map(run, top)
for r,o in zip(top,out): print([round(x,3) for x in r['off']], [(k,[round(x,3) for x in s]) for k,s in o])
off=np.array([min(r['off']) for r in top])
for j,k in enumerate((0.15,0.4)):
    on=np.array([min(o[j][1]) for o in out]); print('k',k,'worse-game delta mean %.4f, max ON %.3f vs max OFF %.3f'%((on-off).mean(),on.max(),off.max()))
