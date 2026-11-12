# Historical experiment script (see ../EXPERIMENTS.md for the commit it was run against).
# REPO = repository root; OUT = where the pickles / jsonl go (default: solution/cache/experiments).
import os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("MTAH_EXP_DIR") or os.path.join(REPO, "solution", "cache", "experiments")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
import sys, json, pickle; sys.path.insert(0,os.path.join(REPO,'solution')); sys.path.insert(0,os.path.join(REPO,'evaluation'))
import numpy as np, possession_metrics as pm
from multiprocessing import Pool
from possession import calibrate, emissions, decode
P=pickle.load(open(os.path.join(OUT,'P.pkl'),'rb'))
R=[json.loads(l) for l in open(os.path.join(OUT,'paired.jsonl'))]
top=sorted(R,key=lambda r:-min(r['off']))[:6]
def ev(r):
    cfg=calibrate.make_cfg(r['d']['switch'],r['d']['min_pos'],r['d']['min_dead']); res=[]
    for k in (0.0,0.3):
        p=dict(r['p'],k_gap=k); mf=[];df=[]
        for G,f,gt,gl in P:
            ll=emissions.tracking_scores(f,p)+emissions.gap_scores(gl,p)
            pred=decode.decode_match(ll,calibrate.blocks_of(gt),G['n_frames'],cfg)
            cm=pm.scored_confusion(pm.confusion(pm.label_codes(gt),pm.label_codes(pred)))
            mf.append(pm.macro_f1(cm)); df.append(pm.per_class_prf(cm,2))
        res.append((k,np.round(mf,3).tolist(),[round(x[2] if isinstance(x,tuple) else x.get('f1',0),3) for x in df]))
    return res
with Pool(2) as pool:
    for o in pool.map(ev,top): print(o)
