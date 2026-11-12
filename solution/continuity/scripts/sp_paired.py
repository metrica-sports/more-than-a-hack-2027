# Historical experiment script (see ../EXPERIMENTS.md for the commit it was run against).
# REPO = repository root; OUT = where the pickles / jsonl go (default: solution/cache/experiments).
import os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("MTAH_EXP_DIR") or os.path.join(REPO, "solution", "cache", "experiments")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
import sys, json, pickle, os; sys.path.insert(0,os.path.join(REPO,'solution'))
import numpy as np, warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
from possession import calibrate
PK=os.path.join(OUT,'P2.pkl')
if not os.path.exists(PK): pickle.dump(calibrate.prepare(seed=0), open(PK,'wb'))
P=pickle.load(open(PK,'rb'))
def run(i):
    p,d=calibrate.sample(np.random.default_rng([13,i])); cfg=calibrate.make_cfg(d['switch'],d['min_pos'],d['min_dead'])
    on=calibrate.score(P,p,cfg)[1]; off=calibrate.score(P,dict(p,k_sp_dead=0.0,k_sp_team=0.0),cfg)[1]
    return {'i':i,'p':p,'d':d,'on':on,'off':off}
if __name__=='__main__':
    N=int(sys.argv[1])
    with Pool(2) as pool, open(os.path.join(OUT,'sp_paired.jsonl'),'w') as fh:
        for r in pool.imap_unordered(run, range(N)): fh.write(json.dumps(r)+'\n'); fh.flush()
