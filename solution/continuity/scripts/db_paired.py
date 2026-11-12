# Historical experiment script (see ../EXPERIMENTS.md for the commit it was run against).
# REPO = repository root; OUT = where the pickles / jsonl go (default: solution/cache/experiments).
import os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("MTAH_EXP_DIR") or os.path.join(REPO, "solution", "cache", "experiments")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
import sys, json, pickle; sys.path.insert(0,os.path.join(REPO,'solution'))
sys.path.insert(0,os.path.join(REPO,'evaluation'))
import numpy as np, warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from possession import calibrate, deadball, emissions, decode
P=pickle.load(open(os.path.join(OUT,'P.pkl'),'rb')); S=pickle.load(open(os.path.join(OUT,'S.pkl'),'rb'))
DL={'hgb':[],'lr':[]}
for i,(G,f,gt,gl) in enumerate(P):
    (X,y)=S[1-i]; Xa=deadball.frame_features(f,G)
    DL['hgb'].append(deadball.predict_logit(deadball.fit(X,y),Xa))
    med=np.nanmedian(X,0); Xi=np.where(np.isnan(X),med,X); mu,sd=Xi.mean(0),Xi.std(0)+1e-6
    lr=LogisticRegression(C=0.3,max_iter=3000).fit(((Xi-mu)/sd)[::4],y[::4])
    z=lr.decision_function((np.where(np.isnan(Xa),med,Xa)-mu)/sd)-np.log(y.mean()/(1-y.mean()))
    DL['lr'].append(np.clip(z,-4,4).astype(np.float32))
pickle.dump(DL,open(os.path.join(OUT,'DL.pkl'),'wb'))
def score(p,cfg,dl):
    s=[]
    import possession_metrics as pm
    for (G,f,gt,gl),d in zip(P,dl):
        ll=emissions.tracking_scores(f,p); ll[:,2]+=p['k_dead']*d
        pred=decode.decode_match(ll,calibrate.blocks_of(gt),G['n_frames'],cfg)
        g,pr=pm.label_codes(gt),pm.label_codes(pred); s.append(pm.spell_scores(g,pr,pm.valid_runs(g))['mean_f1'])
    return s
def run(i):
    rng=np.random.default_rng([11,i]); p,d=calibrate.sample(rng); p['k_gap']=0; p['k_dead']=float(rng.uniform(0,0.8))
    cfg=calibrate.make_cfg(d['switch'],d['min_pos'],d['min_dead'])
    return {'i':i,'p':p,'d':d,'lr':score(p,cfg,DL['lr']),'hgb':score(p,cfg,DL['hgb']),'off':score(dict(p,k_dead=0),cfg,DL['lr'])}
if __name__=='__main__':
    N=int(sys.argv[1])
    with Pool(2) as pool, open(os.path.join(OUT,'db_paired.jsonl'),'w') as fh:
        for r in pool.imap_unordered(run, range(N)): fh.write(json.dumps(r)+'\n'); fh.flush()
