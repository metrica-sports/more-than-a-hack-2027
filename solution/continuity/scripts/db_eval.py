# Historical experiment script (see ../EXPERIMENTS.md for the commit it was run against).
# REPO = repository root; OUT = where the pickles / jsonl go (default: solution/cache/experiments).
import os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.environ.get("MTAH_EXP_DIR") or os.path.join(REPO, "solution", "cache", "experiments")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MTAH_CACHE_DIR", os.path.join(REPO, "solution", "cache"))
import sys, pickle, time; sys.path.insert(0,os.path.join(REPO,'solution')); sys.path.insert(0,os.path.join(REPO,'evaluation'))
import numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from possession import deadball
P=pickle.load(open(os.path.join(OUT,'P.pkl'),'rb'))
t=time.time(); S=deadball.training_set(P); print('features %.1fs'%(time.time()-t), S[0][0].shape)
pickle.dump(S,open(os.path.join(OUT,'S.pkl'),'wb'))
for i in range(2):
    (X,y),(Xt,yt)=S[1-i],S[i]
    m=deadball.fit(X,y); z=deadball.predict_logit(m,Xt)
    med=np.nanmedian(X,0); Xi=np.where(np.isnan(X),med,X); mu,sd=Xi.mean(0),Xi.std(0)+1e-6
    lr=LogisticRegression(C=0.3,max_iter=3000).fit(((Xi-mu)/sd)[::4],y[::4])
    zl=lr.decision_function((np.where(np.isnan(Xt),med,Xt)-mu)/sd)
    print('test g%d: prior %.2f  AUC HGB %.3f  logit %.3f  speed_now %.3f'%(i+1,yt.mean(),roc_auc_score(yt,z),roc_auc_score(yt,zl),roc_auc_score(yt,-np.nan_to_num(Xt[:,17]))))
    pr=z>0; tp=(pr&(yt==1)).sum(); print('   HGB @0: P %.2f R %.2f'%(tp/pr.sum(), tp/yt.sum()))
