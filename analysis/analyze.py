import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import json,zipfile,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import argparse
PARSER=argparse.ArgumentParser(description='Reproduce the Escape the Feed analysis')
PARSER.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/ml-32m.zip')
PARSER.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results')
ARGS=PARSER.parse_args()
ROOT=ARGS.output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
ZIP=ARGS.data.resolve()

SEED=20261018
rng=np.random.default_rng(SEED)
CUT=1672531200 # 2023-01-01 UTC
start=time.time()
def log(*x):print(*x,flush=True)
with zipfile.ZipFile(ZIP) as z:
    movies=pd.read_csv(z.open('ml-32m/movies.csv'))
    expected=dict((line.split()[1],line.split()[0]) for line in z.read('ml-32m/checksums.txt').decode().splitlines())
    checks={}
    for name in ['ratings.csv','movies.csv']:
        h=hashlib.md5()
        with z.open('ml-32m/'+name) as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        checks[name]={'md5':h.hexdigest(),'expected':expected[name],'match':h.hexdigest()==expected[name]}
    assert all(x['match'] for x in checks.values())
    log('Checksums verified',time.time()-start)
    r=pd.read_csv(z.open('ml-32m/ratings.csv'),dtype={'userId':'int32','movieId':'int32','rating':'float32','timestamp':'int64'})
log('Loaded',len(r),'ratings',time.time()-start)
raw_n=len(r);raw_users=r.userId.nunique()
pre=r[r.timestamp<CUT].copy();del r
positive=pre[pre.rating>=4]
pop=positive.groupby('movieId').size()
ids=np.sort(pop[pop>=100].index.values)
movies=movies.set_index('movieId').loc[ids].reset_index()
ids=movies.movieId.to_numpy();M=len(ids)
idmap=pd.Series(np.arange(M),index=ids)
pre=pre[pre.movieId.isin(ids)].copy()
pre['item']=pre.movieId.map(idmap).astype('int32')
pos=pre[pre.rating>=4].sort_values(['userId','timestamp','movieId']).copy()
users=np.sort(pos.userId.unique());useridx=pd.Index(users).get_indexer(pos.userId)
X=sparse.csr_matrix((np.ones(len(pos),dtype=np.float32),(useridx,pos.item.to_numpy())),shape=(len(users),M))
# Column normalisation tempers raw item-frequency dominance before factorisation.
X=X @ sparse.diags(1/np.sqrt(np.asarray(X.sum(axis=0)).ravel()))
log('Training matrix',X.shape,X.nnz,time.time()-start)
svd=TruncatedSVD(n_components=32,n_iter=7,random_state=SEED)
svd.fit(X)
E=svd.components_.T.copy();E/=np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-12)
log('SVD fitted',time.time()-start)
genres=sorted(set('|'.join(movies.genres).split('|'))-{'(no genres listed)','IMAX'})
G=np.array([[g in s.split('|') for g in genres] for s in movies.genres],dtype=bool)
allowed=np.where(G.sum(axis=0)>=100)[0]
hist={int(u):g.item.to_numpy() for u,g in pos.groupby('userId',sort=False)}
seen={int(u):g.item.to_numpy() for u,g in pre.groupby('userId',sort=False)}
counts=pos.groupby('userId').size();eligible=counts[counts>=20].index.to_numpy()
order=rng.permutation(eligible)
records=[];full=[];recent=[];seens=[];signals=[];targets=[];selected_hist=[]
itempop=pop.loc[ids].to_numpy()
for u in order:
    h=hist[int(u)];s=seen[int(u)];f=E[h].sum(axis=0);rec=E[h[-20:]].sum(axis=0)
    base=[]
    for p in [f,rec]:
        score=E@p;score[s]=-np.inf
        top=np.argsort(-score,kind='stable')[:10]
        base.append(G[top].sum(axis=0))
    shares=G[h].mean(axis=0)
    choices=[g for g in allowed if shares[g]<=.10 and base[0][g]<3 and base[1][g]<3 and np.sum(G[:,g])-np.sum(G[s,g])>=10]
    if not choices:continue
    g=int(rng.choice(choices));pool=np.flatnonzero(G[:,g]);pool=np.setdiff1d(pool,s)
    sig=rng.choice(pool,size=10,replace=False)
    records.append({'userId':int(u),'history_n':len(h),'mainstream_score':float(np.log1p(itempop[h]).mean()),'target':genres[g],'target_history_share':float(shares[g])})
    full.append(f);recent.append(rec);seens.append(s);signals.append(sig);targets.append(g);selected_hist.append(h)
    if len(records)==2000:break
N=len(records);targets=np.array(targets);signals=np.array(signals)
log('Evaluation cohort',N,time.time()-start)
np.savez_compressed(ROOT/'model.npz',embeddings=E,movie_ids=ids)
rows=pd.DataFrame(records)
# Tertiles from the evaluation cohort's pre-intervention taste scores.
rows['taste_group']=pd.qcut(rows.mainstream_score,3,labels=['Less-mainstream','Middle','Mainstream'])
rows['length_group']=pd.cut(rows.history_n,bins=[19,50,100,250,np.inf],labels=['20-50','51-100','101-250','251+'])
counts_out={};tops_out={}
for name,profiles in [('Full history',full),('20-item reset',recent)]:
    P=np.array(profiles,dtype=np.float32);out=np.zeros((N,11),dtype=np.int16);tops=np.zeros((N,11,10),dtype=np.int32)
    for k in range(11):
        if k:P+=E[signals[:,k-1]]
        scores=P@E.T
        for i,s in enumerate(seens):
            scores[i,s]=-np.inf
            if k:scores[i,signals[i,:k]]=-np.inf
        top=np.argpartition(-scores,9,axis=1)[:,:10]
        for i in range(N):
            t=top[i];t=t[np.argsort(-scores[i,t],kind='stable')];tops[i,k]=t
            out[i,k]=G[t,targets[i]].sum()
            assert not np.intersect1d(t,seens[i]).size
            assert not np.intersect1d(t,signals[i,:k]).size
        log(name,'step',k,'reach-now',float((out[:,k]>=3).mean()),time.time()-start)
    counts_out[name]=out;tops_out[name]=tops
np.savez_compressed(ROOT/'trajectories.npz',full=counts_out['Full history'],recent=counts_out['20-item reset'],signals=signals,targets=targets,full_top=tops_out['Full history'],recent_top=tops_out['20-item reset'])
rows.to_csv(ROOT/'cohort.csv',index=False)
boot=np.random.default_rng(SEED+1).integers(0,N,(2000,N))
def ci(vals):return [float(x) for x in np.quantile(np.asarray(vals)[boot].mean(axis=1),[.025,.975])]
summary={'seed':SEED,'cutoff':'2023-01-01 UTC','raw_ratings':raw_n,'raw_users':raw_users,'catalog_movies':M,'training_users':len(users),'training_positive_ratings':len(pos),'training_ratings_catalog':len(pre),'eligible_history_users':len(eligible),'cohort_n':N,'checksums':checks,'explained_variance':float(svd.explained_variance_ratio_.sum()),'models':{},'target_counts':rows.target.value_counts().to_dict(),'history_median':float(rows.history_n.median())}
for name,out in counts_out.items():
    reached=np.maximum.accumulate(out,axis=1)>=3
    summary['models'][name]={'cumulative_success':reached.mean(axis=0).tolist(),'current_success':(out>=3).mean(axis=0).tolist(),'mean_target_slots':out.mean(axis=0).tolist(),'success5_count':int(reached[:,5].sum()),'success5_ci':ci(reached[:,5]),'thresholds5':{str(t):float((out[:,:6].max(axis=1)>=t).mean()) for t in [3,4]},'groups':{}}
    rows[name+' success5']=reached[:,5]
    for field in ['taste_group','length_group']:
        summary['models'][name]['groups'][field]={str(g):{'n':int(len(ix)),'success5':float(reached[ix,5].mean()),'history_median':float(rows.loc[ix,'history_n'].median())} for g,ix in rows.groupby(field,observed=True).groups.items()}
a=(counts_out['Full history'][:,:6].max(axis=1)>=3).astype(float)
b=(counts_out['20-item reset'][:,:6].max(axis=1)>=3).astype(float)
summary['paired_gain5']=float((b-a).mean());summary['paired_gain5_ci']=ci(b-a)
# A reproducible contrast example: recent succeeds by 5, full has not; median history among such cases.
mask=(b==1)&(a==0);options=np.where(mask)[0]
ix=int(options[np.argsort(rows.loc[options,'history_n'].to_numpy())[len(options)//2]]) if len(options) else 0
example=records[ix].copy();example['cohort_index']=ix
example['signals']=[{'movieId':int(ids[j]),'title':movies.title.iloc[j],'genres':movies.genres.iloc[j]} for j in signals[ix,:5]]
example['lists']={name:{str(k):[{'movieId':int(ids[j]),'title':movies.title.iloc[j],'target_match':bool(G[j,targets[ix]])} for j in tops_out[name][ix,k]] for k in [0,5]} for name in counts_out}
example['target_counts']={name:counts_out[name][ix].tolist() for name in counts_out}
summary['example']=example
rows.to_csv(ROOT/'cohort.csv',index=False)
(ROOT/'results.json').write_text(json.dumps(summary,indent=2))
log(json.dumps({k:v for k,v in summary.items() if k not in ['example','checksums']},indent=2))
log('Done seconds',time.time()-start)
