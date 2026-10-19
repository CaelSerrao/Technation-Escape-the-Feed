import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import json,zipfile,time
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
PARSER=argparse.ArgumentParser(description='Reproduce the Escape the Feed analysis')
PARSER.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data/ml-32m.zip')
PARSER.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results')
ARGS=PARSER.parse_args()
ROOT=ARGS.output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
ZIP=ARGS.data.resolve()
rng=np.random.default_rng(20261019)
start=time.time()
def log(*x): print(*x,flush=True)
model=np.load(ROOT/'model.npz');E=model['embeddings'];ids=model['movie_ids'];M=len(ids)
cohort=pd.read_csv(ROOT/'cohort.csv');tr=np.load(ROOT/'trajectories.npz')
with zipfile.ZipFile(ZIP) as z:
    movies=pd.read_csv(z.open('ml-32m/movies.csv')).set_index('movieId').loc[ids]
    ratings=pd.read_csv(z.open('ml-32m/ratings.csv'),dtype={'userId':'int32','movieId':'int32','rating':'float32','timestamp':'int64'})
log('Loaded ratings',time.time()-start)
genres=sorted(set('|'.join(movies.genres).split('|'))-{'(no genres listed)','IMAX'})
G=np.array([[g in s.split('|') for g in genres] for s in movies.genres],dtype=np.float64)
missing=G.sum(axis=1)==0
F=G/np.maximum(G.sum(axis=1,keepdims=True),1)
F=np.column_stack([F,missing.astype(float)])
genres=genres+['Unlabelled']
log('Movies without a usable genre',int(missing.sum()))
assert np.allclose(F.sum(axis=1),1)
ratings=ratings[ratings.movieId.isin(ids)].copy()
ratings['item']=pd.Index(ids).get_indexer(ratings.movieId)
pre=ratings[ratings.timestamp<1672531200]
post=ratings[(ratings.timestamp>=1672531200)&(ratings.rating>=4)]
positive=pre[pre.rating>=4]
counts=positive.groupby('userId').size();future=post.groupby('userId').size()
eligible=np.intersect1d(counts[counts>=20].index,future[future>=5].index)
testusers=np.sort(rng.choice(eligible,min(2000,len(eligible)),replace=False))
selected=np.union1d(testusers,cohort.userId)
hist={int(u):g.sort_values(['timestamp','movieId']).item.to_numpy() for u,g in positive[positive.userId.isin(selected)].groupby('userId')}
seen={int(u):g.item.to_numpy() for u,g in pre[pre.userId.isin(selected)].groupby('userId')}
held={int(u):g.item.to_numpy() for u,g in post[post.userId.isin(testusers)].groupby('userId')}
pop=positive.groupby('item').size().reindex(range(M),fill_value=0).to_numpy()
# Q1: original stress-test cohort, full-history baseline lists before any injected signals.
HP=np.stack([F[hist[int(u)]].mean(axis=0) for u in cohort.userId])
dominant=HP.argmax(axis=1)
exposure=F[tr['full_top'][:,0]].mean(axis=1)
assert np.allclose(exposure.sum(axis=1),1)
cohort['dominant_genre']=[genres[j] for j in dominant]
rows=[]
for g in sorted(set(dominant),key=lambda j:-(dominant==j).sum()):
    mask=dominant==g
    rows.append({'group':genres[g],'n':int(mask.sum()),'exposure':exposure[mask].mean(axis=0).tolist(),'history':HP[mask].mean(axis=0).tolist()})
heat={'cohort_n':len(cohort),'genres':genres,'groups':rows,'catalog_share':F.mean(axis=0).tolist(),'overall_exposure':exposure.mean(axis=0).tolist()}
# Q3: separate sample with actual later positive ratings; no synthetic signals.
Pfull=np.stack([E[hist[int(u)]].sum(axis=0) for u in testusers])
Preset=np.stack([E[hist[int(u)][-20:]].sum(axis=0) for u in testusers])
res={};top_arrays={};metric_frames=[];boots=rng.integers(0,len(testusers),size=(2000,len(testusers)))
for name,P in [('Popularity',None),('Full history',Pfull),('20-item reset',Preset)]:
    scores=np.tile(pop,(len(testusers),1)).astype(np.float32) if P is None else P@E.T
    for i,u in enumerate(testusers): scores[i,seen[int(u)]]=-np.inf
    # stable ordering is deterministic for tied popularity counts.
    tops=np.argsort(-scores,axis=1,kind='stable')[:,:10];top_arrays[name]=tops
    vals=[]
    for i,u in enumerate(testusers):
        truth=held[int(u)];assert not np.intersect1d(truth,seen[int(u)]).size
        hits=np.isin(tops[i],truth);dcg=(hits/np.log2(np.arange(2,12))).sum()
        idcg=(1/np.log2(np.arange(2,min(10,len(truth))+2))).sum()
        q=F[tops[i],:-1].mean(axis=0)
        q=q/q.sum() if q.sum()>0 else q
        effective=float(np.exp(-np.sum(q[q>0]*np.log(q[q>0])))) if q.sum()>0 else 0.
        vals.append([dcg/idcg,hits.sum()/len(truth),float(hits.any()),effective,float((G[tops[i]].sum(axis=0)>0).sum())])
    vals=np.array(vals)
    res[name]={}
    for j,metric in enumerate(['ndcg10','recall10','hit_rate10','effective_genres','distinct_genres']):
        ci=np.quantile(vals[:,j][boots].mean(axis=1),[.025,.975])
        res[name][metric]={'mean':float(vals[:,j].mean()),'ci':[float(x) for x in ci]}
    frame=pd.DataFrame(vals,columns=['ndcg10','recall10','hit_rate10','effective_genres','distinct_genres']);frame['userId']=testusers;frame['method']=name;metric_frames.append(frame)
    log(name,res[name])
metrics=pd.concat(metric_frames,ignore_index=True);metrics.to_csv(ROOT/'benchmark_metrics.csv',index=False)
paired={}
for base,compare in [('Popularity','Full history'),('Full history','20-item reset')]:
    a=metrics[metrics.method==base].set_index('userId');b=metrics[metrics.method==compare].set_index('userId')
    paired[compare+' minus '+base]={}
    for metric in ['ndcg10','effective_genres']:
        d=(b[metric]-a[metric]).to_numpy();ci=np.quantile(d[boots].mean(axis=1),[.025,.975])
        paired[compare+' minus '+base][metric]={'difference':float(d.mean()),'ci':ci.tolist()}
bench={'n':len(testusers),'eligible_n':len(eligible),'seed':20261019,'overlap_with_steering_cohort':int(np.isin(testusers,cohort.userId).sum()),'test_positive_ratings':int(sum(len(held[int(u)]) for u in testusers)),'methods':res,'paired':paired}
pd.DataFrame({'userId':testusers,'history_n':[len(hist[int(u)]) for u in testusers],'test_positive_n':[len(held[int(u)]) for u in testusers]}).to_csv(ROOT/'benchmark_cohort.csv',index=False)
np.savez_compressed(ROOT/'benchmark_top10.npz',users=testusers,popularity=top_arrays['Popularity'],full=top_arrays['Full history'],reset=top_arrays['20-item reset'])
# Q5: genre- and baseline-exposure-standardized success gap by starting history length.
# Compare <=50 vs >100 retained positives. Require >=10 in BOTH groups in each stratum.
cohort['group']=np.where(cohort.history_n<=50,'Short (20-50)',np.where(cohort.history_n>100,'Long (101+)','Excluded middle'))
# Original full-history initial exposure defines pre-intervention strata for both methods.
cohort['initial_count']=tr['full'][:,0]
cohort['stratum']=cohort.target+' / '+cohort.initial_count.astype(str)
for key,name in [('full','Full history'),('recent','20-item reset')]:cohort[name+' success5']=(tr[key][:,:6].max(axis=1)>=3).astype(int)
pool=cohort[cohort.group!='Excluded middle'];ct=pd.crosstab(pool.stratum,pool.group)
keep=ct.index[(ct['Short (20-50)']>=10)&(ct['Long (101+)']>=10)]
matched=pool[pool.stratum.isin(keep)].copy();weights=matched.stratum.value_counts(normalize=True).sort_index()
fair={'comparison':'Short (20-50) minus long (101+)','strata':['target genre','initial full-history target count (0, 1 or 2)'],'minimum_per_group_per_stratum':10,'stratum_n':len(keep),'n':len(matched),'group_n':matched.group.value_counts().to_dict(),'excluded_n':len(cohort)-len(matched),'weights':weights.to_dict(),'methods':{}}
for name in ['Full history','20-item reset']:
    estimates={};draws={}
    for group in ['Short (20-50)','Long (101+)']:
        value=0.;d=np.zeros(2000)
        for stratum,w in weights.items():
            a=matched[(matched.group==group)&(matched.stratum==stratum)][name+' success5'].to_numpy()
            value+=w*a.mean();d+=w*a[rng.integers(len(a),size=(2000,len(a)))].mean(axis=1)
        estimates[group]={'rate':float(value),'ci':np.quantile(d,[.025,.975]).tolist()};draws[group]=d
    delta=estimates['Short (20-50)']['rate']-estimates['Long (101+)']['rate']
    dd=draws['Short (20-50)']-draws['Long (101+)']
    fair['methods'][name]={'groups':estimates,'signed_gap':float(delta),'absolute_gap':float(abs(delta)),'gap_ci':np.quantile(dd,[.025,.975]).tolist()}
matched.to_csv(ROOT/'fairness_cohort.csv',index=False)
cohort.to_csv(ROOT/'exposure_cohort.csv',index=False)
fair['uncertainty_note']='Descriptive rates only. Do not interpret degenerate bootstrap intervals from zero outcomes as zero population uncertainty; these intervals are not shown in the report.'
output={'heatmap':heat,'benchmark':bench,'fairness':fair}
(ROOT/'extended_results.json').write_text(json.dumps(output,indent=2))
log('Heatmap groups',[(x['group'],x['n']) for x in rows]);log('Benchmark cohort',len(testusers),'eligible',len(eligible));log('Fairness',fair);log('Finished',time.time()-start)
