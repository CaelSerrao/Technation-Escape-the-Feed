"""Independent audit of the supplied model, rankings, metrics and source data.

Usage: python verify_saved_results.py --results DIR --data ml-32m.zip --output audit.json
This verifies the saved fit; it does not refit SVD or assert scientific validity.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();R=args.results
s=json.loads((R/'results.json').read_text());e=json.loads((R/'extended_results.json').read_text())
m=np.load(R/'model.npz');E=m['embeddings'];ids=m['movie_ids'];N=s['cohort_n'];M=len(ids)
c=pd.read_csv(R/'cohort.csv');t=np.load(R/'trajectories.npz');bt=np.load(R/'benchmark_top10.npz');bc=pd.read_csv(R/'benchmark_cohort.csv');bm=pd.read_csv(R/'benchmark_metrics.csv')
checks=[]
def check(name,actual,expected,tol=1e-10):
    a=np.asarray(actual);b=np.asarray(expected)
    ok=a.shape==b.shape and (np.allclose(a,b,atol=tol,rtol=tol) if a.dtype.kind in 'biufc' else np.array_equal(a,b))
    if not ok: raise AssertionError(f'{name}: actual={actual!r}; expected={expected!r}')
    checks.append(name)
def ci(v,boot):return np.quantile(np.asarray(v)[boot].mean(axis=1),[.025,.975])
selected=np.union1d(c.userId,bt['users']);parts=[];raw_users=set();raw_n=0;allpop={};retained_counts={};pre_catalog_n=0
sha=hashlib.sha256(args.data.read_bytes()).hexdigest();manifest=json.loads((R/'manifest.json').read_text())
check('source_archive_sha256',sha,manifest['input_sha256'])
with zipfile.ZipFile(args.data) as z:
    movies=pd.read_csv(z.open('ml-32m/movies.csv')).set_index('movieId').loc[ids]
    for file in ['ratings.csv','movies.csv']:
        h=hashlib.md5()
        with z.open('ml-32m/'+file) as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        check(file+'_md5',h.hexdigest(),s['checksums'][file]['md5'])
    for chunk in pd.read_csv(z.open('ml-32m/ratings.csv'),dtype={'userId':'int32','movieId':'int32','rating':'float32','timestamp':'int64'},chunksize=1_000_000):
        raw_n+=len(chunk);raw_users.update(chunk.userId.unique().tolist())
        pre=chunk[chunk.timestamp<1672531200];pos=pre[pre.rating>=4]
        for k,v in pos.groupby('movieId').size().items():allpop[int(k)]=allpop.get(int(k),0)+int(v)
        pre=pre[pre.movieId.isin(ids)];pre_catalog_n+=len(pre)
        for k,v in pre[pre.rating>=4].groupby('userId').size().items():retained_counts[int(k)]=retained_counts.get(int(k),0)+int(v)
        parts.append(chunk[chunk.userId.isin(selected)&chunk.movieId.isin(ids)].copy())
        print('Scanned',raw_n,flush=True)
check('raw_ratings',raw_n,s['raw_ratings']);check('raw_users',len(raw_users),s['raw_users'])
check('catalog_selection',np.sort([k for k,v in allpop.items() if v>=100]),ids)
check('catalog_movies',M,s['catalog_movies']);check('training_users',len(retained_counts),s['training_users'])
check('training_positive_ratings',sum(retained_counts.values()),s['training_positive_ratings'])
check('training_ratings_catalog',pre_catalog_n,s['training_ratings_catalog'])
check('eligible_history_users',sum(v>=20 for v in retained_counts.values()),s['eligible_history_users'])
r=pd.concat(parts,ignore_index=True);del parts
r['item']=pd.Index(ids).get_indexer(r.movieId)
pre=r[r.timestamp<1672531200];pos=pre[pre.rating>=4];post=r[(r.timestamp>=1672531200)&(r.rating>=4)]
hist={int(u):g.sort_values(['timestamp','movieId']).item.to_numpy() for u,g in pos.groupby('userId')}
seen={int(u):g.item.to_numpy() for u,g in pre.groupby('userId')};held={int(u):g.item.to_numpy() for u,g in post.groupby('userId')}
genres=sorted(set('|'.join(movies.genres).split('|'))-{'(no genres listed)','IMAX'})
G=np.array([[g in x.split('|') for g in genres] for x in movies.genres],bool)
F=G/np.maximum(G.sum(axis=1,keepdims=True),1);F=np.column_stack([F,G.sum(axis=1)==0])
pop=np.array([allpop[int(i)] for i in ids]);target=t['targets'];sig=t['signals']
check('unit_item_vectors',np.linalg.norm(E,axis=1),np.ones(M),tol=2e-6)
check('cohort_n',len(c),N);check('unique_cohort_users',c.userId.nunique(),N)
check('history_counts',[len(hist[int(u)]) for u in c.userId],c.history_n)
check('history_median',c.history_n.median(),s['history_median'])
check('mainstream_scores',[np.log1p(pop[hist[int(u)]]).mean() for u in c.userId],c.mainstream_score)
check('target_labels',[genres[j] for j in target],c.target)
check('target_historical_shares',[G[hist[int(u)],target[i]].mean() for i,u in enumerate(c.userId)],c.target_history_share)
check('rare_target_constraint',(c.target_history_share<=.10).all(),True)
check('target_catalog_constraint',(G.sum(axis=0)[target]>=100).all(),True)
for i,u in enumerate(c.userId):
    check('signal_valid_'+str(i),len(np.unique(sig[i]))==10 and G[sig[i],target[i]].all() and not np.intersect1d(sig[i],seen[int(u)]).size,True)
for key,recent in [('full',False),('recent',True)]:
    P=np.array([E[hist[int(u)][-20:] if recent else hist[int(u)]].sum(axis=0) for u in c.userId],dtype=np.float32)
    for k in range(11):
        if k:P+=E[sig[:,k-1]]
        scores=P@E.T
        for i,u in enumerate(c.userId):scores[i,seen[int(u)]]=-np.inf;scores[i,sig[i,:k]]=-np.inf
        top=np.argpartition(-scores,9,axis=1)[:,:10]
        top=np.array([v[np.argsort(-scores[i,v],kind='stable')] for i,v in enumerate(top)])
        check(f'{key}_rankings_step_{k}',top,t[key+'_top'][:,k])
        check(f'{key}_genre_counts_step_{k}',G[top,target[:,None]].sum(axis=1),t[key][:,k])
    print('Verified all rankings:',key,flush=True)
boot=np.random.default_rng(s['seed']+1).integers(0,N,(2000,N));success={}
for key,name in [('full','Full history'),('recent','20-item reset')]:
    out=t[key];reach=np.maximum.accumulate(out,axis=1)>=3;success[name]=reach[:,5].astype(int);a=s['models'][name]
    check(name+'_cumulative_curve',reach.mean(axis=0),a['cumulative_success']);check(name+'_current_curve',(out>=3).mean(axis=0),a['current_success'])
    check(name+'_mean_slots',out.mean(axis=0),a['mean_target_slots']);check(name+'_success_count',reach[:,5].sum(),a['success5_count']);check(name+'_success_ci',ci(reach[:,5],boot),a['success5_ci'])
    for threshold,v in a['thresholds5'].items():check(name+'_threshold_'+threshold,(out[:,:6].max(axis=1)>=int(threshold)).mean(),v)
    for field,groups in a['groups'].items():
        for group,vals in groups.items():
            mask=c[field]==group;check(name+field+group+'n',mask.sum(),vals['n']);check(name+field+group+'rate',reach[mask,5].mean(),vals['success5']);check(name+field+group+'median',c.loc[mask,'history_n'].median(),vals['history_median'])
gain=success['20-item reset']-success['Full history'];check('paired_gain5',gain.mean(),s['paired_gain5']);check('paired_gain5_ci',ci(gain,boot),s['paired_gain5_ci'])
check('target_counts',c.target.value_counts().to_dict(),s['target_counts'])
HP=np.stack([F[hist[int(u)]].mean(axis=0) for u in c.userId]);dominant=HP.argmax(axis=1);exposure=F[t['full_top'][:,0]].mean(axis=1)
hm=e['heatmap'];check('genre_order',genres+['Unlabelled'],hm['genres']);check('catalog_fractional_mix',F.mean(axis=0),hm['catalog_share']);check('overall_exposure',exposure.mean(axis=0),hm['overall_exposure'])
for group in hm['groups']:
    mask=dominant==hm['genres'].index(group['group']);check(group['group']+'_n',mask.sum(),group['n']);check(group['group']+'_history',HP[mask].mean(axis=0),group['history']);check(group['group']+'_exposure',exposure[mask].mean(axis=0),group['exposure'])
users=bt['users'];bench=e['benchmark'];check('benchmark_n',len(users),bench['n']);check('benchmark_overlap',np.isin(users,c.userId).sum(),bench['overlap_with_steering_cohort'])
check('benchmark_history_counts',[len(hist[int(u)]) for u in users],bc.history_n);check('benchmark_future_counts',[len(held[int(u)]) for u in users],bc.test_positive_n);check('benchmark_positive_total',sum(len(held[int(u)]) for u in users),bench['test_positive_ratings'])
# RNG state for the benchmark's bootstrap follows its random cohort draw.
rng=np.random.default_rng(20261019);rng.choice(np.arange(bench['eligible_n']),bench['n'],replace=False);boots=rng.integers(0,len(users),(2000,len(users)))
metrics={};names=['ndcg10','recall10','hit_rate10','effective_genres','distinct_genres']
for key,name,rec in [('popularity','Popularity',None),('full','Full history',False),('reset','20-item reset',True)]:
    scores=np.tile(pop,(len(users),1)).astype(np.float32) if rec is None else np.stack([E[hist[int(u)][-20:] if rec else hist[int(u)]].sum(axis=0) for u in users])@E.T
    for i,u in enumerate(users):scores[i,seen[int(u)]]=-np.inf
    tops=np.argsort(-scores,axis=1,kind='stable')[:,:10];check(name+'_benchmark_rankings',tops,bt[key]);values=[]
    for i,u in enumerate(users):
        truth=held[int(u)];check('unseen_future_'+str(u),len(np.intersect1d(truth,seen[int(u)])),0)
        hits=np.isin(tops[i],truth);discount=1/np.log2(np.arange(2,12));ndcg=(hits*discount).sum()/discount[:min(10,len(truth))].sum()
        q=F[tops[i],:-1].mean(axis=0);q=q/q.sum() if q.sum() else q;d=np.exp(-np.sum(q[q>0]*np.log(q[q>0]))) if q.sum() else 0
        values.append([ndcg,hits.sum()/len(truth),float(hits.any()),d,float((G[tops[i]].sum(axis=0)>0).sum())])
    values=np.array(values);metrics[name]=values;check(name+'_individual_metrics',values,bm[bm.method==name][names].to_numpy())
    for j,metric in enumerate(names):check(name+metric+'_mean',values[:,j].mean(),bench['methods'][name][metric]['mean']);check(name+metric+'_ci',ci(values[:,j],boots),bench['methods'][name][metric]['ci'])
for base,comp in [('Popularity','Full history'),('Full history','20-item reset')]:
    for metric,j in [('ndcg10',0),('effective_genres',3)]:
        diff=metrics[comp][:,j]-metrics[base][:,j];v=bench['paired'][comp+' minus '+base][metric];check(comp+base+metric+'_difference',diff.mean(),v['difference']);check(comp+base+metric+'_ci',ci(diff,boots),v['ci'])
# Independently reconstruct standardization, including both groups' denominators.
fc=c.copy();fc['group']=np.where(fc.history_n<=50,'Short (20-50)',np.where(fc.history_n>100,'Long (101+)','Excluded middle'));fc['stratum']=fc.target+' / '+t['full'][:,0].astype(str)
pool=fc[fc.group!='Excluded middle'];ct=pd.crosstab(pool.stratum,pool.group);keep=ct.index[(ct['Short (20-50)']>=10)&(ct['Long (101+)']>=10)];matched=pool[pool.stratum.isin(keep)].copy();weights=matched.stratum.value_counts(normalize=True).sort_index();fair=e['fairness']
check('standardized_sample',len(matched),fair['n']);check('standardized_strata',len(keep),fair['stratum_n']);check('standardized_excluded',N-len(matched),fair['excluded_n']);check('standardized_groups',matched.group.value_counts().to_dict(),fair['group_n'])
for k,w in weights.items():check('weight_'+k,w,fair['weights'][k])
for name in ['Full history','20-item reset']:
    draws={};rates={}
    for group in ['Short (20-50)','Long (101+)']:
        rate=0.;d=np.zeros(2000)
        for stratum,w in weights.items():
            ix=matched.index[(matched.group==group)&(matched.stratum==stratum)];a=success[name][ix];rate+=w*a.mean();d+=w*a[rng.integers(len(a),size=(2000,len(a)))].mean(axis=1)
        rates[group]=rate;draws[group]=d;v=fair['methods'][name]['groups'][group];check(name+group+'_standardized_rate',rate,v['rate']);check(name+group+'_standardized_ci',np.quantile(d,[.025,.975]),v['ci'])
    gap=rates['Short (20-50)']-rates['Long (101+)'];v=fair['methods'][name];check(name+'_gap',gap,v['signed_gap']);check(name+'_absolute_gap',abs(gap),v['absolute_gap']);check(name+'_gap_ci',np.quantile(draws['Short (20-50)']-draws['Long (101+)'],[.025,.975]),v['gap_ci'])
ex=s['example'];ix=ex['cohort_index'];check('example_user',int(c.userId.iloc[ix]),ex['userId'])
for j,item in enumerate(ex['signals']):
    idx=sig[ix,j];check('example_signal_id_'+str(j),int(ids[idx]),item['movieId']);check('example_signal_title_'+str(j),movies.title.iloc[idx],item['title']);check('example_signal_genres_'+str(j),movies.genres.iloc[idx],item['genres'])
for key,name in [('full','Full history'),('recent','20-item reset')]:
    check(name+'_example_counts',t[key][ix],ex['target_counts'][name])
    for k,items in ex['lists'][name].items():
        for j,item in enumerate(items):
            idx=t[key+'_top'][ix,int(k),j];check(name+k+str(j)+'_example_id',int(ids[idx]),item['movieId']);check(name+k+str(j)+'_example_title',movies.title.iloc[idx],item['title']);check(name+k+str(j)+'_example_match',bool(G[idx,target[ix]]),item['target_match'])
audit={'status':'PASS','checks_passed':len(checks),'source_archive_sha256':sha,'scope':'Raw counts and checksums, saved model rankings, steering curves, individual future-rating metrics, bootstrap intervals, exposure distributions, standardized subgroup rates and example. Does not establish causal effects or refit SVD.','not_recomputed_here':['SVD explained variance and factor fit','full eligible future-rating cohort (separate full pipeline reproduction checks this)'],'checks':checks}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='checks'},indent=2),flush=True)
