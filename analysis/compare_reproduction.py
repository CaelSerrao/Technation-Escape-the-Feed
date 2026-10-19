"""Compare fresh pipeline results with the delivered reference outputs."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
def compare(x,y,name):
    if isinstance(x,dict):
        assert x.keys()==y.keys(),name
        for k in x:compare(x[k],y[k],name+'/'+k)
    elif isinstance(x,list):
        assert len(x)==len(y),name
        for i,(u,v) in enumerate(zip(x,y)):compare(u,v,name+'/'+str(i))
    elif isinstance(x,(float,int)) and not isinstance(x,bool):
        assert np.isclose(x,y,rtol=1e-12,atol=1e-12),name
        rows.append((name,float(abs(x-y))))
    else:assert x==y,(name,x,y)
for filename in ['results.json','extended_results.json']:
    compare(json.loads((a.reference/filename).read_text()),json.loads((a.candidate/filename).read_text()),filename)
arrays={}
for filename in ['model.npz','trajectories.npz','benchmark_top10.npz']:
    x=np.load(a.reference/filename);y=np.load(a.candidate/filename);assert x.files==y.files
    arrays[filename]={}
    for k in x.files:
        assert x[k].shape==y[k].shape and np.allclose(x[k],y[k],rtol=1e-12,atol=1e-12),(filename,k)
        arrays[filename][k]={'exact_match':bool(np.array_equal(x[k],y[k])),'max_absolute_difference':float(np.max(np.abs(x[k]-y[k])))}
csv={}
for filename in ['cohort.csv','exposure_cohort.csv','benchmark_cohort.csv','benchmark_metrics.csv','fairness_cohort.csv']:
    x=pd.read_csv(a.reference/filename);y=pd.read_csv(a.candidate/filename);pd.testing.assert_frame_equal(x,y,check_exact=False,rtol=1e-12,atol=1e-12);csv[filename]={'rows':len(x),'columns':len(x.columns),'pass':True}
diffs=[{'path':n,'absolute_difference':v} for n,v in rows if v]
report={'status':'PASS','numeric_json_values_compared':len(rows),'nonzero_json_differences':diffs,'max_json_absolute_difference':max(v for _,v in rows),'npz_arrays':arrays,'csv_tables':csv,'note':'A fresh model fit and both analyses were run with pinned package versions. Floating-point differences below 1e-12 are accepted; report rounding is unaffected.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
