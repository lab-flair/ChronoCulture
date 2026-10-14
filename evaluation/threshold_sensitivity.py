#!/usr/bin/env python3
"""Recompute ChronoCulture summaries under alternative human-shift boundaries."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
EPS=1e-12
SETS={
 'tighter':(0.02,0.075,0.15),
 'default':(0.03,0.10,0.20),
 'looser':(0.04,0.125,0.25),
}


def cat(h,b):
    s,m,v=b
    return 'stable' if h<=s+EPS else 'moderate' if h<m-EPS else 'strong' if h<v-EPS else 'very strong'


def pct(x): return 100*np.asarray(x,dtype=float).mean() if len(x) else np.nan


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',action='append',required=True,help='Retained pair-metrics CSV; repeat for models')
    p.add_argument('--out-dir',required=True); p.add_argument('--tau',type=float,default=.10); a=p.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    acc=[]; align=[]; mag=[]; stable=[]
    for path in a.input:
        df=pd.read_csv(path); model=str(df.model_name.iloc[0])
        for setname,b in SETS.items():
            d=df.copy(); d['sens_category']=[cat(float(x),b) for x in d.human_shift_h]
            for c,g in d.groupby('sens_category'):
                old=g.d_oo<=a.tau+EPS; new=g.d_cc<=a.tau+EPS
                acc.append({'model':model,'set':setname,'shift_category':c,'n_pairs':len(g),'both_periods_accurate_pct':pct(old&new),'one_period_only_pct':pct(old^new)})
                if c!='stable':
                    same=g.anchoring.isin(['older_anchored','current_anchored'])
                    align.append({'model':model,'set':setname,'shift_category':c,'n_pairs':len(g),'separated_pct':pct(g.anchoring=='separated'),'same_period_pct':pct(same),'reversed_pct':pct(g.anchoring=='reversed')})
                    nonzero=g[g.model_shift_m>EPS]
                    mag.append({'model':model,'set':setname,'shift_category':c,'n_pairs':len(g),'zero_model_shift_pct':pct(g.model_shift_m<=EPS),
                                'n_nonzero':len(nonzero),'median_ratio_r':(nonzero.model_shift_m/nonzero.human_shift_h).median() if len(nonzero) else np.nan,
                                'median_cosine':nonzero.cosine.dropna().median() if len(nonzero) else np.nan})
            g=d[d.sens_category=='stable']; stable_bound=b[0]
            if len(g): stable.append({'model':model,'set':setname,'n_pairs':len(g),'preserves_pct':pct(g.model_shift_m<=stable_bound+EPS),
                                      'small_drift_pct':pct((g.model_shift_m>stable_bound+EPS)&(g.model_shift_m<.10-EPS)),
                                      'large_drift_pct':pct(g.model_shift_m>=.10-EPS)})
    for name,rows in [('period_accuracy',acc),('temporal_alignment',align),('direction_magnitude',mag),('stable_behavior',stable)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    print(f'Wrote threshold-sensitivity tables to {out}')

if __name__=='__main__': main()
