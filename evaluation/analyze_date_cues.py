#!/usr/bin/env python3
"""Summarize the benefit/harm of naming the older or newer year."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
EPS=1e-12


def rows_for(df):
    out=[]
    for cat,g in [('all',df)]+[(str(k),v) for k,v in df.groupby('shift_category')]:
        for period,prefix in [('older','old'),('newer','current')]:
            col=f'{prefix}_cue_benefit'; unchanged=f'{prefix}_cue_unchanged'
            if col not in g: continue
            x=pd.to_numeric(g[col],errors='coerce'); valid=x.notna(); x=x[valid]
            if not len(x): continue
            name=str(g.model_name.iloc[0])
            out.append({'model':name,'shift_category':cat.replace('_',' '),'period':period,'n_items':len(x),
                        'mean_benefit':x.mean(),'mean_benefit_x100':100*x.mean(),
                        'helped_pct':100*(x>EPS).mean(),'hurt_pct':100*(x<-EPS).mean(),
                        'unchanged_error_pct':100*(x.abs()<=EPS).mean(),
                        'identical_prediction_pct':100*pd.to_numeric(g.loc[valid,unchanged],errors='coerce').mean() if unchanged in g else np.nan})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',action='append',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    rows=[]
    for f in a.input:
        df=pd.read_csv(f); df=df[df.get('has_no_time',0)==1].copy(); rows+=rows_for(df)
    out=pd.DataFrame(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))

if __name__=='__main__': main()
