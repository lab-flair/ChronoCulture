#!/usr/bin/env python3
"""Summarize older/newer pair accuracy at one or more Wasserstein tolerances."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def norm_cat(s): return str(s).strip().lower().replace('_',' ')


def summarize(df,tau):
    rows=[]
    for (model,cat),g in df.groupby(['model_name','shift_category'],dropna=False):
        old=g.d_oo<=tau+1e-12; new=g.d_cc<=tau+1e-12
        rows.append({'model':model,'tau':tau,'shift_category':norm_cat(cat),'n_pairs':len(g),
            'older_accurate_pct':100*old.mean(),'newer_accurate_pct':100*new.mean(),
            'both_pct':100*(old&new).mean(),'older_only_pct':100*(old&~new).mean(),
            'newer_only_pct':100*(~old&new).mean(),'neither_pct':100*(~old&~new).mean()})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',action='append',required=True,help='Retained pair-metrics CSV; repeat for models')
    p.add_argument('--tau',type=float,action='append',default=None); p.add_argument('--output',required=True); a=p.parse_args()
    taus=a.tau or [.10]; allrows=[]
    for path in a.input:
        df=pd.read_csv(path); req={'model_name','shift_category','d_oo','d_cc'}
        if not req<=set(df): raise SystemExit(f'{path}: missing {sorted(req-set(df))}')
        for t in taus: allrows.extend(summarize(df,float(t)))
    out=pd.DataFrame(allrows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))

if __name__=='__main__': main()
