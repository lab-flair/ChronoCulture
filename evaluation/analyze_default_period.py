#!/usr/bin/env python3
"""Summarize which observed survey period undated predictions most closely match."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ORDER=['oldest','intermediate','newest']


def summarize(path,label=None):
    df=pd.read_csv(path); df=df[df.get('passes',True).astype(str).str.lower().isin(['true','1','1.0','yes','pass'])].copy() if 'passes' in df else df
    if 't_star_position' not in df: raise SystemExit(f'{path}: recompute gates with compute_knowledge_gate.py; t_star_position is missing')
    model=label or (str(df.model.iloc[0]) if 'model' in df and len(df) else Path(path).stem)
    rows=[]
    for coverage,g in [('all',df),('exactly_two',df[df.n_waves==2]),('three_or_more',df[df.n_waves>=3])]:
        if not len(g): continue
        n=len(g)
        for pos in ORDER:
            obs=100*(g.t_star_position==pos).mean()
            if coverage=='exactly_two': base=50.0 if pos in ('oldest','newest') else 0.0
            elif coverage=='three_or_more':
                # Equal-choice baseline: choose uniformly among each item's observed years.
                if pos in ('oldest','newest'): base=100*np.mean(1/g.n_waves)
                else: base=100*np.mean((g.n_waves-2)/g.n_waves)
            else:
                probs=np.where(g.n_waves==2, np.where(pos=='intermediate',0,1/g.n_waves),
                               np.where(pos=='intermediate',(g.n_waves-2)/g.n_waves,1/g.n_waves))
                base=100*np.mean(probs)
            rows.append({'model':model,'coverage_group':coverage,'position':pos,'n_items':n,'observed_pct':obs,'equal_choice_pct':base})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--gate',action='append',required=True,help='Combined gate CSV; repeat for models')
    p.add_argument('--label',action='append',help='Optional labels aligned with --gate'); p.add_argument('--output',required=True); a=p.parse_args()
    labels=a.label or [None]*len(a.gate)
    if len(labels)!=len(a.gate): raise SystemExit('--label count must match --gate count')
    rows=[]
    for path,label in zip(a.gate,labels): rows+=summarize(path,label)
    out=pd.DataFrame(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))

if __name__=='__main__': main()
