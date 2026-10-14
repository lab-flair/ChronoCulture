#!/usr/bin/env python3
"""Compute topic-level both-period accuracy for retained pair metrics.

If ``topic`` is not already present in the metric CSV, provide a mapping CSV with
columns ``dataset,variable,topic``.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',required=True); p.add_argument('--topic-map',default=None); p.add_argument('--tau',type=float,default=.10)
    p.add_argument('--min-n',type=int,default=20); p.add_argument('--combine-strong',action='store_true'); p.add_argument('--output',required=True); a=p.parse_args()
    df=pd.read_csv(a.input)
    if 'topic' not in df or df.topic.fillna('').eq('').all():
        topic_map=a.topic_map or str(Path(__file__).parents[1]/'configs'/'topic_map.csv')
        if not Path(topic_map).exists(): raise SystemExit('No topic column in metrics; supply --topic-map dataset,variable,topic')
        tm=pd.read_csv(topic_map); df=df.drop(columns=['topic'],errors='ignore').merge(tm[['dataset','variable','topic']].drop_duplicates(),on=['dataset','variable'],how='left',validate='many_to_one')
    df['shift_group']=df.shift_category.astype(str).str.replace('_',' ',regex=False)
    if a.combine_strong: df.loc[df.shift_group.isin(['strong','very strong']),'shift_group']='strong+'
    df['both_accurate']=(df.d_oo<=a.tau+1e-12)&(df.d_cc<=a.tau+1e-12)
    out=(df.groupby(['model_name','dataset','topic','shift_group']).both_accurate.agg(n_pairs='size',both_accurate_pct=lambda x:100*x.mean()).reset_index())
    out.loc[out.n_pairs<a.min_n,'both_accurate_pct']=float('nan')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print(out.to_string(index=False))

if __name__=='__main__': main()
