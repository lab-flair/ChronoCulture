#!/usr/bin/env python3
"""Build the supplementary ChronoCulture modal-answer benchmark.

Input is the released dated ChronoCulture JSONL itself; no raw survey files are
needed. The paper construction keeps:
  * stable pairs (human shift <= 0.03), and
  * strong/very-strong pairs (human shift >= 0.10) whose modal response changes,
while excluding moderate shifts. Both dated gold distributions must have a
>=10 percentage-point margin between their most and second-most common options.
"""
from __future__ import annotations
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    with open(path,encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def distribution(row):
    d=row.get('gold_distribution')
    if not isinstance(d,dict): raise ValueError('gold_distribution must be an object')
    keys=[str(x) for x in row.get('valid_options',sorted(d,key=lambda x:int(x)))]
    vals=[]
    for k in keys:
        v=d.get(k,d.get(int(k)) if k.isdigit() else None)
        if v is None: raise ValueError(f'missing gold option {k}')
        vals.append(float(v))
    if any(v<0 for v in vals) or sum(vals)<=0: raise ValueError('invalid gold distribution')
    total=sum(vals)
    return {k:v/total for k,v in zip(keys,vals)}


def top_and_margin(d):
    ranked=sorted(d.items(),key=lambda kv:(-kv[1],int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])))
    top,prob=ranked[0]; second=ranked[1][1] if len(ranked)>1 else 0.0
    return str(top),float(prob),float(prob-second)


def strip_distribution_shell(text):
    s=str(text).strip()
    s=re.sub(r"\n+\s*Instead of choosing one answer,.*$","",s,flags=re.I|re.S)
    s=re.sub(r"\n+\s*Return percentages.*$","",s,flags=re.I|re.S)
    s=re.sub(r"\n+\s*Return only a JSON object.*$","",s,flags=re.I|re.S)
    return s.strip()


def mcq_prompt(row):
    return (strip_distribution_shell(row['prompt_text'])+
            "\n\nChoose the response option expected to be most common. "
            "Return only one valid option code.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True,help='Released dated ChronoCulture JSONL')
    p.add_argument('--output-jsonl',required=True)
    p.add_argument('--review-csv',required=True)
    p.add_argument('--stable-max',type=float,default=.03)
    p.add_argument('--strong-min',type=float,default=.10)
    p.add_argument('--min-top-margin',type=float,default=.10)
    a=p.parse_args()

    grouped=defaultdict(list)
    for r in load_jsonl(a.benchmark): grouped[str(r['pair_id'])].append(r)
    output=[]; review=[]
    for pid,rows in grouped.items():
        eras={str(r.get('wave_type','')):r for r in rows}
        if set(eras)!={'older','current'}:
            review.append({'pair_id':pid,'keep':False,'drop_reason':'missing_or_duplicate_era'}); continue
        old,cur=eras['older'],eras['current']
        try:
            od,cd=distribution(old),distribution(cur)
            ot,op,om=top_and_margin(od); ct,cp,cm=top_and_margin(cd)
            h=float(old.get('gold_shift_w1',cur.get('gold_shift_w1')))
        except Exception as e:
            review.append({'pair_id':pid,'keep':False,'drop_reason':f'invalid_gold:{e}'}); continue
        margins_ok=om>=a.min_top_margin and cm>=a.min_top_margin
        mode_changed=ot!=ct
        if h<=a.stable_max+1e-12 and margins_ok:
            keep=True; mcq_set='stable'
        elif h>=a.strong_min-1e-12 and mode_changed and margins_ok:
            keep=True; mcq_set='shifted_changed_mode'
        else:
            keep=False; mcq_set=''
        reasons=[]
        if not margins_ok: reasons.append('top_margin_below_threshold')
        if a.stable_max<h<a.strong_min: reasons.append('moderate_shift_excluded')
        if h>=a.strong_min and not mode_changed: reasons.append('shifted_but_mode_unchanged')
        review.append({'dataset':old.get('dataset',''),'pair_id':pid,'human_w1':h,'old_top':ot,'new_top':ct,
                       'old_top_margin':om,'new_top_margin':cm,'mode_changed':mode_changed,
                       'keep':keep,'mcq_set':mcq_set,'drop_reason':';'.join(reasons)})
        if not keep: continue
        for row,ans,prob,margin in [(old,ot,op,om),(cur,ct,cp,cm)]:
            o=dict(row); o['prompt_text']=mcq_prompt(row); o['mcq_set']=mcq_set
            o['mcq_gold_answer']=ans; o['mcq_gold_top_prob']=prob; o['mcq_gold_top_margin']=margin
            o['older_gold_top']=ot; o['current_gold_top']=ct; output.append(o)

    op=Path(a.output_jsonl); op.parent.mkdir(parents=True,exist_ok=True)
    with op.open('w',encoding='utf-8') as f:
        for r in output: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    rp=Path(a.review_csv); rp.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in review for k in r})
    with rp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(review)
    kept=sum(bool(r.get('keep')) for r in review)
    print(f'Reviewed {len(review):,} pairs; kept {kept:,}; wrote {len(output):,} prompt rows.')

if __name__=='__main__': main()
