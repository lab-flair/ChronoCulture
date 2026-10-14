#!/usr/bin/env python3
"""Run ChronoCulture verbalized-distribution prompts via an OpenAI-compatible API.

The same runner works with OpenAI or a local vLLM server. It sends each row's
prebuilt ``prompt_text`` unchanged and uses the paper's common system prompt.
Outputs are compatible with ``evaluation/compute_pair_metrics.py`` and
``evaluation/gate/compute_knowledge_gate.py``.
"""
from __future__ import annotations
import argparse, csv, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SYSTEM_MESSAGE=(
    "You estimate survey response distributions. "
    "Return only valid JSON. Do not include explanations, markdown, or extra text."
)
OUTPUT_FIELDS=[
    'prompt_id','pair_id','dataset','variable','country_alpha','country_name','wave_type','target_wave',
    'scale_min','scale_max','scale_points','backend','model','status','error_message',
    'predicted_distribution_json','raw_output','temperature','top_p','max_completion_tokens','attempts'
]


def read_jsonl(path,start=0,limit=None):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    rows=rows[start:]
    return rows if limit is None else rows[:limit]


def extract_json_object(text):
    if text is None: return None
    s=str(text).strip()
    try:
        o=json.loads(s)
        if isinstance(o,dict): return o
    except Exception: pass
    m=re.search(r'\{.*\}',s,re.S)
    if not m: return None
    try:
        o=json.loads(m.group(0)); return o if isinstance(o,dict) else None
    except Exception: return None


def normalized_distribution(obj, row):
    if not isinstance(obj,dict): return None
    lo=int(float(row.get('scale_min',1))); hi=int(float(row.get('scale_max',row.get('scale_points',0))))
    keys=[str(i) for i in range(lo,hi+1)]
    vals=[]
    for k in keys:
        v=obj.get(k,obj.get(int(k)))
        if v is None: return None
        try: vals.append(float(v))
        except Exception: return None
    if any(v<0 for v in vals) or sum(vals)<=0: return None
    total=sum(vals)
    return {k:v/total for k,v in zip(keys,vals)}


def completed_ids(path):
    if not path or not Path(path).exists(): return set()
    with open(path,encoding='utf-8-sig',newline='') as f:
        return {r.get('prompt_id','') for r in csv.DictReader(f) if r.get('prompt_id') and str(r.get('status','')).startswith('ok')}


def run_one(client,row,args):
    prompt=row['prompt_text']; last=''
    for attempt in range(1,args.retries+2):
        try:
            kw=dict(model=args.model,messages=[{'role':'system','content':SYSTEM_MESSAGE},{'role':'user','content':prompt}],
                    temperature=args.temperature,top_p=args.top_p,max_tokens=args.max_completion_tokens)
            if args.seed is not None: kw['seed']=args.seed
            if args.extra_body: kw['extra_body']=json.loads(args.extra_body)
            # Newer OpenAI chat models use max_completion_tokens; older
            # OpenAI-compatible servers may still expose max_tokens.
            token_limit=kw.pop('max_tokens')
            try:
                resp=client.chat.completions.create(**kw,max_completion_tokens=token_limit)
            except TypeError:
                resp=client.chat.completions.create(**kw,max_tokens=token_limit)
            raw=resp.choices[0].message.content or ''
            pred=normalized_distribution(extract_json_object(raw),row)
            if pred is None: raise ValueError('response could not be parsed as a complete nonnegative distribution')
            return pred,raw,attempt,''
        except Exception as e:
            last=f'{type(e).__name__}: {e}'
            if attempt<=args.retries:
                time.sleep(args.retry_base_sleep*(2**(attempt-1))+random.random())
    return None,'',args.retries+1,last


def output_row(row,args,pred,raw,attempts,error):
    country_alpha=row.get('country_alpha',row.get('country_code_iso3',''))
    target=row.get('target_wave',row.get('time_period',''))
    status='ok' if pred is not None else 'error'
    return {k:v for k,v in {
        'prompt_id':row.get('prompt_id',''),'pair_id':row.get('pair_id',''),'dataset':row.get('dataset',''),
        'variable':row.get('variable',''),'country_alpha':country_alpha,'country_name':row.get('country_name',''),
        'wave_type':row.get('wave_type',''),'target_wave':target,'scale_min':row.get('scale_min',''),
        'scale_max':row.get('scale_max',''),'scale_points':row.get('scale_points',''),'backend':args.backend,
        'model':args.model,'status':status,'error_message':error,
        'predicted_distribution_json':json.dumps(pred,sort_keys=True) if pred is not None else '',
        'raw_output':raw,'temperature':args.temperature,'top_p':args.top_p,
        'max_completion_tokens':args.max_completion_tokens,'attempts':attempts}.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl',required=True); p.add_argument('--output-csv',required=True)
    p.add_argument('--backend',choices=['openai','vllm'],default='vllm'); p.add_argument('--model',required=True)
    p.add_argument('--base-url'); p.add_argument('--api-key-env',default='OPENAI_API_KEY')
    p.add_argument('--start',type=int,default=0); p.add_argument('--limit',type=int)
    p.add_argument('--concurrency',type=int,default=4); p.add_argument('--retries',type=int,default=5)
    p.add_argument('--retry-base-sleep',type=float,default=1.5); p.add_argument('--request-timeout',type=float,default=120)
    p.add_argument('--temperature',type=float,default=0.0); p.add_argument('--top-p',type=float,default=1.0)
    p.add_argument('--max-completion-tokens',type=int,default=120); p.add_argument('--seed',type=int)
    p.add_argument('--extra-body',help='Optional JSON object passed as extra_body (e.g., a model-specific vLLM chat-template flag).')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    from openai import OpenAI
    key=os.getenv(args.api_key_env)
    if args.backend=='openai' and not key: raise SystemExit(f'Missing {args.api_key_env}')
    client=OpenAI(api_key=key or 'EMPTY',base_url=args.base_url or (None if args.backend=='openai' else 'http://localhost:8000/v1'),timeout=args.request_timeout)
    rows=read_jsonl(args.input_jsonl,args.start,args.limit)
    done=completed_ids(args.output_csv) if args.resume else set()
    rows=[r for r in rows if r.get('prompt_id','') not in done]
    out=Path(args.output_csv); out.parent.mkdir(parents=True,exist_ok=True)
    append=args.resume and out.exists() and out.stat().st_size>0
    with out.open('a' if append else 'w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=OUTPUT_FIELDS,extrasaction='ignore')
        if not append: writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures={pool.submit(run_one,client,r,args):r for r in rows}
            for fut in as_completed(futures):
                r=futures[fut]; pred,raw,attempts,error=fut.result()
                writer.writerow(output_row(r,args,pred,raw,attempts,error)); f.flush()
    print(f'Wrote {len(rows)} rows to {out}')


if __name__=='__main__': main()
