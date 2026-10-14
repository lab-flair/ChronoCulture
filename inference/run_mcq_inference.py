#!/usr/bin/env python3
"""Run the supplementary ChronoCulture modal-answer (MCQ) prompts."""
from __future__ import annotations
import argparse, asyncio, json, os, re
from pathlib import Path
from openai import AsyncOpenAI

SYSTEM_MESSAGE=("You are answering a multiple-choice survey question. "
                "Return only the number of the best answer option. Do not explain.")


def read_jsonl(path):
    with open(path,encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]


def parse_answer(text):
    if text is None: return None
    s=str(text).strip()
    try:
        obj=json.loads(s)
        if isinstance(obj,dict):
            for k in ('answer','option','choice','selected_answer'):
                if k in obj:
                    m=re.search(r'\b([1-9][0-9]?)\b',str(obj[k]));
                    if m: return m.group(1)
    except Exception: pass
    m=re.search(r'\b([1-9][0-9]?)\b',s)
    return m.group(1) if m else None


async def one(client,row,args,sem):
    async with sem:
        err=''
        for attempt in range(1,args.retries+2):
            try:
                kwargs=dict(model=args.model,
                    messages=[{'role':'system','content':SYSTEM_MESSAGE},{'role':'user','content':row['prompt_text']}],
                    temperature=0.0,top_p=1.0)
                try:
                    resp=await client.chat.completions.create(**kwargs,max_completion_tokens=8)
                except TypeError:
                    resp=await client.chat.completions.create(**kwargs,max_tokens=8)
                raw=resp.choices[0].message.content or ''; pred=parse_answer(raw)
                gold=str(row.get('mcq_gold_answer','')).strip()
                return dict(row,model=args.model,status='ok' if pred else 'parse_error',raw_output=raw,
                            predicted_answer=pred or '',correct=int(pred==gold) if pred and gold else '',attempts=attempt,error_message='')
            except Exception as e:
                err=f'{type(e).__name__}: {e}'
                if attempt<=args.retries: await asyncio.sleep(min(30,2**(attempt-1)))
        return dict(row,model=args.model,status='error',raw_output='',predicted_answer='',correct='',attempts=args.retries+1,error_message=err)


async def amain(args):
    key=os.getenv(args.api_key_env)
    if args.backend=='openai' and not key: raise SystemExit(f'Missing {args.api_key_env}')
    client=AsyncOpenAI(api_key=key or 'EMPTY',base_url=args.base_url or (None if args.backend=='openai' else 'http://localhost:8000/v1'),timeout=args.request_timeout)
    rows=read_jsonl(args.input_jsonl); sem=asyncio.Semaphore(args.concurrency)
    results=await asyncio.gather(*(one(client,r,args,sem) for r in rows))
    out=Path(args.output_jsonl); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in results: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    ok=[r for r in results if r.get('correct')!='']
    print(f'Wrote {len(results)} rows; scored={len(ok)}; accuracy={sum(int(r["correct"]) for r in ok)/len(ok) if ok else float("nan"):.4f}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl',required=True); p.add_argument('--output-jsonl',required=True)
    p.add_argument('--backend',choices=['openai','vllm'],default='vllm'); p.add_argument('--model',required=True)
    p.add_argument('--base-url'); p.add_argument('--api-key-env',default='OPENAI_API_KEY')
    p.add_argument('--concurrency',type=int,default=8); p.add_argument('--retries',type=int,default=3)
    p.add_argument('--request-timeout',type=float,default=120)
    args=p.parse_args(); asyncio.run(amain(args))


if __name__=='__main__': main()
