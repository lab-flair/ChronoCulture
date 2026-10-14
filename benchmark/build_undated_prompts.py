#!/usr/bin/env python3
"""Build one undated verbalized-distribution prompt per ChronoCulture pair.

The transformation starts from the released dated ``prompt_text`` so response-
scale wording is preserved exactly. It removes only the dated country/time
phrase and the corresponding time-period wording in the distribution request.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path


def make_undated_prompt(row):
    s=str(row['prompt_text'])
    country=re.escape(str(row['country_name']))
    # Dated release prompts begin with forms such as:
    #   In Australia in 1993, ...
    #   In Thailand around 2005–2009, ...
    s,n=re.subn(rf'^In {country}(?: in| around| during) [^,]+,',f"In {row['country_name']},",s,count=1)
    if n!=1:
        raise ValueError(f"could not remove time phrase for {row['pair_id']}")
    needle='Instead of choosing one answer, estimate the distribution of responses among people in this country and time period.'
    replacement=('Instead of choosing one answer, estimate the distribution of responses among people in this country.\n'
                 'No specific time period is provided; use only the country and question context.')
    if needle not in s:
        raise ValueError(f"could not find dated distribution instruction for {row['pair_id']}")
    return s.replace(needle,replacement,1)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--benchmark',required=True); p.add_argument('--output-jsonl',required=True); a=p.parse_args()
    pairs={}
    with open(a.benchmark,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); pid=str(r['pair_id'])
            if pid not in pairs or r.get('wave_type')=='older': pairs[pid]=r
    out=Path(a.output_jsonl); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for pid,row in pairs.items():
            keep=['dataset','variable','variable_title','country_code_iso3','country_name','scale_min','scale_max','scale_points','question_text','valid_options','response_options','pair_id']
            o={k:row.get(k) for k in keep if k in row}
            o.update(prompt_id=pid+'__no_time',wave_type='no_time',target_wave='no_time',time_period='',target_year='',
                     country_alpha=row.get('country_code_iso3',''),prompt_style='approach_c_no_time_verbalized_distribution',
                     expected_output='json_distribution_percentages',prompt_text=make_undated_prompt(row))
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    print(f'Wrote {len(pairs):,} undated prompts to {out}')

if __name__=='__main__': main()
