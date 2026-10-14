#!/usr/bin/env python3
"""ChronoCulture pair-level evaluation.

Compute all valid pair metrics first, then apply an external knowledge-filter gate.
This keeps metric computation independent of filtering and makes gate changes auditable.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

EPS = 1e-12
VERSION = '3.0-clean'
GROUPS = ['moderate', 'strong', 'very strong']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output_dir(path):
    p = Path(path)
    require(not p.exists() or not any(p.iterdir()), f'Output directory must be new or empty: {p}')
    p.mkdir(parents=True, exist_ok=True)
    return p


def fingerprint(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def json_save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def vector(value, keys):
    d = json.loads(value) if isinstance(value, str) else value
    require(isinstance(d, dict), 'Distribution must be a JSON object')
    d = {str(k): v for k, v in d.items()}
    require(set(d) == set(keys), f'Options differ: expected {keys}, got {list(d)}')
    require(not any(isinstance(d[k], bool) for k in keys), 'Boolean probability')
    a = np.array([float(d[k]) for k in keys], dtype=float)
    require(np.isfinite(a).all() and (a >= 0).all(), 'Negative or nonfinite probability')
    total = a.sum()
    require(np.isfinite(total) and total > 0, 'Nonpositive/nonfinite distribution total')
    return a / total


def change(old, current):
    return np.cumsum(current - old)[:-1]


def w1(p, q):
    return float(np.abs(change(p, q)).mean())


def category(h):
    # Snap numerical rounding at documented boundaries, never substantive changes.
    for boundary in (0.03, 0.10, 0.20):
        if abs(h - boundary) <= EPS:
            h = boundary
    return 'stable' if h <= .03 else 'moderate' if h < .10 else 'strong' if h < .20 else 'very strong'


def sign_label(x, positive, negative, tie='tie'):
    if not np.isfinite(x):
        return 'undefined'
    return positive if x > EPS else negative if x < -EPS else tie


def pair_metrics(go, gc, mo, mc, no_time=None, tau=.10):
    H, M = change(go, gc), change(mo, mc)
    hn, mn = float(np.linalg.norm(H)), float(np.linalg.norm(M))
    h, m = w1(go, gc), w1(mo, mc)
    cos = float(np.clip(np.dot(H, M) / hn / mn, -1, 1)) if hn > EPS and mn > EPS else np.nan
    alpha = float(np.dot(M, H) / hn**2) if hn > EPS else np.nan
    residual = float(np.linalg.norm(M - alpha * H) / hn) if hn > EPS else np.nan
    d_oo, d_oc, d_co, d_cc = w1(mo, go), w1(mo, gc), w1(mc, go), w1(mc, gc)
    old_score = (d_oc - d_oo) / h if h > EPS else np.nan
    cur_score = (d_co - d_cc) / h if h > EPS else np.nan
    order = (old_score + cur_score) / 2
    op = sign_label(d_oc - d_oo, 'older', 'current')
    cp = sign_label(d_co - d_cc, 'current', 'older')
    anchors = {('older','older'): 'older_anchored', ('current','current'): 'current_anchored',
               ('older','current'): 'separated', ('current','older'): 'reversed'}
    r = {'human_shift_h': h, 'model_shift_m': m, 'shift_category': category(h),
         'magnitude_ratio': m / h if category(h) != 'stable' else np.nan,
         'cosine': cos, 'cosine_defined': int(np.isfinite(cos)),
         'direction_label': 'no_model_movement' if mn <= EPS else sign_label(cos, 'positive_alignment', 'negative_alignment', 'orthogonal'),
         'no_model_movement': int(mn <= EPS), 'near_zero_movement': int(m <= .05 + EPS),
         'projection_alpha': alpha, 'projection_residual_ratio': residual,
         'd_oo': d_oo, 'd_oc': d_oc, 'd_co': d_co, 'd_cc': d_cc,
         'best_era_error': min(d_oo, d_cc), 'worst_era_error': max(d_oo, d_cc),
         'era_error_gap': abs(d_oo - d_cc),
         'both_eras_within_tau': int(max(d_oo, d_cc) <= tau + EPS),
         'either_era_within_tau': int(min(d_oo, d_cc) <= tau + EPS),
         'old_era_score': old_score, 'current_era_score': cur_score, 'temporal_order_score': order,
         'anchoring': anchors.get((op, cp), 'tie'),
         'gold_change_json': json.dumps(H.tolist()), 'model_change_json': json.dumps(M.tolist()),
         'has_no_time': int(no_time is not None)}
    if no_time is not None:
        dn_o, dn_c = w1(no_time, go), w1(no_time, gc)
        old_move, cur_move = w1(no_time, mo), w1(no_time, mc)
        r.update(no_time_distance_old=dn_o, no_time_distance_current=dn_c,
                 no_time_endpoint_anchor=sign_label(dn_o-dn_c, 'current', 'older'),
                 old_cue_movement=old_move, current_cue_movement=cur_move,
                 old_cue_benefit=dn_o-d_oo, current_cue_benefit=dn_c-d_cc,
                 old_cue_unchanged=int(np.max(np.abs(no_time-mo)) <= EPS),
                 current_cue_unchanged=int(np.max(np.abs(no_time-mc)) <= EPS))
    return r


def load_crosswalk(path):
    maps = {'pair': defaultdict(lambda: defaultdict(set)), 'prompt': defaultdict(lambda: defaultdict(set))}
    if not path:
        return maps
    cw = read_csv(path)
    require('source_pair_id' in cw and 'pair_id' in cw, 'Crosswalk needs source_pair_id,pair_id')
    for r in cw.to_dict('records'):
        ds = r.get('dataset', '').strip().lower()
        for kind in ('pair', 'prompt'):
            src, dst = r.get('source_'+kind+'_id', ''), r.get(kind+'_id', '')
            if src and dst:
                maps[kind][src][ds].add(dst)
    return maps


def mapped(maps, kind, key, dataset=''):
    if not key:
        return ''
    candidates = set()
    for ds, dst in maps[kind].get(key, {}).items():
        if not dataset or not ds or ds == dataset:
            candidates.update(dst)
    require(len(candidates) <= 1, f'Ambiguous crosswalk for {key}: {candidates}')
    return next(iter(candidates)) if candidates else key


def load_benchmark(path):
    pairs, prompts = {}, {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid, prompt, era = r['pair_id'], r['prompt_id'], r['wave_type']
            require(era in ('older','current'), 'Benchmark must contain explicit older/current rows')
            require(prompt not in prompts and era not in pairs.get(pid, {}), f'Duplicate benchmark ID: {prompt}')
            pairs.setdefault(pid, {})[era] = r
            prompts[prompt] = (pid, era)
    require(bool(pairs), 'Empty benchmark')
    for pid, eras in pairs.items():
        require(set(eras) == {'older','current'}, f'Incomplete benchmark pair: {pid}')
    return pairs, prompts


def prediction_index(paths, pairs, prompts, maps, no_time, issues):
    index, model_ids = {}, set()
    for path in paths:
        df = read_csv(path)
        require('predicted_distribution_json' in df and 'status' in df, f'{path}: needs status and predicted_distribution_json')
        for line, r in enumerate(df.to_dict('records'), 2):
            if r.get('model', '').strip():
                model_ids.add(r['model'].strip())
            try:
                ds = r.get('dataset', '').strip().lower()
                pid = mapped(maps, 'pair', r.get('pair_id',''), ds)
                prompt = mapped(maps, 'prompt', r.get('prompt_id',''), ds)
                era = r.get('wave_type', '')
                if no_time:
                    require(era in ('', 'no_time'), 'No-time input contains a dated row')
                    if not pid and prompt.endswith('__no_time'):
                        pid = prompt[:-len('__no_time')]
                    era = 'no_time'
                elif prompt in prompts:
                    p_pid, p_era = prompts[prompt]
                    require(not pid or pid == p_pid, 'pair_id disagrees with prompt_id')
                    require(not era or era == p_era, 'wave_type disagrees with prompt_id')
                    pid, era = p_pid, p_era
                require(pid in pairs, 'Unmatched pair ID; supply an explicit source-to-release crosswalk')
                require(not ds or ds == pairs[pid]['older']['dataset'], 'Dataset mismatch')
                require(no_time or era in ('older','current'), 'Explicit input requires older/current rows')
                key = (pid, era)
            except (ValueError, KeyError) as e:
                issues.append({'file':str(path),'line':line,'pair_id':r.get('pair_id',''),'issue':str(e)})
                continue
            require(key not in index, f'Duplicate prediction for {key}; resolve repeated runs explicitly')
            index[key] = r
    require(len(model_ids) <= 1, f'Mixed model IDs in input set: {model_ids}; analyze one model at a time')
    return index, sorted(model_ids)


def valid_prediction(row, keys):
    require(row is not None, 'Missing prediction')
    require(row['status'].lower() in ('ok','ok_with_parse_failures'), 'Prediction status: '+row['status'])
    return vector(row['predicted_distribution_json'], keys)


def compute(args):
    out = output_dir(args.out_dir)
    pairs, prompts = load_benchmark(args.benchmark)
    maps, issues = load_crosswalk(args.crosswalk), []
    exp, exp_ids = prediction_index(args.predictions, pairs, prompts, maps, False, issues)
    nt, nt_ids = prediction_index(args.no_time or [], pairs, prompts, maps, True, issues)
    if exp_ids and nt_ids and exp_ids != nt_ids:
        require(args.allow_model_alias_difference, 'Explicit/no-time model IDs differ. Confirm same checkpoint, then use --allow-model-alias-difference.')
    records, audit = [], []
    for pid, eras in pairs.items():
        old, cur = eras['older'], eras['current']
        base = {'model_name':args.model_name, 'dataset':old['dataset'], 'pair_id':pid,
                'variable':old.get('variable',''), 'variable_title':old.get('variable_title',''),
                'topic':old.get('topic',''), 'country':old.get('country_name',''),
                'older_period':old.get('time_period',''), 'current_period':cur.get('time_period','')}
        check = dict(base, eval_ok=0, no_time_status='not_supplied' if not args.no_time else 'missing', error='')
        try:
            for k in ('dataset','variable','country_name','scale_min','scale_max','valid_options','response_options'):
                require(old.get(k) == cur.get(k), f'Inconsistent benchmark {k}')
            lo, hi = int(old['scale_min']), int(old['scale_max'])
            keys = [str(i) for i in range(lo, hi+1)]
            require(len(keys) >= 2 and old.get('valid_options', keys) == keys,
                    'Requires consecutive equally spaced valid response codes in numerical order')
            go, gc = vector(old['gold_distribution'], keys), vector(cur['gold_distribution'], keys)
            mo = valid_prediction(exp.get((pid,'older')), keys)
            mc = valid_prediction(exp.get((pid,'current')), keys)
            no = None
            if args.no_time:
                try:
                    no = valid_prediction(nt.get((pid,'no_time')), keys)
                    check['no_time_status'] = 'ok'
                except (ValueError, TypeError, KeyError) as e:
                    check['no_time_status'] = str(e)
            metrics = pair_metrics(go, gc, mo, mc, no, args.tau)
            reported = old.get('gold_shift_w1')
            check['gold_w1_difference'] = abs(float(reported)-metrics['human_shift_h']) if reported is not None else np.nan
            require(reported is None or (np.isfinite(float(reported)) and check['gold_w1_difference'] <= args.gold_tolerance),
                    'Recomputed gold W1 disagrees with benchmark; reconcile gold before use')
            source_cat = str(old.get('gold_shift_category','')).lower().replace('_',' ')
            check['source_category'] = source_cat
            check['recomputed_category'] = metrics['shift_category']
            check['category_mismatch'] = int(bool(source_cat) and source_cat != metrics['shift_category'])
            check['source_scale_points'] = old.get('scale_points','')
            check['actual_scale_points'] = len(keys)
            records.append(dict(base, scale_points=len(keys), alignment_tau=args.tau,
                                benchmark_gold_shift_w1=reported, **metrics))
            check['eval_ok'] = 1
        except (ValueError, TypeError, KeyError, OverflowError) as e:
            check['error'] = str(e)
        audit.append(check)
    pd.DataFrame(audit).to_csv(out/'pair_audit.csv', index=False)
    pd.DataFrame(issues, columns=['file','line','pair_id','issue']).to_csv(out/'input_issues.csv', index=False)
    require(bool(records), f'No evaluable pairs. Inspect {out}/pair_audit.csv and input_issues.csv')
    pd.DataFrame(records).to_csv(out/'pair_metrics_all.csv', index=False)
    manifest = {'version':VERSION, 'mode':'compute', 'model_name':args.model_name,
                'benchmark':fingerprint(args.benchmark), 'predictions':[fingerprint(p) for p in args.predictions],
                'no_time':[fingerprint(p) for p in args.no_time or []],
                'crosswalk':fingerprint(args.crosswalk) if args.crosswalk else None,
                'explicit_model_ids':exp_ids, 'no_time_model_ids':nt_ids,
                'benchmark_pairs':len(pairs), 'evaluable_pairs':len(records), 'input_issues':len(issues),
                'alignment_tau':args.tau, 'gold_tolerance':args.gold_tolerance,
                'eps':EPS, 'gate_applied':False, 'tables_created':False}
    json_save(out/'compute_manifest.json', manifest)
    print(f'Saved {len(records)}/{len(pairs)} valid pairs; {len(issues)} unmatched/invalid input rows. No gate or tables applied.')


def parse_bool(value):
    value = str(value).strip().lower()
    require(value in ('true','false','1','0','1.0','0.0','yes','no','pass','fail'), f'Unrecognized gate flag: {value!r}')
    return value in ('true','1','1.0','yes','pass')


def join_gate(df, args):
    if args.unfiltered:
        return df.assign(gate_status='unfiltered', retained=1), {'gate_mode':'explicitly_unfiltered'}
    gate = read_csv(args.gate_csv)
    require(args.gate_pair_column in gate, 'Gate pair-ID column absent')
    if args.gate_model_column in gate:
        require(args.gate_model_value is not None, 'Gate has a model column: specify --gate-model-value exactly')
        gate = gate[gate[args.gate_model_column] == args.gate_model_value].copy()
    elif args.gate_model_value is not None:
        raise ValueError('Supplied --gate-model-value but model column is absent')
    require(len(gate) > 0, 'No gate rows selected')
    if not args.gate_is_pass_list:
        require(args.gate_pass_column in gate, 'Gate pass column absent; set --gate-pass-column, or explicitly use --gate-is-pass-list')
    maps = load_crosswalk(args.crosswalk)
    flags = {}
    known = set(df['pair_id'])
    for r in gate.to_dict('records'):
        pid = mapped(maps, 'pair', r[args.gate_pair_column], r.get('dataset','').lower())
        require(bool(pid) and pid not in flags, f'Empty or duplicate gate pair ID: {pid}')
        if pid in known and r.get('dataset'):
            require(r['dataset'].lower() == df.loc[df.pair_id == pid,'dataset'].iloc[0], 'Gate dataset disagrees with pair')
        flags[pid] = True if args.gate_is_pass_list else parse_bool(r[args.gate_pass_column])
    require(bool(known & flags.keys()), 'Zero gate/metric ID matches; check ID scheme, model selection, and crosswalk')
    def status(pid):
        if pid not in flags:
            return 'not_in_pass_list' if args.gate_is_pass_list else 'missing_gate'
        return 'pass' if flags[pid] else 'fail'
    result = df.copy()
    result['gate_status'] = [status(pid) for pid in result.pair_id]
    result['retained'] = (result.gate_status == 'pass').astype(int)
    return result, {'gate_mode':'pass_list' if args.gate_is_pass_list else 'flags',
                    'selected_gate_rows':len(flags), 'gate_rows_without_evaluable_pair':len(flags.keys()-known),
                    'missing_gate_rows':int((result.gate_status == 'missing_gate').sum()),
                    'gate_model_value':args.gate_model_value}


def pct(a):
    return float(np.asarray(a, dtype=float).mean()*100) if len(a) else np.nan


def summaries(df):
    for dataset in ['all'] + sorted(df.dataset.unique().tolist()):
        ds = df if dataset == 'all' else df[df.dataset == dataset]
        for label in GROUPS + ['all_shifted']:
            g = ds[ds.shift_category.isin(GROUPS)] if label == 'all_shifted' else ds[ds.shift_category == label]
            if len(g):
                yield {'model_name':g.model_name.iloc[0], 'dataset':dataset, 'shift_category':label}, g


def core_summary(g):
    cos = g.cosine.dropna()
    return {'n_pairs':len(g), 'n_cosine_defined':len(cos), 'n_cosine_undefined':len(g)-len(cos),
            'mean_h':g.human_shift_h.mean(), 'mean_m':g.model_shift_m.mean(),
            'median_h':g.human_shift_h.median(), 'median_m':g.model_shift_m.median(),
            'median_magnitude_ratio':g.magnitude_ratio.median(),
            'median_magnitude_percent':100*g.magnitude_ratio.median(),
            'underreaction_pct':pct(g.magnitude_ratio < 1),
            'severe_underreaction_pct':pct(g.magnitude_ratio < .5),
            'overreaction_pct':pct(g.magnitude_ratio > 1),
            'no_movement_pct':pct(g.no_model_movement), 'near_zero_movement_pct':pct(g.near_zero_movement),
            'mean_cosine':cos.mean(), 'median_cosine':cos.median(),
            'positive_cosine_pct_defined':pct(cos > EPS), 'negative_cosine_pct_defined':pct(cos < -EPS),
            'orthogonal_cosine_pct_defined':pct(np.abs(cos) <= EPS)}


def permutation(g, n, rng, kind):
    d = g[g.cosine.notna()].copy()
    strata = []
    for _, sub in d.groupby(['dataset','shift_category','scale_points'], sort=True):
        H = np.array([json.loads(x) for x in sub.gold_change_json])
        M = np.array([json.loads(x) for x in sub.model_change_json])
        H /= np.linalg.norm(H, axis=1)[:,None]
        M /= np.linalg.norm(M, axis=1)[:,None]
        strata.append((H,M))
    eligible = [(h,m) for h,m in strata if len(h) >= 2] if kind == 'pair_shuffle' else strata
    count = sum(len(h) for h,m in eligible)
    result = {'test':kind, 'n_cosine_defined':len(d), 'n_tested':count,
              'n_excluded_singletons':len(d)-count, 'n_permutations':n,
              'strata':'dataset + gold shift category + actual response option count'}
    if not count or n == 0:
        return dict(result, test_status='not_testable' if not count else 'disabled')
    observed = sum(float((h*m).sum()) for h,m in eligible)/count
    null = np.empty(n)
    values = np.concatenate([(h*m).sum(axis=1) for h,m in eligible])
    for i in range(n):
        if kind == 'pair_shuffle':
            null[i] = sum(float((h*m[rng.permutation(len(m))]).sum()) for h,m in eligible)/count
        else:
            null[i] = np.dot(values, rng.choice([-1,1],size=count))/count
    result.update(test_status='ok', observed_mean_cosine=observed, null_mean=float(null.mean()),
                  null_sd=float(null.std()), observed_minus_null=observed-float(null.mean()),
                  null_95th_percentile=float(np.quantile(null,.95)),
                  p_one_sided=float((1+(null >= observed-1e-15).sum())/(n+1)))
    return result


def tables(df, out, args):
    main, diagnostic, floors, tests = [], [], [], []
    rng = np.random.default_rng(args.seed)
    for meta,g in summaries(df):
        main.append(dict(meta, **core_summary(g)))
        diagnostic.append(dict(meta, n_pairs=len(g), median_projection_alpha=g.projection_alpha.median(),
            median_projection_residual_ratio=g.projection_residual_ratio.median(),
            median_old_era_score=g.old_era_score.median(), median_current_era_score=g.current_era_score.median(),
            median_temporal_order_score=g.temporal_order_score.median(),
            positive_order_pct=pct(g.temporal_order_score > EPS),
            **{k+'_pct':pct(g.anchoring == k) for k in ['older_anchored','current_anchored','separated','reversed','tie']}))
        for floor in (.01,.02,.05):
            moving = g[g.model_shift_m > floor]
            c = moving.cosine.dropna()
            floors.append(dict(meta, movement_floor=floor, n_total=len(g), n_above_floor=len(moving),
                               n_cosine_defined=len(c), above_floor_pct=100*len(moving)/len(g),
                               mean_cosine=c.mean(), median_cosine=c.median(), positive_cosine_pct_defined=pct(c>EPS)))
        if args.permutations:
            for kind in ['pair_shuffle'] + (['temporal_signflip'] if args.signflip else []):
                tests.append(dict(meta, **permutation(g,args.permutations,rng,kind)))
    pd.DataFrame(main).to_csv(out/'direction_magnitude.csv',index=False)
    pd.DataFrame(diagnostic).to_csv(out/'supporting_diagnostics.csv',index=False)
    pd.DataFrame(floors).to_csv(out/'movement_floor_sensitivity.csv',index=False)
    if tests:
        pd.DataFrame(tests).to_csv(out/'permutation_tests.csv',index=False)
    scale = []
    shifted = df[df.shift_category.isin(GROUPS)]
    for (ds,k),g in shifted.groupby(['dataset','scale_points']):
        scale.append(dict(dataset=ds,scale_points=k,**core_summary(g)))
    pd.DataFrame(scale).to_csv(out/'direction_magnitude_by_scale.csv',index=False)
    stable, alignment, no_time = [], [], []
    for ds in ['all'] + sorted(df.dataset.unique().tolist()):
        d = df if ds == 'all' else df[df.dataset == ds]
        g = d[d.shift_category == 'stable']
        if len(g):
            stable.append(dict(dataset=ds,n_pairs=len(g),
                preserved_stability_pct=pct(g.model_shift_m<=.03+EPS),
                small_false_drift_pct=pct((g.model_shift_m>.03+EPS)&(g.model_shift_m<.10-EPS)),
                large_false_drift_pct=pct(g.model_shift_m>=.10-EPS)))
        for cat,g in d.groupby('shift_category'):
            alignment.append(dict(dataset=ds,shift_category=cat,n_pairs=len(g),tau=g.alignment_tau.iloc[0],
                mean_d_old=g.d_oo.mean(),mean_d_current=g.d_cc.mean(),mean_best=g.best_era_error.mean(),
                mean_worst=g.worst_era_error.mean(),mean_gap=g.era_error_gap.mean(),
                both_eras_within_tau_pct=pct(g.both_eras_within_tau),either_era_within_tau_pct=pct(g.either_era_within_tau)))
            nt = g[g.has_no_time == 1]
            if len(nt):
                for era in ['old','current']:
                    b = nt[era+'_cue_benefit']
                    no_time.append(dict(dataset=ds,shift_category=cat,era=era,n_pairs=len(nt),
                        mean_cue_movement=nt[era+'_cue_movement'].mean(),mean_cue_benefit=b.mean(),
                        helped_pct=pct(b>EPS),hurt_pct=pct(b< -EPS),equal_error_pct=pct(np.abs(b)<=EPS),
                        unchanged_distribution_pct=pct(nt[era+'_cue_unchanged']),
                        endpoint_old_pct=pct(nt.no_time_endpoint_anchor=='older'),
                        endpoint_current_pct=pct(nt.no_time_endpoint_anchor=='current'),
                        endpoint_tie_pct=pct(nt.no_time_endpoint_anchor=='tie')))
    pd.DataFrame(stable).to_csv(out/'stable_false_drift.csv',index=False)
    pd.DataFrame(alignment).to_csv(out/'period_error_summary.csv',index=False)
    if no_time:
        pd.DataFrame(no_time).to_csv(out/'date_cue_diagnostics.csv',index=False)


def summarize(args):
    out = output_dir(args.out_dir)
    df = pd.read_csv(args.metrics,keep_default_na=True)
    require(len(df)>0 and df.model_name.nunique()==1, 'Metrics must contain exactly one model')
    require(not df.pair_id.duplicated().any(), 'Duplicate metric pair IDs')
    require(df.alignment_tau.nunique()==1, 'Mixed alignment tolerances')
    joined, info = join_gate(df,args)
    joined.to_csv(out/'pair_metrics_with_gate.csv',index=False)
    retained = joined[joined.retained == 1].copy()
    retained.to_csv(out/'pair_metrics_retained.csv',index=False)
    coverage = joined.groupby(['model_name','dataset','shift_category','gate_status'],dropna=False).size().reset_index(name='n_pairs')
    coverage.to_csv(out/'gate_coverage.csv',index=False)
    manifest = dict(version=VERSION,mode='summarize',metrics=fingerprint(args.metrics),
                    gate=fingerprint(args.gate_csv) if args.gate_csv else None,
                    crosswalk=fingerprint(args.crosswalk) if args.crosswalk else None,
                    gate_pair_column=args.gate_pair_column,gate_pass_column=args.gate_pass_column,
                    gate_model_column=args.gate_model_column,
                    model_name=df.model_name.iloc[0],input_evaluable_pairs=len(df),retained_pairs=len(retained),
                    permutations=args.permutations,seed=args.seed,signflip=args.signflip,**info)
    json_save(out/'summary_manifest.json',manifest)
    require(len(retained)>0, 'No retained pairs; gate audit saved, no tables generated')
    tables(retained,out,args)
    print(f'Saved tables for {len(retained)}/{len(df)} evaluable pairs. Gate coverage: {out}/gate_coverage.csv')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('compute',help='Save all valid pair metrics, without tables or filtering')
    c.add_argument('--benchmark',required=True)
    c.add_argument('--predictions',action='append',required=True,help='Explicit CSV; repeat for files from the SAME model')
    c.add_argument('--no-time',action='append',help='Optional no-time CSV from the same checkpoint; repeat for WVS/ISSP')
    c.add_argument('--model-name',required=True,help='Analysis label, e.g. llama33_70b')
    c.add_argument('--crosswalk',help='CSV with source_pair_id,pair_id and optional dataset,source_prompt_id,prompt_id')
    c.add_argument('--allow-model-alias-difference',action='store_true')
    c.add_argument('--tau',type=float,default=.10)
    c.add_argument('--gold-tolerance',type=float,default=1e-6)
    c.add_argument('--out-dir',required=True)
    s=sub.add_parser('summarize',help='Apply a supplied knowledge-filter gate before summaries/tests')
    s.add_argument('--metrics',required=True)
    gate=s.add_mutually_exclusive_group(required=True)
    gate.add_argument('--gate-csv')
    gate.add_argument('--unfiltered',action='store_true',help='Explicitly request exploratory ungated tables')
    s.add_argument('--gate-pair-column',default='pair_id')
    s.add_argument('--gate-pass-column',default='passes')
    s.add_argument('--gate-model-column',default='model')
    s.add_argument('--gate-model-value')
    s.add_argument('--gate-is-pass-list',action='store_true',help='Explicitly declare every gate row passes; omitted IDs fail')
    s.add_argument('--crosswalk')
    s.add_argument('--permutations',type=int,default=5000,help='0 disables tests; default uses pair shuffling within dataset/category/K')
    s.add_argument('--signflip',action='store_true',help='Also run the older temporal-label sign-flip null')
    s.add_argument('--seed',type=int,default=7)
    s.add_argument('--out-dir',required=True)
    return p


def main():
    args=parser().parse_args()
    if args.command=='compute':
        require(0<=args.tau<=1 and args.gold_tolerance>=0,'Invalid tolerance')
        compute(args)
    else:
        require(args.permutations>=0,'Permutations must be nonnegative')
        summarize(args)


if __name__=='__main__':
    try:
        main()
    except (ValueError,KeyError) as e:
        print('ERROR:',e,file=sys.stderr)
        sys.exit(2)
