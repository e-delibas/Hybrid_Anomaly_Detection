"""Validate the numerical supplement using only the Python standard library."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
METRICS = ['AP', 'ROC_AUC', 'precision', 'recall', 'F1', 'Macro_F1']
KEYS = ['dataset', 'preprocessing', 'representation', 'method', 'RUL_window']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(name):
    with (ROOT/name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def close(a, b, context):
    require(math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12), context)


def main():
    folds, aggregates = read_csv('fold_metrics.csv'), read_csv('aggregate_metrics.csv')
    require(len(folds) == 13860 and len(aggregates) == 462, 'Unexpected record counts')
    groups, paired, seen = defaultdict(list), defaultdict(dict), set()
    expected_runs = {(str(s), str(f)) for s in [42,43,44] for f in range(1,11)}
    for r in folds:
        key = tuple(r[c] for c in KEYS)
        run = (r['seed'], r['fold'])
        require((key,run) not in seen, 'Duplicate fold record')
        seen.add((key,run))
        groups[key].append(r)
        paired[key[:-1]+run][r['RUL_window']] = r
        require(all(math.isfinite(float(r[m])) and 0 <= float(r[m]) <= 1 for m in METRICS), 'Invalid metric')
        tp,fp,tn,fn = [int(r[c]) for c in ['TP','FP','TN','FN']]
        require(min(tp,fp,tn,fn) >= 0, 'Negative confusion count')
        div = lambda a,b: a/b if b else 0.0
        values = {'precision':div(tp,tp+fp), 'recall':div(tp,tp+fn),
                  'F1':div(2*tp,2*tp+fp+fn),
                  'Macro_F1':(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn))/2}
        for m,v in values.items():
            close(r[m],v,'Confusion metric mismatch: '+m)
    aggregate_keys=set()
    for r in aggregates:
        key=tuple(r[c] for c in KEYS)
        require(key not in aggregate_keys, 'Duplicate aggregate record')
        aggregate_keys.add(key)
        rows=groups[key]
        require(len(rows)==int(r['runs'])==30, 'Incomplete aggregate')
        require({(x['seed'],x['fold']) for x in rows}==expected_runs, 'Missing fold/seed')
        for m in METRICS:
            values=[float(x[m]) for x in rows]
            close(statistics.mean(values),r[m+'_mean'], 'Mean mismatch: '+m)
            close(statistics.stdev(values),r[m+'_descriptive_sd'], 'SD mismatch: '+m)
    require(aggregate_keys==set(groups), 'Aggregate/fold key mismatch')
    classical={'DBSCAN','IF','LOF','OC-SVM','kNN'}
    coverage=defaultdict(set)
    for r in aggregates:
        if r['method'] in classical:
            coverage[(r['dataset'],r['preprocessing'],r['representation'],r['RUL_window'])].add(r['method'])
    require(all(s==classical for s in coverage.values()), 'Missing classical detector')
    for ws in paired.values():
        require(set(ws)=={'20','30','40'}, 'Missing RUL window')
        b=ws['30']
        for r in ws.values():
            close(r['threshold'],b['threshold'], 'Threshold changed between windows')
            require(json.loads(r['selected_parameters_json'])==json.loads(b['selected_parameters_json']), 'Parameters changed')
            require(int(r['TP'])+int(r['FP'])==int(b['TP'])+int(b['FP']), 'Alarm count changed')
            require(sum(int(r[c]) for c in ['TP','FP','TN','FN'])==sum(int(b[c]) for c in ['TP','FP','TN','FN']), 'Sample count changed')
    lookup={tuple(r[c] for c in KEYS):r for r in aggregates}
    wide=read_csv('AP_F1_by_RUL_window.csv')
    require(len(wide)==154, 'Unexpected wide-table coverage')
    wide_seen=set()
    for r in wide:
        base=tuple(r[c] for c in KEYS[:-1])
        require(base not in wide_seen, 'Duplicate wide-table row')
        wide_seen.add(base)
        for w in ['20','30','40']:
            source=lookup[base+(w,)]
            for m in ['AP','F1']:
                close(r[m+'_'+w+'_mean'],source[m+'_mean'],'Wide mean mismatch')
                close(r[m+'_'+w+'_sd'],source[m+'_descriptive_sd'],'Wide SD mismatch')
    manifest=ROOT/'SHA256SUMS.txt'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest, name=line.split('  ',1)
            require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, 'Fingerprint mismatch: '+name)
    print('PASS: 462 aggregate rows; 13,860 fold rows; 154 window comparisons.')
    print('All five classical detectors retained. Means, sample SDs and fixed-window checks passed.')
    print('Scope: exported records; no model fitting or per-cycle AP/ROC-AUC recomputation.')


if __name__=='__main__':
    main()
