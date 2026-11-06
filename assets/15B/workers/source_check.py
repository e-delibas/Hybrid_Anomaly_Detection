from pathlib import Path
import hashlib,json,csv,math
from collections import Counter
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def check_sources(project, config):
    findings = []
    for relative, expected in config['required_hashes'].items():
        path = project / relative
        item = {'relative_path': relative, 'expected_sha256': expected}
        if not path.is_file():
            item['status'] = 'MISSING'
        else:
            item['sha256'] = sha256(path)
            item['status'] = 'MATCH' if item['sha256'] == expected else 'DIFFERENT'
        findings.append(item)
    result = {'files': findings, 'status': 'SOURCE_REVIEW_REQUIRED'}
    if not all(x['status'] == 'MATCH' for x in findings):
        return result
    protocol = project / 'revision_outputs/02_FD001_protocol'
    with (protocol / 'engine_splits.csv').open(encoding='utf-8-sig', newline='') as stream:
        split_rows = list(csv.DictReader(stream))
    with (protocol / 'train_evaluation_reference.csv').open(encoding='utf-8-sig', newline='') as stream:
        reference = list(csv.DictReader(stream))
    if len(reference) != 20631:
        raise ValueError('FD001 referans satır sayısı 20631 değil.')
    raw_keys = []
    with (project / 'data/train_FD001.txt').open() as stream:
        for row in stream:
            if not row.strip():
                continue
            values = [float(x) for x in row.split()]
            if len(values) != 26 or not all(math.isfinite(x) for x in values):
                raise ValueError('FD001 satırında sütun sayısı veya sonluluk sorunu.')
            if values[0] != int(values[0]) or values[1] != int(values[1]):
                raise ValueError('Motor/çevrim kimliği tam sayı değil.')
            raw_keys.append((int(values[0]), int(values[1])))
    reference_keys = [(int(float(r['UnitNumber'])), int(float(r['Time']))) for r in reference]
    if raw_keys != reference_keys or len(set(raw_keys)) != len(raw_keys):
        raise ValueError('FD001 veri/referans sırası veya çevrim benzersizliği uyuşmuyor.')
    maxima = {}
    for engine, cycle in raw_keys:
        maxima[engine] = max(maxima.get(engine, 0), cycle)
    engine_keys = {r['engine_key'] for r in reference}
    if len(maxima) != 100 or len(engine_keys) != 100:
        raise ValueError('FD001 motor sayısı 100 değil.')
    engine_key_pairs = {(int(float(r['UnitNumber'])), r['engine_key']) for r in reference}
    if len(engine_key_pairs) != 100:
        raise ValueError('Motor kimliği ile engine_key eşlemesi birebir değil.')
    if len({r['sample_key'] for r in reference}) != len(reference):
        raise ValueError('sample_key benzersiz değil.')
    for r, (engine, cycle) in zip(reference, raw_keys):
        rul = maxima[engine] - cycle
        label = str(r['late_life_le_30']).strip().lower()
        if label not in ('0', '1', '0.0', '1.0', 'true', 'false'):
            raise ValueError('Bilinmeyen RUL etiket gösterimi.')
        positive = label in ('1', '1.0', 'true')
        if float(r['RUL']) != rul or positive != (rul <= 30):
            raise ValueError('RUL referansı ham çevrimlerle uyuşmuyor.')
    cv_rows = [r for r in split_rows if r['scheme'] == 'cv']
    if {int(float(r['fold'])) for r in cv_rows} != set(range(1, 11)):
        raise ValueError('CV fold kümesi 1–10 değil.')
    outer_seen, folds = Counter(), []
    for fold in range(1, 11):
        rows = [r for r in cv_rows if int(float(r['fold'])) == fold]
        roles = {r['engine_key']: r['role'] for r in rows}
        if len(rows) != 100 or set(roles) != engine_keys:
            raise ValueError(f'Fold {fold}: eksik veya yinelenen motor.')
        counts = Counter(roles.values())
        if counts != {'fit': 70, 'early_stop': 10, 'selection': 10, 'outer_valid': 10}:
            raise ValueError(f'Fold {fold}: 70/10/10/10 bölümü uyuşmuyor.')
        outer_seen.update(k for k, v in roles.items() if v == 'outer_valid')
        row_counts = Counter(roles[r['engine_key']] for r in reference)
        eligible = {r['engine_key'] for r in reference
                    if roles[r['engine_key']] == 'fit' and float(r['RUL']) <= 30}
        if len(eligible) < 30:
            raise ValueError(f'Fold {fold}: 30 farklı fit motorundan geç yaşam çevrimi seçilemiyor.')
        folds.append({'fold': fold, 'engine_counts': dict(counts),
                      'row_counts': dict(row_counts),
                      'fit_engines_eligible_for_30_proxy_labels': len(eligible)})
    if set(outer_seen) != engine_keys or any(v != 1 for v in outer_seen.values()):
        raise ValueError('Dış CV motor kapsamı eşleşmiyor.')
    result.update(status='PASS_SOURCES', folds=folds, n_cycles=len(raw_keys), n_engines=100,
                  label_definition='RUL <= 30; model-independent late-life proxy, not annotated anomaly onset')
    return result