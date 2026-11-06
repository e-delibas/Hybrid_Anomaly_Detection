from pathlib import Path
import sys,json,platform,importlib.metadata as im
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from source_check import check_sources,sha256


def prepare(project,out,plan):
    result=check_sources(project,plan)
    if result['status']!='PASS_SOURCES':
        raise ValueError('Onaylı veri/protokol dosyası eksik veya değişmiş: '+str(result))
    directory=project/'revision_outputs/02_FD001_protocol'
    ref=pd.read_csv(directory/'train_evaluation_reference.csv')
    splits=pd.read_csv(directory/'engine_splits.csv')
    roles=ref.engine_key.map(splits.loc[splits.scheme.eq('cv') & splits.fold.eq(plan['fold'])].set_index('engine_key').role)
    raw=pd.read_csv(project/'data/train_FD001.txt',sep=r'\s+',header=None)
    sensors=raw.iloc[:,5:].to_numpy(dtype=np.float32)
    masks={r:roles.eq(r).to_numpy() for r in ['fit','early_stop','selection']}
    scaler=StandardScaler().fit(sensors[masks['fit']])
    arrays={'scaler_mean':scaler.mean_, 'scaler_scale':scaler.scale_}
    for role,mask in masks.items():
        arrays['X_'+role]=scaler.transform(sensors[mask]).astype(np.float32)
        arrays['keys_'+role]=ref.loc[mask,'sample_key'].astype(str).to_numpy(dtype=str)
    # Only 30 positive proxy labels are exposed to second-stage fitting.
    # Other fit examples remain in the pool, including unlabelled late-life cycles.
    fit_ref=ref.loc[masks['fit']].reset_index(drop=True)
    positive_pool=fit_ref.loc[fit_ref.RUL.le(30)]
    rng=np.random.default_rng(plan['seed'])
    engines=sorted(positive_pool.engine_key.unique())
    chosen_engines=rng.choice(engines,size=30,replace=False)
    indices=np.array([rng.choice(positive_pool.index[positive_pool.engine_key.eq(engine)]) for engine in chosen_engines],dtype=np.int64)
    weak=np.zeros(len(fit_ref),dtype=np.int64);weak[indices]=1
    arrays['weak_y_fit']=weak
    arrays['y_selection']=ref.loc[masks['selection'],'RUL'].le(30).to_numpy(dtype=np.int64)
    if weak.sum()!=30 or len(set(chosen_engines))!=30:
        raise ValueError('30 farklı fit motoru etiket bütçesi karşılanmadı.')
    # Outer features and labels deliberately have no key in this training export.
    np.savez_compressed(out/'prepared_data.npz',**arrays)
    label_records=fit_ref.loc[indices,['engine_key','sample_key','RUL']].to_dict('records')
    result.update(status='PASS_PREPARE',fold=plan['fold'],seed=plan['seed'],
        training_label_budget=30,known_positive_samples=label_records,
        other_fit_samples='retained as unlabelled; no true-label-based cleaning',
        exported_roles=list(masks),outer_data_exported=False,
        input_standardization='StandardScaler fitted on fitting engines only; 21 sensor columns in original order',
        versions={'python':platform.python_version(),**{k:im.version(k) for k in ['numpy','pandas','scikit-learn','scipy','joblib']}},
        prepared_sha256=sha256(out/'prepared_data.npz'))
    (out/'prepare_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print('HAZIR: fit/early_stop/selection; 30 pozitif vekil etiket; dış motor verisi aktarılmadı.',flush=True)


if __name__=='__main__':
    out=Path(sys.argv[1]);plan=json.loads((out/'plan.json').read_text())
    prepare(Path(sys.argv[2]),out,plan)
