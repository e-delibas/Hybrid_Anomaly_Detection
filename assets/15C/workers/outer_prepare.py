from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from source_check import check_sources,sha256


def prepare(source,out,project,expected):
    plan=json.loads((source/'plan.json').read_text())
    checked=check_sources(project,plan)
    if checked['status']!='PASS_SOURCES':raise ValueError('Source verification failed')
    prep=json.loads((source/'prepare_summary.json').read_text())
    if sha256(source/'prepared_data.npz')!=prep['prepared_sha256']:raise ValueError('Prepared data changed')
    data=np.load(source/'prepared_data.npz',allow_pickle=False)
    ref=pd.read_csv(project/'revision_outputs/02_FD001_protocol/train_evaluation_reference.csv')
    roles=pd.read_csv(project/'revision_outputs/02_FD001_protocol/engine_splits.csv')
    fold=roles.loc[roles.scheme.eq('cv') & roles.fold.eq(expected['fold'])].set_index('engine_key').role
    mapped=ref.engine_key.map(fold)
    if mapped.isna().any():raise ValueError('Unknown motor role')
    raw=pd.read_csv(project/'data/train_FD001.txt',sep=r'\s+',header=None).iloc[:,5:].to_numpy(dtype=np.float32)
    scaler=StandardScaler()
    scaler.mean_=data['scaler_mean'];scaler.scale_=data['scaler_scale']
    scaler.n_features_in_=21;scaler.n_samples_seen_=len(data['X_fit'])
    for role in ('fit','early_stop','selection'):
        mask=mapped.eq(role).to_numpy()
        if not np.array_equal(ref.loc[mask,'sample_key'].to_numpy(dtype=str),data['keys_'+role]):raise ValueError('Sample identity differs '+role)
        if not np.array_equal(scaler.transform(raw[mask]),data['X_'+role]):raise ValueError('Saved input transformation differs '+role)
    mask=mapped.eq('outer_valid').to_numpy()
    outer_ref=ref.loc[mask,['engine_key','sample_key','UnitNumber','Time','RUL']].copy()
    if outer_ref.engine_key.nunique()!=10 or not outer_ref.sample_key.is_unique:raise ValueError('Outer motor/sample coverage mismatch')
    if set(outer_ref.engine_key)&set(ref.loc[mapped.eq('fit'),'engine_key']):raise ValueError('Fit/outer engine overlap')
    np.savez_compressed(out/'outer_inputs.npz',X_outer=scaler.transform(raw[mask]).astype(np.float32),keys=outer_ref.sample_key.to_numpy(dtype=str))
    outer_ref.to_csv(out/'outer_reference.csv',index=False)
    result=dict(status='PASS_OUTER_PREPARE',fold=expected['fold'],seed=expected['seed'],outer_engines=sorted(outer_ref.engine_key.unique()),
                outer_rows=len(outer_ref),outer_positives={str(w):int(outer_ref.RUL.le(w).sum()) for w in (20,30,40)},
                saved_transform_reproduced=True,outer_inputs_sha256=sha256(out/'outer_inputs.npz'),reference_sha256=sha256(out/'outer_reference.csv'))
    (out/'outer_prepare_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('DIS_VERI_HAZIR',expected['seed'],expected['fold'],len(outer_ref),flush=True)


if __name__=='__main__':
    out=Path(sys.argv[1]);context=json.loads((out/'evaluation_context.json').read_text())
    prepare(Path(context['source']),out,Path(context['project']),context)
