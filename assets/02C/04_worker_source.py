"""One recorded FD001 fold: load saved AE, refit selected classical detectors, compare scores."""
from pathlib import Path
import json,sys,hashlib,traceback,platform


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(condition,message):
    if not bool(condition):raise ValueError(message)


def compare_array(name,actual,expected,rtol=1e-5,atol=1e-7):
    actual=np.asarray(actual);expected=np.asarray(expected)
    require(actual.shape==expected.shape,name+': shape mismatch')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),name+': nonfinite')
    return {'name':name,'shape':list(actual.shape),'rtol':rtol,'atol':atol,
            'max_abs_difference':float(np.max(np.abs(actual-expected))) if actual.size else 0.0,
            'pass':bool(np.allclose(actual,expected,rtol=rtol,atol=atol))}


def fit_candidate(method,params,X,seed):
    if method=='IF':
        model=IsolationForest(**params,max_samples='auto',contamination='auto',random_state=seed,n_jobs=-1)
    elif method=='LOF':
        require(params['n_neighbors']<len(X),'Insufficient fit rows for LOF')
        model=LocalOutlierFactor(**params,novelty=True,contamination='auto',metric='euclidean',n_jobs=-1)
    elif method=='kNN':
        require(params['n_neighbors']<=len(X),'Insufficient fit rows for kNN')
        model=NearestNeighbors(**params,metric='euclidean',n_jobs=-1)
    else:
        raise ValueError('Unsupported reproduction method: '+str(method))
    return model.fit(X)


def score(method,model,X):
    if method=='kNN':values=model.kneighbors(X,return_distance=True)[0][:,-1]
    elif method in ['IF','LOF']:values=-model.score_samples(X)
    else:raise ValueError('Unsupported scoring method: '+str(method))
    require(np.isfinite(values).all(),'Nonfinite scores')
    return values


def choose_threshold(y,scores):
    p,r,t=precision_recall_curve(y,scores)
    f=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(t),where=(p[:-1]+r[:-1])>0)
    i=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]
    return float(t[i])


def metrics(y,scores,threshold):
    pred=(scores>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'AP':float(average_precision_score(y,scores)),'ROC_AUC':float(roc_auc_score(y,scores)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'F1':float(f1_score(y,pred,zero_division=0)),
            'Macro_F1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'TP':int(tp),'FP':int(fp),'TN':int(tn),'FN':int(fn)}


def execute(config):
    global np,pd,tf,StandardScaler,IsolationForest,LocalOutlierFactor,NearestNeighbors
    global average_precision_score,roc_auc_score,precision_recall_curve,precision_score,recall_score,f1_score,confusion_matrix
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors
    from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,precision_score,recall_score,f1_score,confusion_matrix
    import sklearn
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'tensorflow':tf.__version__}
    for k,v in config['expected_versions'].items():require(versions[k]==v,'Runtime version mismatch: '+k)
    root=Path(config['revision_root']);data=Path(config['data_directory'])
    folder=root/config['fold_relative_directory'];protocol=root/'02_FD001_protocol'
    checked=[]
    for relative,expected in config['known_hashes'].items():
        p=root/relative;require(p.is_file() and digest(p)==expected,'Source file missing/changed: '+str(p));checked.append(relative)
    train_path=data/'train_FD001.txt'
    require(digest(train_path)==config['train_sha256'],'Training data hash changed')
    record=json.loads((folder/'fold_summary.json').read_text())
    require(record['seed']==42 and record['fold']==1 and record['status']=='PASS','Unexpected fold record')
    # The independently supplied 05 summary anchors this fold's primary results.
    expected_primary={(r['representation'],r['method']):r for r in config['approved_primary_results']}
    require(len(expected_primary)==7 and len(record['primary_results'])==7,'Expected seven primary score pipelines')
    for r in record['primary_results']:
        require(r==expected_primary[(r['representation'],r['method'])],'Fold record differs from supplied CV summary')
    predpath=folder/'outer_predictions.csv'
    require('outer_predictions.csv' in record['output_sha256'],'Prediction hash not recorded')
    require(digest(predpath)==record['output_sha256']['outer_predictions.csv'],'Prediction file hash mismatch')
    oldpred=pd.read_csv(predpath,float_precision='round_trip')
    require(not oldpred.duplicated(['representation','method','sample_key']).any(),'Duplicate prediction keys')
    sensors=[f'Sensor{i}' for i in range(1,22)]
    raw=pd.read_csv(train_path,sep=r'\s+',header=None)
    require(raw.shape==(20631,26),'Unexpected FD001 shape')
    raw.columns=['UnitNumber','Time','OpSet1','OpSet2','OpSet3']+sensors
    reference=pd.read_csv(protocol/'train_evaluation_reference.csv')
    require(len(reference)==len(raw) and np.array_equal(reference.source_row,np.arange(len(raw))),'Metadata row mismatch')
    require(np.array_equal(raw[['UnitNumber','Time']],reference[['UnitNumber','Time']]),'Metadata alignment mismatch')
    splits=pd.read_csv(protocol/'engine_splits.csv')
    fold=splits.loc[splits.scheme.eq('cv') & splits.fold.eq(1)]
    require(len(fold)==100 and fold.engine_key.is_unique,'Engine-role collision')
    require(fold.role.value_counts().to_dict()=={'fit':70,'early_stop':10,'selection':10,'outer_valid':10},'Unexpected role counts')
    roles=reference.engine_key.map(fold.set_index('engine_key').role)
    require(roles.notna().all(),'Engine missing from role map')
    masks={r:roles.eq(r).to_numpy() for r in ['fit','selection','outer_valid']}
    X=raw[sensors].to_numpy(dtype=np.float32)
    scaler=StandardScaler().fit(X[masks['fit']])
    Xs={r:scaler.transform(X[m]) for r,m in masks.items()}
    stored_scalers=json.loads((folder/'scaler_parameters.json').read_text())
    require(stored_scalers['input']['features']==sensors,'Sensor order changed')
    arrays=[compare_array('input_mean',scaler.mean_,stored_scalers['input']['mean'],1e-10,1e-12),
            compare_array('input_scale',scaler.scale_,stored_scalers['input']['scale'],1e-10,1e-12)]
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
    print('Kayıtlı AE ve encoder yükleniyor; AE eğitimi yapılmıyor.',flush=True)
    ae=tf.keras.models.load_model(folder/'autoencoder.keras',compile=False)
    encoder=tf.keras.models.load_model(folder/'encoder.keras',compile=False)
    require(ae.input_shape[-1]==21 and ae.output_shape[-1]==21 and encoder.output_shape[-1]==21,'Saved model dimensions differ')
    Z={r:encoder.predict(x,batch_size=256,verbose=0) for r,x in Xs.items()}
    for r,z in Z.items():require(z.shape==(len(Xs[r]),21) and np.isfinite(z).all(),'Bad latent representation '+r)
    embedded_encoder=tf.keras.Model(ae.input,ae.get_layer('latent').output)
    arrays.append(compare_array('encoder_vs_AE_latent',Z['fit'][:256],embedded_encoder.predict(Xs['fit'][:256],batch_size=256,verbose=0)))
    latent_scaler=StandardScaler().fit(Z['fit'])
    arrays.extend([compare_array('latent_mean',latent_scaler.mean_,stored_scalers['latent']['mean']),compare_array('latent_scale',latent_scaler.scale_,stored_scalers['latent']['scale'])])
    representations={'Raw':Xs,'AE_latent21':{r:latent_scaler.transform(z) for r,z in Z.items()}}
    mse={r:np.mean(np.square(Xs[r]-ae.predict(Xs[r],batch_size=256,verbose=0)),axis=1) for r in ['selection','outer_valid']}
    held=reference.loc[masks['outer_valid']].reset_index(drop=True)
    ys=reference.loc[masks['selection'],'late_life_le_30'].to_numpy(dtype=int)
    require(set(zip(oldpred.representation,oldpred.method))==set(expected_primary),'Unexpected prediction pipeline set')
    all_metrics={(r['representation'],r['method'],r['RUL_window']):r for r in record['primary_results']+record['sensitivity_results']}
    require(len(all_metrics)==21,'Expected 21 stored metric rows')
    rows=[]
    for (rep,method),old in expected_primary.items():
        print('Yeniden üretim:',rep,method,flush=True)
        if method=='MSE':ss,os=mse['selection'],mse['outer_valid']
        else:
            model=fit_candidate(method,old['selected_params'],representations[rep]['fit'],42)
            ss=score(method,model,representations[rep]['selection']);os=score(method,model,representations[rep]['outer_valid'])
        threshold=choose_threshold(ys,ss)
        oldframe=oldpred.loc[oldpred.representation.eq(rep)&oldpred.method.eq(method)].set_index('sample_key')
        require(len(oldframe)==len(held) and set(oldframe.index)==set(held.sample_key),'Prediction sample keys differ')
        oldframe=oldframe.loc[held.sample_key]
        for col in ['UnitNumber','Time','RUL','late_life_le_20','late_life_le_30','late_life_le_40']:
            require(np.array_equal(oldframe[col].to_numpy(),held[col].to_numpy()),'Prediction metadata mismatch: '+col)
        score_check=compare_array(rep+'/'+method+'/outer_scores',os,oldframe.score.to_numpy())
        threshold_check=compare_array(rep+'/'+method+'/selected_threshold',[threshold],[old['threshold']])
        selection_ap=float(average_precision_score(ys,ss))
        selection_match=bool(np.isclose(selection_ap,old['selection_AP_primary'],rtol=0,atol=1e-10))
        old_thresholds_ok=bool(np.allclose(oldframe.threshold,old['threshold'],rtol=0,atol=1e-12))
        frozen_pred=(os>=old['threshold']).astype(int);fresh_pred=(os>=threshold).astype(int)
        mismatches_frozen=int(np.count_nonzero(frozen_pred!=oldframe.prediction.to_numpy()))
        mismatches_fresh=int(np.count_nonzero(fresh_pred!=oldframe.prediction.to_numpy()))
        metric_checks=[]
        for window in [20,30,40]:
            y=held[f'late_life_le_{window}'].to_numpy(dtype=int)
            now=metrics(y,os,old['threshold']);saved=all_metrics[(rep,method,window)]
            delta={m:float(now[m]-saved[m]) for m in now}
            ok=all(abs(v)<=1e-10 for v in delta.values())
            metric_checks.append({'RUL_window':window,'pass':ok,'recomputed':now,'differences':delta})
        passed=score_check['pass'] and threshold_check['pass'] and selection_match and old_thresholds_ok and mismatches_frozen==0 and mismatches_fresh==0 and all(m['pass'] for m in metric_checks)
        rows.append({'representation':rep,'method':method,'selected_params':old['selected_params'],'status':'PASS' if passed else 'REVIEW_REQUIRED','score_comparison':score_check,'threshold_comparison':threshold_check,'selection_AP_recomputed':selection_ap,'selection_AP_recorded':old['selection_AP_primary'],'selection_AP_match':selection_match,'saved_prediction_thresholds_match':old_thresholds_ok,'prediction_mismatches_frozen_threshold':mismatches_frozen,'prediction_mismatches_recomputed_threshold':mismatches_fresh,'metrics':metric_checks})
    return {'stage':'02C_FD001_saved_model_reproduction','status':'PASS_SINGLE_FOLD_REPRODUCTION' if all(x['pass'] for x in arrays) and all(x['status']=='PASS' for x in rows) else 'REVIEW_REQUIRED','versions':versions,'seed':42,'fold':1,'source_fold_directory':str(folder),'AE_training_performed':False,'selected_classical_detector_refits':6,'hyperparameter_search_performed':False,'new_DBSCAN_or_OCSVM_results_produced':False,'test_files_read':False,'previous_outputs_modified':False,'source_files_checked':checked,'scaler_and_encoder_checks':arrays,'pipelines':rows,'limits':['This verifies one FD001 fold/seed, not all 780 representation jobs.','Remaining saved representations require corresponding checks before reuse.','No missing-detector experiment has been completed by this check.']}


if __name__=='__main__':
    config=json.loads(Path(sys.argv[1]).read_text());out=Path(config['output_directory'])
    try:
        result=execute(config)
    except Exception as error:
        result={'stage':'02C_FD001_saved_model_reproduction','status':'ERROR','error':str(error),'traceback':traceback.format_exc(),'AE_training_performed':False,'new_DBSCAN_or_OCSVM_results_produced':False}
        traceback.print_exc()
    (out/'model_reproduction_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print('SONUÇ:',result['status'],flush=True)
