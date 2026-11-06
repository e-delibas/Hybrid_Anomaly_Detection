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


def verify_and_prepare(config):
    seed=int(config["seed"]);fold_id=int(config["fold"])
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
    record=json.loads((folder/'result.json').read_text())
    rep=config['representation'];dim=config['latent_dim']
    require(record['config']==config['ae_config'] and record['plan_sha256']==config['source_plan_sha256'],'AE configuration/plan differs')
    record['primary_results']=[r for r in record['metrics'] if r['RUL_window']==30]
    record['sensitivity_results']=[r for r in record['metrics'] if r['RUL_window']!=30]
    require(record['seed']==seed and record['fold']==fold_id and record['status']=='PASS','Unexpected fold record')
    # The independently supplied 05 summary anchors this fold's primary results.
    expected_primary={(r['representation'],r['method']):r for r in config['approved_primary_results']}
    require(len(expected_primary)==4 and len(record['primary_results'])==4,'Expected four existing pipelines')
    for r in record['primary_results']:
        require(r==expected_primary[(r['representation'],r['method'])],'Fold record differs from supplied CV summary')
    predpath=folder/'outer_predictions.csv'
    require('outer_predictions.csv' in record['output_sha256'],'Prediction hash not recorded')
    require(digest(predpath)==record['output_sha256']['outer_predictions.csv'],'Prediction file hash mismatch')
    oldpred=pd.read_csv(predpath,float_precision='round_trip')
    oldpred['representation']=rep
    require(not oldpred.duplicated(['representation','method','sample_key']).any(),'Duplicate prediction keys')
    sensors=[f'Sensor{i}' for i in range(1,22)]
    raw=pd.read_csv(train_path,sep=r'\s+',header=None)
    require(raw.shape==(20631,26),'Unexpected FD001 shape')
    raw.columns=['UnitNumber','Time','OpSet1','OpSet2','OpSet3']+sensors
    reference=pd.read_csv(protocol/'train_evaluation_reference.csv')
    require(len(reference)==len(raw) and np.array_equal(reference.source_row,np.arange(len(raw))),'Metadata row mismatch')
    require(np.array_equal(raw[['UnitNumber','Time']],reference[['UnitNumber','Time']]),'Metadata alignment mismatch')
    splits=pd.read_csv(protocol/'engine_splits.csv')
    fold=splits.loc[splits.scheme.eq('cv') & splits.fold.eq(fold_id)]
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
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);tf.config.experimental.enable_op_determinism()
    print('Kayıtlı AE ve encoder yükleniyor; AE eğitimi yapılmıyor.',flush=True)
    ae=tf.keras.models.load_model(folder/'autoencoder.keras',compile=False)
    encoder=tf.keras.models.load_model(folder/'encoder.keras',compile=False)
    require(ae.input_shape[-1]==21 and ae.output_shape[-1]==21 and encoder.output_shape[-1]==dim,'Saved model dimensions differ')
    Z={r:encoder.predict(x,batch_size=256,verbose=0) for r,x in Xs.items()}
    for r,z in Z.items():require(z.shape==(len(Xs[r]),dim) and np.isfinite(z).all(),'Bad latent representation '+r)
    embedded_encoder=tf.keras.Model(ae.input,ae.get_layer('latent').output)
    arrays.append(compare_array('encoder_vs_AE_latent',Z['fit'][:256],embedded_encoder.predict(Xs['fit'][:256],batch_size=256,verbose=0)))
    latent_scaler=StandardScaler().fit(Z['fit'])
    arrays.extend([compare_array('latent_mean',latent_scaler.mean_,stored_scalers['latent']['mean']),compare_array('latent_scale',latent_scaler.scale_,stored_scalers['latent']['scale'])])
    representations={rep:{r:latent_scaler.transform(z) for r,z in Z.items()}}
    mse={r:np.mean(np.square(Xs[r]-ae.predict(Xs[r],batch_size=256,verbose=0)),axis=1) for r in ['selection','outer_valid']}
    held=reference.loc[masks['outer_valid']].reset_index(drop=True)
    ys=reference.loc[masks['selection'],'late_life_le_30'].to_numpy(dtype=int)
    require(set(zip(oldpred.representation,oldpred.method))==set(expected_primary),'Unexpected prediction pipeline set')
    all_metrics={(r['representation'],r['method'],r['RUL_window']):r for r in record['primary_results']+record['sensitivity_results']}
    require(len(all_metrics)==12,'Expected 12 stored metric rows')
    rows=[]
    for (rep,method),old in expected_primary.items():
        print('Yeniden üretim:',rep,method,flush=True)
        if method=='MSE':ss,os=mse['selection'],mse['outer_valid']
        else:
            model=fit_candidate(method,old['selected_params'],representations[rep]['fit'],seed)
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
    result = {'stage':'02C_FD001_saved_model_reproduction','status':'PASS_SINGLE_FOLD_REPRODUCTION' if all(x['pass'] for x in arrays) and all(x['status']=='PASS' for x in rows) else 'REVIEW_REQUIRED','versions':versions,'seed':seed,'fold':fold_id,'source_fold_directory':str(folder),'AE_training_performed':False,'selected_classical_detector_refits':3,'hyperparameter_search_performed':False,'new_DBSCAN_or_OCSVM_results_produced':False,'test_files_read':False,'previous_outputs_modified':False,'source_files_checked':checked,'scaler_and_encoder_checks':arrays,'pipelines':rows,'limits':['This verifies one FD001 fold/seed, not all 780 representation jobs.','Remaining saved representations require corresponding checks before reuse.','No missing-detector experiment has been completed by this check.']}



    return result, {'representations':representations, 'masks':masks, 'reference':reference, 'held':held, 'y_selection':ys, 'source_record':record, 'versions':versions}

# Explicit missing-method implementations; no generic fallback to kNN.
import time,warnings,gc
import joblib
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.neighbors import BallTree
from sklearn.exceptions import ConvergenceWarning


class CandidateInvalid(ValueError):
    pass


class ResourceLimit(RuntimeError):
    pass


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def dbscan_candidates(X,spec):
    F=np.ascontiguousarray(X,dtype=np.float64)
    largest=max(spec['min_samples'])
    require(largest<=len(F),'DBSCAN min_samples exceeds fit size')
    # Explicit query F includes the point itself, as DBSCAN min_samples does.
    nn=NearestNeighbors(n_neighbors=largest,algorithm='ball_tree',metric='euclidean',n_jobs=-1).fit(F)
    distances=nn.kneighbors(F,return_distance=True)[0]
    return [{'min_samples':int(m),'eps_quantile':float(q),
             'eps':float(np.quantile(distances[:,m-1],q,method='linear'))}
            for m in spec['min_samples'] for q in spec['eps_quantiles']]


def fit_missing(method,params,X,limits):
    F=np.ascontiguousarray(X,dtype=np.float64)
    require(np.isfinite(F).all(),'Nonfinite fitting features')
    if method=='DBSCAN':
        eps=float(params['eps']);m=int(params['min_samples'])
        if eps<=0:raise CandidateInvalid('Nonpositive eps from fit-only distance quantile')
        tree=BallTree(F,metric='euclidean')
        counts=tree.query_radius(F,r=eps,count_only=True)
        entries=int(counts.sum())
        if entries>limits['max_dbscan_neighbor_entries']:
            raise ResourceLimit(f'DBSCAN would allocate {entries} neighbor entries; limit={limits["max_dbscan_neighbor_entries"]}. No candidate will be silently dropped.')
        # Same Euclidean geometry and float64 data as the resource pre-check.
        db=DBSCAN(eps=eps,min_samples=m,metric='euclidean',algorithm='ball_tree',n_jobs=-1).fit(F)
        if len(db.core_sample_indices_)==0:raise CandidateInvalid('No fit-set core samples; core-distance score undefined')
        core_nn=NearestNeighbors(n_neighbors=1,algorithm='ball_tree',metric='euclidean',n_jobs=-1).fit(db.components_)
        model={'kind':'DBSCAN_core_distance_v1','dbscan':db,'core_search':core_nn,'eps':eps}
        diag={'core_samples':int(len(db.core_sample_indices_)),'fit_rows':len(F),'fit_noise_fraction':float(np.mean(db.labels_==-1)),
              'clusters':int(len(set(db.labels_)-{-1})),'neighbor_entries':entries,'eps':eps,'min_samples':m}
        return model,diag
    if method=='OC-SVM':
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always',ConvergenceWarning)
            model=OneClassSVM(kernel='rbf',nu=params['nu'],gamma=params['gamma'],tol=1e-3,
                              cache_size=512,shrinking=True,max_iter=100000).fit(F)
        if model.fit_status_!=0 or any(issubclass(w.category,ConvergenceWarning) for w in recorded):
            raise CandidateInvalid('OC-SVM did not converge within the frozen iteration limit')
        return model,{'support_vectors':int(len(model.support_)),'fit_rows':len(F),'fit_status':int(model.fit_status_),
                      'iterations':int(model.n_iter_),'effective_gamma':float(model._gamma)}
    raise ValueError('Unsupported missing method: '+str(method))


def score_missing(method,model,X):
    Q=np.ascontiguousarray(X,dtype=np.float64)
    if method=='DBSCAN':
        values=model['core_search'].kneighbors(Q,return_distance=True)[0][:,0]/model['eps']
    elif method=='OC-SVM':
        values=-model.score_samples(Q)
    else:raise ValueError('Unsupported missing scoring method: '+str(method))
    require(np.isfinite(values).all(),'Nonfinite new-method scores')
    return values


def evaluate_fixed_radius(y,scores):
    # Diagnostic only. Fixed cores and strict distance > eps; no clustering on queries.
    pred=(scores>1.0).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'F1':float(f1_score(y,pred,zero_division=0)),
            'Macro_F1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'TP':int(tp),'FP':int(fp),'TN':int(tn),'FN':int(fn)}


def execute_fold(config):
    seed=int(config["seed"]);fold_id=int(config["fold"])
    out=Path(config['output_directory']);protocol=config['five_method_addendum']
    require(canonical_hash(protocol)==config['addendum_sha256'],'Addendum hash mismatch')
    # This also verifies source hashes and the reused three-method baseline.
    baseline,ctx=verify_and_prepare(config)
    write_json(out/'baseline_reproduction.json',baseline)
    require(baseline['status']=='PASS_SINGLE_FOLD_REPRODUCTION','Baseline reproduction failed; no new models run')
    trials=[];chosen={};models={};feature_sets=ctx['representations'];ys=ctx['y_selection']
    for rep in [config['representation']]:
        F=feature_sets[rep]['fit'];S=feature_sets[rep]['selection']
        candidates={'DBSCAN':dbscan_candidates(F,protocol['DBSCAN']),
                    'OC-SVM':[{'nu':nu,'gamma':gamma} for nu in protocol['OC-SVM']['nu'] for gamma in protocol['OC-SVM']['gamma']]}
        for method in protocol['new_methods']:
            best=None
            for order,params in enumerate(candidates[method]):
                print('ADAY',rep,method,order+1,'/',len(candidates[method]),params,flush=True)
                started=time.perf_counter();entry={'representation':rep,'method':method,'candidate_order':order,'params':params}
                try:
                    model,diag=fit_missing(method,params,F,protocol['resources'])
                    sel=score_missing(method,model,S)
                    ap=float(average_precision_score(ys,sel))
                    entry.update(status='VALID',selection_AP=ap,diagnostics=diag)
                    if best is None or ap>best['selection_AP']:
                        best={'model':model,'params':params,'selection_AP':ap,'scores':sel,'candidate_order':order,'diagnostics':diag}
                    del model
                except CandidateInvalid as error:
                    entry.update(status='INVALID',reason=str(error))
                except ResourceLimit as error:
                    entry.update(status='RESOURCE_LIMIT',reason=str(error),seconds=time.perf_counter()-started)
                    trials.append(entry);write_json(out/'selection_trials.json',trials);raise
                entry['seconds']=time.perf_counter()-started;trials.append(entry)
                write_json(out/'selection_trials.json',trials)
                gc.collect()
            require(best is not None,f'No valid candidates for {rep}/{method}; comparison incomplete')
            best['threshold']=choose_threshold(ys,best['scores'])
            models[(rep,method)]=best
            chosen[rep+'/'+method]={k:v for k,v in best.items() if k not in ['model','scores']}
            joblib.dump(best['model'],out/f'{rep}_{method}_selected.joblib',compress=3)
    # Freeze all selections before computing any new-method outer performance.
    write_json(out/'selected_configurations.json',chosen)
    held=ctx['held'];reference=ctx['reference'];selected_rows=reference.loc[ctx['masks']['selection']].reset_index(drop=True)
    metadata=['sample_key','engine_key','UnitNumber','Time','RUL','late_life_le_20','late_life_le_30','late_life_le_40']
    new_rows=[];predictions=[];selections=[];fixed_radius=[]
    for (rep,method),best in models.items():
        scores=score_missing(method,best['model'],feature_sets[rep]['outer_valid'])
        variant='DBSCAN_core_distance_v1' if method=='DBSCAN' else 'OCSVM_rbf_negative_score_samples'
        for window in [20,30,40]:
            y=held[f'late_life_le_{window}'].to_numpy(dtype=int)
            new_rows.append({'representation':rep,'method':method,'method_display_name':'DBSCAN (core-distance)' if method=='DBSCAN' else method,'variant':variant,'RUL_window':window,'seed':seed,'fold':fold_id,
                             'selected_params':best['params'],'selection_AP_primary':best['selection_AP'],'threshold':best['threshold'],
                             'result_origin':'new_missing_method_cv_completion',**metrics(y,scores,best['threshold'])})
            if method=='DBSCAN':
                fixed_radius.append({'representation':rep,'method':'DBSCAN','RUL_window':window,
                                     'rule':'distance_to_fixed_fit_cores > eps; score > 1',
                                     'not_native_predict':True,**evaluate_fixed_radius(y,scores)})
        frame=held[metadata].copy();frame['representation']=rep;frame['method']=method;frame['variant']=variant
        frame['score']=scores;frame['threshold']=best['threshold'];frame['prediction']=(scores>=best['threshold']).astype(int)
        if method=='DBSCAN':frame['fixed_radius_prediction']=(scores>1.0).astype(int)
        predictions.append(frame)
        selection=selected_rows[metadata].copy();selection['representation']=rep;selection['method']=method
        selection['score']=best['scores'];selection['threshold']=best['threshold'];selections.append(selection)
    baseline_rows=[{**r,'method_display_name':r['method'],'result_origin':'stage06_reused_and_reproduced'} for r in ctx['source_record']['primary_results']+ctx['source_record']['sensitivity_results']]
    combined=baseline_rows+new_rows
    expected={(rep,m,w) for rep in [config['representation']] for m in ['DBSCAN','IF','LOF','OC-SVM','kNN'] for w in [20,30,40]}
    expected|={(config['representation'],'MSE',w) for w in [20,30,40]}
    keys=[(r['representation'],r['method'],r['RUL_window']) for r in combined]
    require(len(keys)==18 and len(set(keys))==18 and set(keys)==expected,'Five-method scope incomplete or duplicated')
    pd.DataFrame(combined).to_csv(out/'combined_pilot_metrics.csv',index=False)
    pd.DataFrame(fixed_radius).to_csv(out/'dbscan_fixed_radius_diagnostics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'new_outer_predictions.csv.gz',index=False,compression='gzip')
    pd.concat(selections,ignore_index=True).to_csv(out/'new_selection_scores.csv.gz',index=False,compression='gzip')
    required_candidates=18
    all_valid=len(trials)==required_candidates and all(r['status']=='VALID' for r in trials)
    result={'stage':'06B_FD001_ablation_pilot_fold','status':'PASS_FOLD' if all_valid else 'REVIEW_REQUIRED_CANDIDATES',
            'addendum_sha256':config['addendum_sha256'],'full_plan_sha256':config['full_plan_sha256'],'versions':ctx['versions'],'seed':seed,'fold':fold_id,'representation':config['representation'],
            'baseline_reproduction_status':baseline['status'],'detectors':['DBSCAN','IF','LOF','OC-SVM','kNN'],
            'DBSCAN_variant':'fit-set DBSCAN plus fixed-core distance score; not native DBSCAN predict',
            'DBSCAN_score':'min Euclidean distance to fit-set core samples / eps',
            'DBSCAN_evaluation_difference_from_original':'Original re-fit on test batch; this pilot fixes core samples learned on fitting engines and calibrates score threshold on selection engines.',
            'AE_training_performed':False,'new_candidate_attempts':len(trials),'all_candidate_fits_valid':all_valid,
            'new_methods':['DBSCAN','OC-SVM'],'new_DBSCAN_or_OCSVM_results_produced':True,'test_files_read':False,'previous_outputs_modified':False,
            'primary_results':[r for r in combined if r['RUL_window']==30],
            'sensitivity_results':[r for r in combined if r['RUL_window']!=30],
            'selection_trials':trials,'selected_configurations':chosen,'fixed_radius_diagnostics':fixed_radius,
            'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name not in ['fold_completion_summary.json','worker.log']},
            'limits':['One saved ablation configuration/fold/seed; this is a five-configuration pilot, not full ablation CV.',
                      'Candidates are a limited follow-up grid frozen before this pilot, not a claim of optimal tuning or equal search budget across all methods.',
                      'DBSCAN continuous scoring and selection threshold are an explicitly disclosed extension.',
                      'The fixed-radius diagnostic is not an additional sixth detector and does not re-cluster outer observations.']}
    print('\nBu fold için beş yöntem, ana RUL<=30 sonuçları:',flush=True)
    print(pd.DataFrame(result['primary_results'])[['representation','method_display_name','AP','F1','Macro_F1']].to_string(index=False),flush=True)
    return result


if __name__=='__main__':
    config=json.loads(Path(sys.argv[1]).read_text());out=Path(config['output_directory'])
    try:
        result=execute_fold(config)
    except Exception as error:
        result={'stage':'06B_FD001_ablation_pilot_fold','status':'RESOURCE_LIMIT' if isinstance(error,ResourceLimit) else 'ERROR',
                'error':str(error),'traceback':traceback.format_exc(),'AE_training_performed':False,
                'completed_five_method_fold':False,'partial_outputs_directory':str(out)}
        traceback.print_exc()
    write_json(out/'fold_completion_summary.json',result)
    print('SONUÇ:',result['status'],flush=True)
