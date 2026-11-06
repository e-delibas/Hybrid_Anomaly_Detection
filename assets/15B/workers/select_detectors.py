from pathlib import Path
import sys,json,time,hashlib,platform,importlib.metadata as im
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,f1_score,confusion_matrix
from detector_core import (require,dbscan_candidates,fit_candidate,score,fit_missing,score_missing,
                           CandidateInvalid,verify_stream_backend)


def threshold(y,s):
    p,r,t=precision_recall_curve(y,s)
    f=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(t),where=p[:-1]+r[:-1]>0)
    return float(t[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]])


def selection_metrics(y,s,t):
    pred=s>=t
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(selection_AP=float(average_precision_score(y,s)),selection_ROC_AUC=float(roc_auc_score(y,s)),
                selection_F1=float(f1_score(y,pred,zero_division=0)),threshold=t,
                selection_TP=int(tp),selection_FP=int(fp),selection_TN=int(tn),selection_FN=int(fn))


def execute(out,plan):
    started=time.perf_counter()
    training=json.loads((out/'training_summary.json').read_text())
    require(training['status']=='PASS_TRAINING','Training incomplete')
    require(hashlib.sha256((out/'trained_outputs.npz').read_bytes()).hexdigest()==training['trained_outputs_sha256'],'Training outputs changed')
    values=np.load(out/'trained_outputs.npz',allow_pickle=False)
    data=np.load(out/'prepared_data.npz',allow_pickle=False)
    for role in ['fit','selection']:
        require(np.array_equal(values['keys_'+role],data['keys_'+role]),'Sample keys differ: '+role)
    require(np.array_equal(values['y_selection'],data['y_selection']),'Selection labels differ')
    y=values['y_selection'];Zfit=values['Z_fit'];Zsel=values['Z_selection']
    require(Zfit.shape==(len(data['X_fit']),plan['training']['latent_dim']),'Latent shape differs')
    post=StandardScaler().fit(Zfit)
    fit=post.transform(Zfit);sel=post.transform(Zsel)
    joblib.dump(post,out/'sclm_latent_scaler.joblib')
    dbspec={'min_samples':[20,50,100],'eps_quantiles':[.5,.75,.9]}
    candidates={'DBSCAN':dbscan_candidates(fit,dbspec),
       'IF':[{'n_estimators':n} for n in [100,200,400]],
       'LOF':[{'n_neighbors':n} for n in [20,50,100]],
       'OC-SVM':[{'nu':v,'gamma':g} for v in [.01,.05,.1] for g in [.0001,.01,'scale']],
       'kNN':[{'n_neighbors':n} for n in [20,50,100]]}
    require(list(candidates)==['DBSCAN','IF','LOF','OC-SVM','kNN'],'Five-detector scope changed')
    trials=[];selected=[];selection_scores={}
    for method,grid in candidates.items():
        best=None
        for index,params in enumerate(grid):
            print('DETEKTOR',method,'ADAY',index+1,'/',len(grid),flush=True)
            begin=time.perf_counter();trial=dict(method=method,order=index,params=params)
            try:
                if method in ('DBSCAN','OC-SVM'):
                    model,diag=fit_missing(method,params,fit,{'max_dbscan_neighbor_entries':20_000_000})
                    scores=score_missing(method,model,sel)
                else:
                    model=fit_candidate(method,params,fit,plan['seed']);diag={}
                    scores=score(method,model,sel)
                ap=float(average_precision_score(y,scores))
                trial.update(status='VALID',selection_AP=ap,diagnostics=diag)
                if best is None or ap>best['ap']:
                    best=dict(ap=ap,params=params,model=model,scores=scores,order=index)
            except CandidateInvalid as exc:
                trial.update(status='INVALID',reason=str(exc))
            trial['elapsed_seconds']=time.perf_counter()-begin;trials.append(trial)
            (out/'detector_trials.json').write_text(json.dumps(trials,ensure_ascii=False,indent=2,allow_nan=False))
        require(best is not None,'No valid candidate for '+method)
        scores=best['scores'];cut=threshold(y,scores)
        row=dict(representation='SCLM_pretrained_d64',method=method,selected_params=best['params'],
                 selected_candidate_order=best['order'],**selection_metrics(y,scores,cut))
        model_path=out/('detector_'+method.replace('-','_')+'.joblib')
        joblib.dump(best['model'],model_path)
        restored=joblib.load(model_path)
        repeated=score_missing(method,restored,sel) if method in ('DBSCAN','OC-SVM') else score(method,restored,sel)
        require(np.array_equal(repeated,scores),'Saved detector scores differ: '+method)
        row['checkpoint_score_reproduced']=True
        selection_scores[method]=scores;selected.append(row)
    standalone=[]
    for variant in plan['training']['stage2_variants']:
        scores=values[f'APM_{variant}_selection']
        all_scores=values[f'APM_{variant}_all_epoch_scores']
        aps=np.array([average_precision_score(y,x) for x in all_scores])
        record=training['stage2'][variant]
        require(int(np.argmax(aps))+1==record['selected_epoch'],'Checkpoint selection does not match sklearn AP')
        require(abs(float(aps.max())-record['selection_AP'])<1e-12,'AP implementations disagree')
        require(np.array_equal(scores,all_scores[record['selected_epoch']-1]),'Wrong epoch scores')
        cut=threshold(y,scores)
        standalone.append(dict(method='SEAD_adapted_'+variant,primary=(variant=='raw_error'),
                               selected_epoch=record['selected_epoch'],**selection_metrics(y,scores,cut)))
        selection_scores['SEAD_'+variant]=scores
    np.savez_compressed(out/'selection_scores.npz',keys=values['keys_selection'],y=y,**selection_scores)
    all_valid=len(trials)==27 and all(t['status']=='VALID' for t in trials)
    result=dict(status='PASS_SELECTION' if all_valid else 'REVIEW_REQUIRED_CANDIDATES',
                detectors=['DBSCAN','IF','LOF','OC-SVM','kNN'],candidate_count=len(trials),all_candidates_valid=all_valid,
                selected_detectors=selected,standalone=standalone,fold=plan['fold'],seed=plan['seed'],
                late_life_reference='RUL <= 30',outer_evaluation_performed=False,
                warning='All metrics in this file are selection-set diagnostics, not held-out generalization estimates.',
                versions={'python':platform.python_version(),**{k:im.version(k) for k in ['numpy','pandas','scikit-learn','scipy','joblib']}},
                elapsed_seconds=time.perf_counter()-started)
    (out/'selection_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print('SECIM_TAMAM',result['status'],flush=True)


if __name__=='__main__':
    out=Path(sys.argv[1]);plan=json.loads((out/'plan.json').read_text())
    execute(out,plan)
