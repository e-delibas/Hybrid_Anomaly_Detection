from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,confusion_matrix
from detector_core import score,score_missing
from select_detectors import threshold,selection_metrics


def metric(y,s,t):
    pred=s>=t
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(AP=float(average_precision_score(y,s)),ROC_AUC=float(roc_auc_score(y,s)),
                precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
                F1=float(f1_score(y,pred,zero_division=0)),Macro_F1=float(f1_score(y,pred,average='macro',zero_division=0)),
                TP=int(tp),FP=int(fp),TN=int(tn),FN=int(fn))


def evaluate(source,out,context):
    selection=json.loads((source/'selection_summary.json').read_text())
    inference=json.loads((out/'outer_predict_summary.json').read_text())
    if selection['status']!='PASS_SELECTION' or not selection['all_candidates_valid']:raise ValueError('Unapproved candidate selection')
    if hashlib.sha256((out/'outer_neural_outputs.npz').read_bytes()).hexdigest()!=inference['output_sha256']:raise ValueError('Neural outputs changed')
    neural=np.load(out/'outer_neural_outputs.npz',allow_pickle=False)
    recorded=np.load(source/'trained_outputs.npz',allow_pickle=False)
    saved=np.load(source/'selection_scores.npz',allow_pickle=False)
    ref=pd.read_csv(out/'outer_reference.csv')
    if not np.array_equal(neural['keys'],ref.sample_key.to_numpy(dtype=str)):raise ValueError('Outer identity mismatch')
    if not np.array_equal(saved['keys'],recorded['keys_selection']) or not np.array_equal(saved['y'],recorded['y_selection']):raise ValueError('Selection identity mismatch')
    scaler=joblib.load(source/'sclm_latent_scaler.joblib')
    Zsel=scaler.transform(recorded['Z_selection']);Zout=scaler.transform(neural['Z_outer'])
    models={r['method']:r for r in selection['selected_detectors']}
    expected=['DBSCAN','IF','LOF','OC-SVM','kNN']
    if set(models)!=set(expected):raise ValueError('Five-detector scope mismatch')
    all_scores={};all_thresholds={};parameters={}
    for name in expected:
        model=joblib.load(source/('detector_'+name.replace('-','_')+'.joblib'))
        scorer=score_missing if name in ('DBSCAN','OC-SVM') else score
        ss=scorer(name,model,Zsel);frozen=models[name]
        if not np.array_equal(ss,saved[name]):raise ValueError('Saved selection scores differ '+name)
        cut=threshold(saved['y'],ss)
        if cut!=frozen['threshold']:raise ValueError('Frozen threshold differs '+name)
        check=selection_metrics(saved['y'],ss,cut)
        for key,value in check.items():
            if abs(value-frozen[key])>1e-12:raise ValueError('Selection metric mismatch '+name+'/'+key)
        name_full='SCLM64/'+name
        all_scores[name_full]=scorer(name,model,Zout);all_thresholds[name_full]=cut;parameters[name_full]=frozen['selected_params']
    for frozen in selection['standalone']:
        variant=frozen['method'].removeprefix('SEAD_adapted_')
        key='SEAD_'+variant;ss=recorded['APM_'+variant+'_selection']
        if not np.array_equal(ss,saved[key]):raise ValueError('APM selection scores differ')
        cut=threshold(saved['y'],ss)
        if cut!=frozen['threshold']:raise ValueError('APM threshold differs')
        check=selection_metrics(saved['y'],ss,cut)
        for field,value in check.items():
            if abs(value-frozen[field])>1e-12:raise ValueError('APM selection metric mismatch '+field)
        name=frozen['method'];all_scores[name]=neural[key];all_thresholds[name]=cut
        parameters[name]={'selected_epoch':frozen['selected_epoch'],'primary':frozen['primary']}
    if len(all_scores)!=7:raise ValueError('Seven evaluation groups required')
    rows=[];frames=[]
    for name,scores in all_scores.items():
        if scores.shape!=(len(ref),) or not np.isfinite(scores).all():raise ValueError('Invalid outer scores')
        cut=all_thresholds[name]
        for window in [30,20,40]:
            y=ref.RUL.le(window).to_numpy(dtype=int)
            rows.append(dict(seed=context['seed'],fold=context['fold'],method=name,RUL_window=window,
                             threshold=cut,selected_parameters=parameters[name],n_outer=len(ref),**metric(y,scores,cut)))
        frame=ref.copy();frame['method']=name;frame['score']=scores;frame['threshold']=cut;frame['prediction']=(scores>=cut).astype(int)
        frames.append(frame)
    pd.concat(frames,ignore_index=True).to_csv(out/'outer_predictions.csv.gz',index=False,compression='gzip')
    report=dict(status='PASS_OUTER_FOLD',seed=context['seed'],fold=context['fold'],outer_rows=len(ref),
                outer_engines=sorted(ref.engine_key.unique()),selection_reproduced=True,thresholds_reselected_on_outer=False,
                groups=7,metric_rows=rows,original_detectors=expected,
                predictions_sha256=hashlib.sha256((out/'outer_predictions.csv.gz').read_bytes()).hexdigest())
    (out/'outer_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print('DIS_DEGERLENDIRME_TAMAM',context['seed'],context['fold'],flush=True)


if __name__=='__main__':
    out=Path(sys.argv[1]);context=json.loads((out/'evaluation_context.json').read_text())
    evaluate(Path(context['source']),out,context)
