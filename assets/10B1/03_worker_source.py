"""One recorded FD001 PCA fold: restore saved transform, reproduce old scores, add missing methods."""
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



def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def scaler_parameters(s):
    return {'mean':s.mean_.tolist(),'scale':s.scale_.tolist(),'variance':s.var_.tolist()}

def validate_metadata(m,ds,rows,units):
    cols=['dataset','split','source_row','UnitNumber','Time','engine_key','sample_key','RUL']
    require(set(cols).issubset(m.columns) and len(m)==rows,'Eğitim metadata şeması/satır sayısı farklı.')
    require(m[cols].notna().all().all(),'Metadata içinde eksik değer var.')
    require(m.dataset.eq(ds).all() and m.split.eq('train').all(),'Veri kümesi/bölme karışmış.')
    m=m.copy()
    for col in ['source_row','UnitNumber','Time','RUL']:
        a=pd.to_numeric(m[col],errors='raise').to_numpy(dtype=float)
        require(np.isfinite(a).all() and np.all(a==np.floor(a)),f'{col} sonlu tam sayı olmalı.')
        m[col]=a.astype('int64')
    require(np.array_equal(m.source_row.to_numpy(),np.arange(rows)),'source_row sırası değişmiş.')
    require(np.array_equal(np.sort(m.UnitNumber.unique()),np.arange(1,units+1)),'Motor kimlikleri farklı.')
    require(not m.duplicated(['UnitNumber','Time']).any(),'Tekrarlanan motor/çevrim var.')
    engines=ds+':train:'+m.UnitNumber.astype(str)
    require(m.engine_key.eq(engines).all(),'Motor anahtarı geçersiz.')
    require(m.sample_key.eq(engines+':'+m.Time.astype(str)).all() and m.sample_key.is_unique,'Örnek anahtarı geçersiz.')
    for engine,g in m.groupby('engine_key',sort=False):
        require(np.array_equal(g.Time.to_numpy(),np.arange(1,len(g)+1)),f'{engine}: çevrim sırası bozuk.')
        require(np.array_equal(g.RUL.to_numpy(),g.Time.max()-g.Time.to_numpy()),f'{engine}: RUL hesabı farklı.')
    return m

def validate_splits(s,ds,keys):
    keys=set(keys);n=len(keys);size=n//10
    require(s.dataset.eq(ds).all() and set(s.scheme)=={'cv','final'},'Bölme kimliği farklı.')
    require(not s.duplicated(['scheme','fold','engine_key']).any(),'Aynı motor birden fazla role atanmış.')
    cv=s.loc[s.scheme.eq('cv')]
    require(set(cv.fold)==set(range(1,11)),'Dış fold kapsamı farklı.')
    for _,g in cv.groupby('fold'):
        require(len(g)==n and set(g.engine_key)==keys,'Bir fold tüm motorları tam bir kez içermeli.')
        require(g.role.value_counts().to_dict()=={'fit':7*size,'early_stop':size,'selection':size,'outer_valid':size},'CV rol sayıları farklı.')
    outer=cv.loc[cv.role.eq('outer_valid'),'engine_key'].value_counts()
    require(set(outer.index)==keys and outer.eq(1).all(),'Her motor tam bir kez dış değerlendirmede olmalı.')
    final=s.loc[s.scheme.eq('final')]
    require(len(final)==n and set(final.engine_key)==keys and final.fold.eq(0).all(),'Nihai bölme kapsamı farklı.')
    require(final.role.value_counts().to_dict()=={'fit':8*size,'early_stop':size,'selection':size},'Nihai rol sayıları farklı.')

def load_inputs(data_dir,protocol_dir,approved):
    summary_path=protocol_dir/'additional_protocol_summary.json'
    require(summary_path.is_file(),f'09 özeti bulunamadı: {summary_path}')
    s=json.loads(summary_path.read_text(encoding='utf-8'))
    require(s==approved,'09 özeti onaylanan çıktıdan farklı.')
    for relative,h in approved['output_file_sha256'].items():
        path=protocol_dir/relative
        require(path.is_file() and digest(path)==h,f'09 çıktısı eksik/değişmiş: {path}')
    p=json.loads((protocol_dir/'experiment_protocol.json').read_text(encoding='utf-8'))
    require(hashlib.sha256(canonical(p).encode()).hexdigest()==approved['canonical_protocol_sha256'],
            'Protokol içerik hash değeri farklı.')
    require(p['source_training_files']==approved['source_training_files'],'Kaynak veri planı farklı.')
    splits=pd.read_csv(protocol_dir/'engine_splits.csv')
    raw={};refs={}
    for ds,info in approved['source_training_files'].items():
        path=data_dir/f'train_{ds}.txt'
        require(path.is_file() and digest(path)==info['train_raw_sha256'],f'Ham eğitim dosyası eksik/değişmiş: {path}')
        frame=pd.read_csv(path,sep=r'\s+',header=None)
        require(frame.shape==(info['rows'],26),'Ham eğitim boyutu farklı.')
        frame.columns=COLUMNS
        require(np.isfinite(frame.to_numpy(dtype=float)).all(),'Sonlu olmayan ham değer.')
        ref=pd.read_csv(protocol_dir/ds/'train_evaluation_reference.csv')
        ref=validate_metadata(ref,ds,info['rows'],info['units'])
        require(np.array_equal(frame[['UnitNumber','Time']].to_numpy(),ref[['UnitNumber','Time']].to_numpy()),
                'Ham sensör satırları ile referans motor/çevrim sırası uyuşmuyor.')
        for w in [20,30,40]:
            require(np.array_equal(ref[f'late_life_le_{w}'].to_numpy(),ref.RUL.le(w).astype(int).to_numpy()),
                    'RUL referansı değişmiş.')
        validate_splits(splits.loc[splits.dataset.eq(ds)],ds,ref.engine_key.unique())
        raw[ds]=frame;refs[ds]=ref
    return p,splits,raw,refs

def transform_preprocessor(pre,frame):
    X=frame[SENSORS].to_numpy(dtype=np.float32)
    if pre['mode']=='global':
        out=pre['sensor_scaler'].transform(X).astype(np.float32,copy=False)
        labels=None
    else:
        settings=pre['settings_scaler'].transform(frame[SETTINGS].to_numpy(dtype=np.float64))
        labels=pre['kmeans'].predict(settings)
        out=np.empty_like(X)
        for k,scaler in pre['sensor_scalers'].items():
            mask=labels==k
            if mask.any():out[mask]=scaler.transform(X[mask])
    require(out.shape==(len(frame),21) and np.isfinite(out).all(),'Geçersiz sensör dönüşümü.')
    return out,labels

def describe_preprocessor(pre):
    if pre['mode']=='global':return {'mode':'global','sensor_scaler':scaler_parameters(pre['sensor_scaler'])}
    return {'mode':'conditionwise','fit_cluster_counts':pre['fit_cluster_counts'],
            'settings_scaler':scaler_parameters(pre['settings_scaler']),
            'centers_scaled_settings':pre['kmeans'].cluster_centers_.tolist(),
            'centers_original_settings':pre['settings_scaler'].inverse_transform(pre['kmeans'].cluster_centers_).tolist(),
            'sensor_scalers':{str(k):scaler_parameters(s) for k,s in pre['sensor_scalers'].items()}}

def verify_and_prepare(config):
    global np,pd,tf,StandardScaler,NearestNeighbors
    global average_precision_score,roc_auc_score,precision_recall_curve,precision_score,recall_score,f1_score,confusion_matrix
    global SENSORS,SETTINGS,COLUMNS
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import sklearn,importlib.metadata
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,precision_score,recall_score,f1_score,confusion_matrix
    SENSORS=[f'Sensor{i}' for i in range(1,22)];SETTINGS=['OpSet1','OpSet2','OpSet3'];COLUMNS=['UnitNumber','Time']+SETTINGS+SENSORS
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'tensorflow':tf.__version__,'keras':tf.keras.__version__,'scipy':importlib.metadata.version('scipy'),'joblib':joblib.__version__}
    require(versions==config['expected_versions'],'Runtime versions differ')
    root=Path(config['revision_root']);folder=root/config['fold_relative_directory']
    for rel,h in config['known_hashes'].items():require(digest(root/rel)==h,'Original file changed: '+rel)
    protocol,splits,frames,refs=load_inputs(Path(config['data_directory']),root/'09_FD002_FD003_protocol',config['approved09_summary'])
    ident={k:config[k] for k in ['dataset','preprocessing','seed','fold','representation']};ds=ident['dataset'];rep=ident['representation'];mode=ident['preprocessing'];seed=ident['seed'];fold=ident['fold']
    record=json.loads((folder/'result.json').read_text())
    require(record['status']=='PASS' and record['identity']==ident and record['plan_sha256']==config['source_plan_sha256'],'Original job identity differs')
    for name,h in record['output_sha256'].items():
        require(Path(name).name==name and digest(folder/name)==h,'Original output changed: '+name)
    primary=[r for r in record['metrics'] if r['RUL_window']==30]
    require(primary==config['approved_primary_results'],'Primary results differ from supplied10 summary')
    ref=refs[ds];frame=frames[ds];foldmap=splits.loc[splits.dataset.eq(ds)&splits.scheme.eq('cv')&splits.fold.eq(fold)]
    roles=ref.engine_key.map(foldmap.set_index('engine_key').role);require(roles.notna().all(),'Unassigned engine')
    masks={r:roles.eq(r).to_numpy() for r in ['fit','early_stop','selection','outer_valid']}
    pre=joblib.load(folder/'preprocessor.joblib');require(pre['mode']==mode,'Preprocessing changed')
    require(describe_preprocessor(pre)==record['fold_info']['preprocessor'],'Preprocessor differs from stored JSON')
    arrays=[];fit=frame.loc[masks['fit']];Xfit=fit[SENSORS].to_numpy(dtype=np.float32)
    def check_scaler(name,stored,values):
        fresh=StandardScaler().fit(values)
        arrays.extend([compare_array(name+'_mean',fresh.mean_,stored.mean_,1e-10,1e-12),compare_array(name+'_scale',fresh.scale_,stored.scale_,1e-10,1e-12)])
    if mode=='global':check_scaler('sensor',pre['sensor_scaler'],Xfit)
    else:
        check_scaler('settings',pre['settings_scaler'],fit[SETTINGS].to_numpy(dtype=np.float64))
        labels=pre['kmeans'].predict(pre['settings_scaler'].transform(fit[SETTINGS].to_numpy(dtype=np.float64)))
        require(np.bincount(labels,minlength=6).tolist()==pre['fit_cluster_counts'],'Fit cluster counts differ')
        for k in range(6):check_scaler('sensor_condition_'+str(k),pre['sensor_scalers'][k],Xfit[labels==k])
    Xs={};cluster_counts={}
    for role,mask in masks.items():
        Xs[role],labels=transform_preprocessor(pre,frame.loc[mask])
        if labels is not None:cluster_counts[role]=np.bincount(labels,minlength=6).tolist()
    require(cluster_counts==record['fold_info']['cluster_rows_by_role'],'Role cluster counts differ')
    mse=None
    if rep=='Raw':Z={r:Xs[r] for r in ['fit','selection','outer_valid']}
    elif rep in ['PCA_d4','PCA_scaled_d4']:
        bundle=joblib.load(folder/'representation.joblib');pca=bundle['pca'];post=bundle['post_scaler']
        require(pca.components_.shape==(4,21) and not pca.whiten,'PCA shape/whiten differs')
        arrays.append(compare_array('PCA_fit_mean',pca.mean_,np.asarray(Xs['fit'],dtype=np.float64).mean(axis=0),1e-10,1e-12))
        Z={r:pca.transform(np.asarray(Xs[r],dtype=np.float64)) for r in ['fit','selection','outer_valid']}
        require((post is not None)==(rep=='PCA_scaled_d4'),'PCA post-scaling differs')
        if post is not None:
            check_scaler('PCA_post',post,Z['fit']);Z={r:post.transform(z) for r,z in Z.items()}
    else:
        require(rep in ['AE_deep_d4','AE_deep_d21'],'Unknown representation')
        dim=4 if rep.endswith('_d4') else 21
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);tf.config.experimental.enable_op_determinism()
        ae=tf.keras.models.load_model(folder/'autoencoder.keras',compile=False);encoder=tf.keras.models.load_model(folder/'encoder.keras',compile=False)
        require(ae.input_shape[-1]==ae.output_shape[-1]==21 and encoder.output_shape[-1]==dim,'AE shape differs')
        Z={r:encoder.predict(Xs[r],batch_size=256,verbose=0) for r in ['fit','selection','outer_valid']}
        embedded=tf.keras.Model(ae.input,ae.get_layer('latent').output)
        arrays.append(compare_array('encoder_vs_AE',Z['fit'][:256],embedded.predict(Xs['fit'][:256],batch_size=256,verbose=0)))
        latent=joblib.load(folder/'latent_scaler.joblib');check_scaler('latent',latent,Z['fit']);Z={r:latent.transform(z) for r,z in Z.items()}
        mse={r:np.mean(np.square(Xs[r]-ae.predict(Xs[r],batch_size=256,verbose=0)),axis=1) for r in ['selection','outer_valid']}
    require(all(np.isfinite(z).all() for z in Z.values()),'Nonfinite transformed data')
    require(all(a['pass'] for a in arrays),'Stored preprocessing/representation reproduction failed')
    held=ref.loc[masks['outer_valid']].reset_index(drop=True);selection=ref.loc[masks['selection']].reset_index(drop=True);ys=selection.late_life_le_30.to_numpy(dtype=int)
    saved_outer=pd.read_csv(folder/'outer_predictions.csv.gz',float_precision='round_trip');saved_selection=pd.read_csv(folder/'selection_scores.csv.gz',float_precision='round_trip')
    methods=['IF','LOF','kNN']+(['MSE'] if mse is not None else [])
    require({r['method'] for r in primary}==set(methods) and len(record['metrics'])==len(methods)*3,'Original scope incomplete')
    checks=[]
    for old in primary:
        method=old['method']
        if method=='MSE':ss,os=mse['selection'],mse['outer_valid']
        else:
            model=joblib.load(folder/f'detector_{method}.joblib')
            ss=score(method,model,Z['selection']);os=score(method,model,Z['outer_valid'])
        for name,values,stored,meta in [('selection',ss,saved_selection,selection),('outer',os,saved_outer,held)]:
            part=stored.loc[stored.method.eq(method)].set_index('sample_key')
            require(part.index.is_unique and set(part.index)==set(meta.sample_key),'Score sample keys differ')
            part=part.loc[meta.sample_key]
            for col in ['engine_key','UnitNumber','Time','RUL','late_life_le_20','late_life_le_30','late_life_le_40']:require(np.array_equal(part[col].to_numpy(),meta[col].to_numpy()),'Score metadata differs')
            c=compare_array(method+'/'+name,values,part.score.to_numpy());checks.append(c);require(c['pass'],'Stored score mismatch')
            require(np.allclose(part.threshold,old['threshold'],rtol=0,atol=1e-12),'Stored threshold differs')
            if name=='outer':require(np.array_equal((values>=old['threshold']).astype(int),part.prediction.to_numpy()),'Stored predictions differ')
        th=choose_threshold(ys,ss);require(np.isclose(th,old['threshold'],rtol=1e-5,atol=1e-7),'Recomputed threshold differs')
        require(np.array_equal(os>=th,os>=old['threshold']),'Recomputed threshold changes predictions')
        require(abs(float(average_precision_score(ys,ss))-old['selection_AP_primary'])<=1e-10,'Selection AP differs')
        for window in [20,30,40]:
            now=metrics(held[f'late_life_le_{window}'].to_numpy(dtype=int),os,old['threshold']);oldrow=next(r for r in record['metrics'] if r['method']==method and r['RUL_window']==window)
            require(all(abs(now[k]-oldrow[k])<=1e-10 for k in now),'Recomputed metric differs')
    result={'stage':'10B_saved_models_reproduction','status':'PASS_SINGLE_FOLD_REPRODUCTION','identity':ident,'versions':versions,'AE_training_performed':False,'PCA_refitted':False,'KMeans_refitted':False,'old_detector_refitted':False,'preprocessing_checks':arrays,'score_checks':checks,'test_files_read':False}
    source={'primary_results':primary,'sensitivity_results':[r for r in record['metrics'] if r['RUL_window']!=30]}
    return result,dict(representations={rep:Z},masks=masks,reference=ref,held=held,y_selection=ys,source_record=source,versions=versions)


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


def bounded_dbscan(F, eps, min_samples, entry_limit):
    """Exact unweighted Euclidean DBSCAN with bounded radius-neighbor batches."""
    from collections import deque
    cap = min(250_000, int(entry_limit))
    n = len(F)
    require(0 < n <= cap, 'A single neighborhood might exceed the entry budget')
    # Even a fully dense batch can return at most cap neighbor indices.
    width = min(256, cap // n)
    search = NearestNeighbors(radius=eps, algorithm='ball_tree', leaf_size=30,
                              metric='euclidean', n_jobs=1).fit(F)
    peak_entries = 0

    def neighborhoods(indices):
        nonlocal peak_entries
        found = search.radius_neighbors(F[indices], return_distance=False)
        entries = sum(len(neighbors) for neighbors in found)
        require(entries <= cap, 'Radius batch exceeded budget')
        peak_entries = max(peak_entries, entries)
        return found

    counts = np.empty(n, dtype=np.intp)
    for start in range(0, n, width):
        stop = min(start + width, n)
        counts[start:stop] = [len(row) for row in neighborhoods(slice(start, stop))]
    core = counts >= min_samples
    core_indices = np.flatnonzero(core)
    if not len(core_indices):
        raise CandidateInvalid('No fit-set core samples; core-distance score undefined')

    # Start components in native input order. Only core points expand a cluster.
    # An ambiguous border point belongs to the first component that reaches it.
    labels = np.full(n, -1, dtype=np.intp)
    cluster = 0
    for first in core_indices:
        if labels[first] != -1:
            continue
        labels[first] = cluster
        pending = deque([int(first)])
        while pending:
            batch = [pending.popleft() for _ in range(min(width, len(pending)))]
            for neighbors in neighborhoods(batch):
                fresh = neighbors[labels[neighbors] == -1]
                labels[fresh] = cluster
                pending.extend(fresh[core[fresh]].tolist())
        cluster += 1

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean',
                algorithm='ball_tree', n_jobs=-1)
    db.core_sample_indices_ = core_indices
    db.components_ = F[core_indices].copy()
    db.labels_ = labels
    db.n_features_in_ = F.shape[1]
    diagnostics = {'backend':'bounded_radius_batches_v1', 'batch_rows':width,
                   'batch_entry_cap':cap, 'peak_batch_entries':peak_entries,
                   'counted_neighbor_entries':int(counts.sum())}
    return db, diagnostics


def verify_stream_backend():
    # Run once in the actual worker's sklearn version before using the fallback.
    if getattr(verify_stream_backend, 'passed', False):
        return
    probe = np.array([[-1.75]] * 4 + [[-1.0], [0.0], [1.0]] + [[1.75]] * 4 + [[4.0]])
    for values in (probe, probe[::-1].copy()):
        ref = DBSCAN(eps=1., min_samples=4, algorithm='ball_tree', metric='euclidean', n_jobs=1).fit(values)
        got, _ = bounded_dbscan(values, 1., 4, len(values))
        require(np.array_equal(ref.core_sample_indices_, got.core_sample_indices_) and
                np.array_equal(ref.labels_, got.labels_) and np.array_equal(ref.components_, got.components_),
                'Native/streamed DBSCAN compatibility check failed')
    verify_stream_backend.passed = True


def fit_missing(method,params,X,limits):
    F=np.ascontiguousarray(X,dtype=np.float64)
    require(np.isfinite(F).all(),'Nonfinite fitting features')
    if method=='DBSCAN':
        eps=float(params['eps']);m=int(params['min_samples'])
        if eps<=0:raise CandidateInvalid('Nonpositive eps from fit-only distance quantile')
        tree=BallTree(F,metric='euclidean')
        counts=tree.query_radius(F,r=eps,count_only=True)
        entries=int(counts.sum())
        if entries > limits['max_dbscan_neighbor_entries']:
            verify_stream_backend()
            db, backend_info = bounded_dbscan(F, eps, m, limits['max_dbscan_neighbor_entries'])
        else:
            db = DBSCAN(eps=eps,min_samples=m,metric='euclidean',algorithm='ball_tree',n_jobs=-1).fit(F)
            backend_info = {'backend':'native_sklearn'}
        if len(db.core_sample_indices_)==0:raise CandidateInvalid('No fit-set core samples; core-distance score undefined')
        core_nn=NearestNeighbors(n_neighbors=1,algorithm='ball_tree',metric='euclidean',n_jobs=-1).fit(db.components_)
        model={'kind':'DBSCAN_core_distance_v1','dbscan':db,'core_search':core_nn,'eps':eps}
        diag={'core_samples':int(len(db.core_sample_indices_)),'fit_rows':len(F),'fit_noise_fraction':float(np.mean(db.labels_==-1)),
              'clusters':int(len(set(db.labels_)-{-1})),'neighbor_entries':entries,'eps':eps,'min_samples':m}
        diag.update(backend_info)
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
    baseline_rows=[{**r,'method_display_name':r['method'],'result_origin':'stage10_reused_and_reproduced'} for r in ctx['source_record']['primary_results']+ctx['source_record']['sensitivity_results']]
    for row in new_rows:row.update(dataset=config['dataset'],preprocessing=config['preprocessing'])
    combined=baseline_rows+new_rows
    methods=['DBSCAN','IF','LOF','OC-SVM','kNN']+(['MSE'] if config['representation'].startswith('AE_') else [])
    expected={(config['representation'],m,w) for m in methods for w in [20,30,40]}
    keys=[(r['representation'],r['method'],r['RUL_window']) for r in combined]
    require(len(keys)==len(expected) and len(set(keys))==len(expected) and set(keys)==expected,'Five-method scope incomplete or duplicated')
    pd.DataFrame(combined).to_csv(out/'combined_pilot_metrics.csv',index=False)
    pd.DataFrame(fixed_radius).to_csv(out/'dbscan_fixed_radius_diagnostics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'new_outer_predictions.csv.gz',index=False,compression='gzip')
    pd.concat(selections,ignore_index=True).to_csv(out/'new_selection_scores.csv.gz',index=False,compression='gzip')
    required_candidates=18
    all_valid=len(trials)==required_candidates and all(r['status']=='VALID' for r in trials)
    result={'stage':'10B_additional_five_method_pilot_job','status':'PASS_FOLD' if all_valid else 'REVIEW_REQUIRED_CANDIDATES',
            'addendum_sha256':config['addendum_sha256'],'full_plan_sha256':config['full_plan_sha256'],'versions':ctx['versions'],'seed':seed,'fold':fold_id,'representation':config['representation'],'dataset':config['dataset'],'preprocessing':config['preprocessing'],
            'baseline_reproduction_status':baseline['status'],'detectors':['DBSCAN','IF','LOF','OC-SVM','kNN'],
            'DBSCAN_variant':'fit-set DBSCAN plus fixed-core distance score; not native DBSCAN predict',
            'DBSCAN_score':'min Euclidean distance to fit-set core samples / eps',
            'DBSCAN_evaluation_difference_from_original':'Original re-fit on test batch; this pilot fixes core samples learned on fitting engines and calibrates score threshold on selection engines.',
            'PCA_refitted':False,'AE_training_performed':False,'new_candidate_attempts':len(trials),'all_candidate_fits_valid':all_valid,
            'new_methods':['DBSCAN','OC-SVM'],'new_DBSCAN_or_OCSVM_results_produced':True,'test_files_read':False,'previous_outputs_modified':False,
            'primary_results':[r for r in combined if r['RUL_window']==30],
            'sensitivity_results':[r for r in combined if r['RUL_window']!=30],
            'selection_trials':trials,'selected_configurations':chosen,'fixed_radius_diagnostics':fixed_radius,
            'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name not in ['fold_completion_summary.json','worker.log']},
            'limits':['One additional-dataset representation job; this is a 15-job pilot, not the full450-job comparison.',
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
        result={'stage':'10B_additional_five_method_pilot_job','status':'RESOURCE_LIMIT' if isinstance(error,ResourceLimit) else 'ERROR',
                'error':str(error),'traceback':traceback.format_exc(),'PCA_refitted':False,'AE_training_performed':False,
                'completed_five_method_fold':False,'partial_outputs_directory':str(out)}
        traceback.print_exc()
    write_json(out/'fold_completion_summary.json',result)
    print('SONUÇ:',result['status'],flush=True)
