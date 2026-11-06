from pathlib import Path
import json,time,warnings,gc
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors,BallTree
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.exceptions import ConvergenceWarning
def require(condition,message):
    if not bool(condition):raise ValueError(message)

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

class CandidateInvalid(ValueError):
    pass

class ResourceLimit(RuntimeError):
    pass

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