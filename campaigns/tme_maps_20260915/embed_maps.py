"""Fit prespecified unsupervised UMAP and PCA; inputs contain covariates only."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import umap

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def embed(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());reports={};start=time.monotonic()
    for cohort in ['nsclc','brca','crc','blca']:
        folder=root/cohort;path=folder/'embedding_input.npz'
        if sha(path)!=m['prepared_artifact_sha256'][str(path)]:raise ValueError('Covariates changed')
        with np.load(path,allow_pickle=False) as z:
            if z.files!=['x']:raise ValueError('Embedding process accepts covariates only')
            x=z['x']
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Invalid embedding input')
        destination=folder/'embedding.npz'
        if destination.exists():raise FileExistsError('Refusing to search/overwrite an existing projection')
        begin=time.monotonic();u=umap.UMAP(**m['policy']['umap']).fit_transform(x)
        pca=PCA(**m['policy']['pca']);p=pca.fit_transform(x)
        if u.shape!=(len(x),2) or not np.isfinite(u).all():raise ValueError('Invalid UMAP output')
        np.savez_compressed(destination,umap=u,pca=p)
        reports[cohort]={'patients':len(x),'features':x.shape[1],'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
                        'umap_trustworthiness_k15':float(trustworthiness(x,u,n_neighbors=15)),
                        'pca_trustworthiness_k15':float(trustworthiness(x,p,n_neighbors=15)),
                        'embedding_sha256':sha(destination),'wall_seconds':time.monotonic()-begin}
        print(cohort,json.dumps(reports[cohort]),flush=True)
    r={'manifest_identity':m['identity'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'versions':{k:importlib.metadata.version(k) for k in ['umap-learn','numba','pynndescent','numpy','scipy','scikit-learn']},
       'wall_seconds':time.monotonic()-start,'cohorts':reports,
       'diagnostic_note':'Neighborhood preservation is descriptive; no hyperparameters selected by this diagnostic or outcomes.'}
    (root/'embedding_report.json').write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);embed(p.parse_args().output)
