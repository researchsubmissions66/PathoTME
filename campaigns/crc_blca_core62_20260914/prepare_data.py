"""Validate transferred features and freeze patient-separated CRC/BLCA inputs."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import h5py

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(a):
 sys.path[:0]=[str(a.pgvl),str(a.project)]
 from pathotme.hive_geometry import hierarchy_indices,resolve_hierarchy_geometry,array_sha
 from pathotme.controlled_vila import donor_maps
 from pathotme.locked_tcga import check_phases
 from pathotme.shared_panel import transform_rows,common_spec
 from pathotme.focus_contract import validate_donors
 source=a.project/'text_prompts/crc_blca_gpt6_astra_v1/source.json';authored=json.loads(source.read_text());task=authored['cohorts'][a.cohort]
 previous=json.loads((a.pgvl/'benchmarks/pathotme_cohort_candidates_20260912.json').read_text())
 prior=next(g for g in previous['groups'] if ('CRC' if a.cohort=='crc' else 'BLCA') in g['cohort'])
 clinical=a.data/'external/crc_blca_labels_v1'/Path(prior['label_source_path']).name
 if sha(clinical)!=prior['label_source_sha256']:raise ValueError('Clinical source changed')
 with clinical.open() as f:patients=list(csv.DictReader((line for line in f if not line.startswith('#')),delimiter='\t'))
 if len({r['PATIENT_ID'] for r in patients})!=len(patients):raise ValueError('Duplicate clinical patient')
 labels={r['PATIENT_ID']:r[task['label_field']] for r in patients}
 table=a.data/f'panels/shared_core62_20260914_v1/{a.cohort}/core62.csv';meta=json.loads(table.with_suffix('.metadata.json').read_text());identities=meta['original_slide_identities']
 if sha(table)!=meta['selected_table_sha256']:raise ValueError('TME table changed')
 files={}
 for encoder in ['plip','clip-rn50']:
  for scale in [5,10,20]:
   directory=a.data/'features'/task['cohort']/encoder/f'{scale}x'
   with os.scandir(directory) as it:found={p.name[:-3].lower():str(directory/p.name) for p in it if p.name.endswith('.h5')}
   if set(identities.values())-set(found):raise ValueError(f'Transfer incomplete: {encoder}/{scale}x')
   files[encoder,scale]=found
 if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
 a.output.mkdir(parents=True,exist_ok=True)
 slides={};manifest=[];exclusions=[];start=time.monotonic()
 for index,(slide,original) in enumerate(sorted(identities.items())):
  patient=slide[:12];source_label=labels.get(patient,'')
  if source_label not in task['source_label_mapping']:
   exclusions.append(dict(slide_id=slide,case_id=patient,source_label=source_label,reason='missing_or_excluded_clinical_histology'));continue
  label_id=task['source_label_mapping'][source_label];label=task['classes'][label_id]
  row=dict(slide_id=slide,case_id=patient,label=label,label_id=label_id,OncoTreeCode=label,Diagnosis=label,cohort=a.cohort,original_slide_identity=original)
  coords={};attrs={};record=dict(files={},hierarchy={})
  for encoder in ['plip','clip-rn50']:
   width=768 if encoder=='plip' else 1024
   for scale in [5,10,20]:
    path=Path(files[encoder,scale][original]);before=path.stat()
    with h5py.File(path,'r') as h:
     f,c=h['features'],h['coords']
     expected_encoder=encoder.replace('-','_')
     if f.ndim!=2 or f.shape[1]!=width or not len(f) or c.shape!=(len(f),2):raise ValueError(f'Invalid feature shape: {path}')
     if f.dtype not in (np.dtype('float16'),np.dtype('float32')) or c.dtype.kind not in 'iu':raise ValueError(f'Invalid feature/coordinate dtype: {path}')
     if f.attrs.get('encoder')!=expected_encoder:raise ValueError(f'Encoder header mismatch: {path}')
     if str(f.attrs.get('name','')).lower()!=original:raise ValueError(f'Slide identity mismatch: {path}')
     at={k:float(c.attrs[k]) for k in ['patch_size','target_magnification','patch_size_level0','level0_magnification','level0_width','level0_height','overlap']}
     if at['patch_size']!=224 or at['target_magnification']!=scale or at['overlap']!=0:raise ValueError(f'Patch geometry mismatch: {path}')
     if at['patch_size_level0']!=224*at['level0_magnification']/scale:raise ValueError(f'Inconsistent level-0 span: {path}')
     coordinates=np.asarray(c[:],dtype=np.int64)
     if len(np.unique(coordinates,axis=0))!=len(coordinates) or (coordinates<0).any():raise ValueError(f'Invalid coordinates: {path}')
     if (coordinates[:,0]>=at['level0_width']).any() or (coordinates[:,1]>=at['level0_height']).any():raise ValueError('Coordinates outside slide')
     digest=hashlib.sha256()
     for begin in range(0,len(f),4096):
      values=np.asarray(f[begin:begin+4096])
      if not np.isfinite(values).all():raise ValueError(f'Nonfinite feature values: {path}')
      digest.update(values.tobytes())
     coords[encoder,scale]=coordinates;attrs[encoder,scale]=at
     record['files'][str(path)]=dict(size=before.st_size,mtime_ns=before.st_mtime_ns,shape=list(f.shape),dtype=str(f.dtype),encoder=expected_encoder,geometry=at,coords_sha256=array_sha(coordinates),features_array_sha256=digest.hexdigest(),extraction_weights='historical_checkpoint_digest_not_provided; supplied_encoder_and_header_binding')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError(f'File changed during audit: {path}')
    row[f'feature__{expected_encoder}_{scale}x']=str(path)
  for scale in [5,10,20]:
   if not np.array_equal(coords['plip',scale],coords['clip-rn50',scale]) or attrs['plip',scale]!=attrs['clip-rn50',scale]:raise ValueError(f'Cross-encoder coordinate mismatch: {slide}/{scale}x')
  for encoder in ['plip','clip-rn50']:
   geometry=resolve_hierarchy_geometry(attrs[encoder,5],attrs[encoder,20]);parents,children=hierarchy_indices(coords[encoder,5],coords[encoder,20],geometry['low_patch_span_level0'])
   record['hierarchy'][encoder]=dict(geometry=geometry,retained_parents=len(parents),parent_indices_sha256=array_sha(parents),child_indices_sha256=array_sha(children))
  row['level0_mag']=attrs['plip',5]['level0_magnification'];slides[slide]=record;manifest.append(row)
  if len(manifest)%25==0:
   dump(a.output/'audit_progress.json',{'slides_checked':len(manifest),'seconds':time.monotonic()-start})
   print(a.cohort,'feature-checked',len(manifest),'slides',flush=True)
 import pandas as pd
 from scripts.tcga_benchmark import build_splits
 manifest_path=a.output/f'data/{a.cohort}/manifest.csv';manifest_path.parent.mkdir(parents=True)
 pd.DataFrame(manifest).to_csv(manifest_path,index=False)
 protocol={'shots':[16],'folds':5,'seed':1,'cohorts':{a.cohort:{'labels':task['classes'],'split_group_column':'case_id','split_group_label_policy':'strict'}}}
 build_splits(protocol,a.output)
 with table.open() as f:tme=list(csv.DictReader(f))
 transformed=dict(zip([r['slide_id'] for r in tme],transform_rows(tme)))
 all_test=[];fold_counts={}
 for fold in range(5):
  split=a.output/f'splits/{a.cohort}/16shot/fold{fold}';phases={phase:pd.read_csv(split/f'{phase}.csv').to_dict('records') for phase in ['train','val','test']}
  check_phases(phases,transformed);maps=donor_maps(phases,1,fold);validate_donors(phases,maps);dump(split/'donors.json',maps)
  values=np.asarray([transformed[r['slide_id']] for r in phases['train']],dtype=float)
  if np.isnan(values).all(0).any():raise ValueError('Training fold has wholly missing fixed-panel measurements')
  all_test += [r['slide_id'] for r in phases['test']]
  fold_counts[str(fold)]={phase:dict(slides=len(rows),patients=len({r['case_id'] for r in rows})) for phase,rows in phases.items()}
 if sorted(all_test)!=sorted(slides):raise ValueError('Outer test folds must partition the common cohort')
 dump(a.output/'feature_inventory.json',{'cohort':a.cohort,'slides':slides,'full_feature_payloads_finite':True,'cross_encoder_coordinates_equal':True,'hierarchy_validated':True})
 dump(a.output/'label_exclusions.json',exclusions)
 result={'status':'passed','cohort':a.cohort,'slides':len(manifest),'patients':len({r['case_id'] for r in manifest}),'class_patient_counts':{label:len({r['case_id'] for r in manifest if r['label']==label}) for label in task['classes']},'folds':fold_counts,'tme_csv':str(table),'feature_schema_sha256':common_spec()['feature_schema_sha256'],'clinical_source_sha256':sha(clinical),'source_sha256':sha(__file__),'input_hashes':{str(clinical):sha(clinical),str(source):sha(source),str(table):sha(table)},'expert_slide_histology_review_attested':False,'label_policy':'exact original slide UUID to TME and exact patient ID to frozen clinical histology; excluded missing/other diagnoses','elapsed_seconds':time.monotonic()-start}
 dump(a.output/'readiness.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cohort',choices=['crc','blca'],required=True)
 for key in ['data','output','pgvl','project']:p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
