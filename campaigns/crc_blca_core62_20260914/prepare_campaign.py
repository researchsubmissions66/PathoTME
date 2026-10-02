"""Freeze seven-method baseline/TME configurations for one completed cohort."""
import argparse,hashlib,json,sys
from copy import deepcopy
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def main(a):
 sys.path[:0]=[str(a.project),str(a.pgvl)]
 from common.configuration import load_yaml_config
 from common.prompts.vila_mil import load_vila_prompt_bank
 from common.prompts.focus import load_focus_prompt_bank
 from pathotme.yaml_runtime_config import write_runtime_config
 from pathotme.shared_panel import common_spec
 from pathotme.locked_tcga import identity
 ready=json.loads((a.prepared/'readiness.json').read_text())
 if ready['status']!='passed' or ready['cohort']!=a.cohort:raise ValueError('Matching prepared cohort required')
 bankroot=a.project/'text_prompts/crc_blca_gpt6_astra_v1';banks=json.loads((bankroot/'MANIFEST.json').read_text());source=json.loads((bankroot/'source.json').read_text());task=source['cohorts'][a.cohort]
 if a.output.exists():raise FileExistsError(a.output)
 sources=['tcga_locked_16shot_20260909_v1','tcga_cross_encoders_16shot_20260909_v1','focus_tcga_16shot_20260911_v1','muse_tcga_16shot_20260912_v1','hive_tcga_16shot_20260912_v1','dyko_tcga_16shot_20260912_v1','mscpt_tcga_16shot_20260914_v1']
 templates={}; encoder_bindings={}
 for name in sources:
  path=a.results/name/'launch.json';launch=json.loads(path.read_text())
  for key,digest in launch.get('file_sha256',{}).items():
   if key=='/path/to/.cache/clip/RN50.pt' or '/model_cache/plip/' in key:encoder_bindings[key]=digest
  for plan in launch['plans']:
   if plan['cohort']!='nsclc' or plan['fold']!=0:continue
   cfgpath=Path(plan['config']);cfg=json.loads(cfgpath.read_text()) if cfgpath.suffix=='.json' else load_yaml_config(cfgpath)
   templates[cfg['method'],cfg['backbone']]=(cfg,path,cfgpath,launch['protocol'])
 methods=['vila_mil','mgpath','focus','muse','hive_mil','dyko','mscpt'];encoders=['plip','clip-rn50'];plans=[];contracts={};bindings={}
 for method in methods:
  for encoder in encoders:
   cfg,parent_path,base_path,protocol=templates[method,encoder];cfg=deepcopy(cfg);key=f'{a.cohort}_{method}_{encoder}'
   labels=task['classes'];names=[source['classes'][c]['name'] for c in labels]
   cfg.update(task=a.cohort,benchmark='pathotme_crc_blca_shared62',experiment=key,label_dict={c:i for i,c in enumerate(labels)},classnames=names,n_classes=2,shots=16,k=5,k_start=0,k_end=1,seed=1,_fold_index=0,num_workers=0,max_batch_failure_rate=0.0,dataset_csv=str(a.prepared/f'data/{a.cohort}/manifest.csv'),split_dir=str(a.prepared/f'splits/{a.cohort}/16shot'),results_dir=str(a.output/f'runs/{method}/{encoder}/fold0/native'),prompt_provenance='generated',prompt_generator='GPT 6 ASTRA',pathotme_cohort_extension='crc_blca_shared62_v1',pathotme_arm='native',fidelity_note='PathoTME CRC/BLCA task extension with the existing method/encoder recipe, authored GPT 6 ASTRA banks and fixed common62 TME controls. Not an upstream cohort reproduction.')
   for name in ['hive_token_ids_sha256','mscpt_token_ids_sha256','selection_prompt_sha256','selection_prompt_source']:
    cfg.pop(name,None)
   for name in ['data_folder_l','data_folder_s','feat_data_dir','selected_5x_dir']:
    if name in cfg:cfg[name]=''
   record=[r for r in banks['records'] if r['cohort']==a.cohort and r['method']==method]
   assets={str(bankroot/r['file']):r['sha256'] for r in record}
   def asset(role):return bankroot/next(r['file'] for r in record if r['role']==role)
   if method in ['vila_mil','focus','mgpath']:
    path=asset('ordered_class_scale_descriptions');cfg['text_prompt_path']=str(path)
    if method=='mgpath':cfg.update(text_prompt_file_sha256=sha(path),prompt_class_bindings=labels*2,prompt_source='mgpath_generated_native_four_views')
    else:
     prefix='vila' if method=='vila_mil' else 'focus';loader=load_vila_prompt_bank if method=='vila_mil' else load_focus_prompt_bank
     bank=loader(path,class_names=labels,file_class_names=labels,expected_provenance='generated')
     cfg.update({prefix+'_prompt_file_classnames':labels,prefix+'_prompt_file_sha256':sha(path),prefix+'_prompt_bank_sha256':bank.ordered_prompt_bank_sha256,'prompt_source':method+'_generated_native_two_scale_csv'})
   elif method=='muse':
    cfg['prompt_csvs']={name:str(bankroot/a.cohort/'muse'/f'generated_new_{i}.csv') for i,name in enumerate(names)}
    cfg['prompt_source']='muse_generated_separate_retrieval_and_class_semantics'
   elif method=='hive_mil':
    path=asset('four_coarse_twelve_fine');cfg.update(text_prompt_path=str(path),text_prompt_file_sha256=sha(path),prompt_source='hive_mil_generated_hierarchical_bank')
   elif method=='mscpt':
    path=asset('multiscale_description_graph');cfg.update(description_prompt_path=str(path),description_prompt_sha256=sha(path),selection_prompt_path=None,dataset_name='TCGA_'+a.cohort.upper(),prompt_source='mscpt_generated_multiscale_description_json')
   elif method=='dyko':
    encoded=a.concepts/a.cohort/'encoding.json';proof=json.loads(encoded.read_text())
    if proof['status']!='completed' or proof['source_bank_sha256']!=sha(asset('unlabeled_visual_knowledge')):raise ValueError('Frozen DyKo concepts not encoded')
    path=asset('class_descriptions');tensor=Path(proof['tensor_path']);assets[str(tensor)]=proof['tensor_sha256'];assets[str(encoded)]=sha(encoded)
    knowledge={'class_prompt_path':str(path),'class_prompt_sha256':sha(path),'tensor_path':str(tensor),'tensor_sha256':sha(tensor),'source_bank_path':str(asset('unlabeled_visual_knowledge')),'source_bank_sha256':sha(asset('unlabeled_visual_knowledge')),'encoding_path':str(encoded),'encoding_sha256':sha(encoded)}
    cfg.update(text_prompt_path=str(path),text_prompt_file_sha256=sha(path),prompt_file_classnames=names,prompt_class_bindings=labels,concept_feature_path=str(tensor),concept_feature_sha256=sha(tensor),concept_count=64,pathotme_knowledge_bank=knowledge,concept_source='pathotme_crc_blca_authored64_pinned_titan',prompt_source='dyko_generated_class_descriptions')
   for path,digest in assets.items():
    if sha(path)!=digest:raise ValueError('New asset hash mismatch')
   adapter=deepcopy(protocol['adapter']);selection=deepcopy(protocol['selection'][method]);resources=deepcopy(protocol['resources'])
   if method=='muse':resources['fold_time']['muse']='06:00:00'
   cfg['lr']=float(cfg['lr']);cfg['weight_decay']=float(cfg['weight_decay'])
   contract={'native_config':deepcopy(cfg),'adapter':adapter,'selection':selection,'asset_sha256':assets,'tme_csv':ready['tme_csv'],'feature_schema_sha256':common_spec()['feature_schema_sha256'],'parent_recipe_path':str(base_path),'parent_recipe_sha256':sha(base_path),'parent_launch_path':str(parent_path),'parent_launch_sha256':sha(parent_path),'changes':{k:[templates[method,encoder][0].get(k),v] for k,v in cfg.items() if templates[method,encoder][0].get(k)!=v}}
   path=a.output/f'contracts/{method}_{encoder}.json';dump(path,contract);contracts[method+'/'+encoder]=str(path)
   cfg.update(cohort_contract_path=str(path),cohort_contract_sha256=sha(path))
   for fold in range(5):
    out=a.output/f'runs/{method}/{encoder}/fold{fold}';foldcfg={**cfg,'_fold_index':fold,'k_start':fold,'k_end':fold+1,'results_dir':str(out/'native')}
    path=a.output/f'configs/{method}_{encoder}_fold{fold}.yaml';write_runtime_config(path,foldcfg)
    plans.append(dict(cohort=a.cohort,method=method,encoder=encoder,fold=fold,config=str(path),native_dir=str(out/'native'),output=str(out),smoke_checkpoint_dir=str(a.output/f'smokes/{method}/{encoder}/native_fixture'),tme_csv=ready['tme_csv'],donor_maps=str(a.prepared/f'splits/{a.cohort}/16shot/fold{fold}/donors.json'),feature_inventory=str(a.prepared/'feature_inventory.json'),adapter=adapter,selection=selection,resources=resources,native_reused=False))
   bindings.update(assets);bindings[str(base_path)]=sha(base_path);bindings[str(parent_path)]=sha(parent_path)
 for path,digest in encoder_bindings.items():
  if sha(path)!=digest:raise ValueError('Frozen encoder weights changed')
  bindings[path]=digest
 for path in Path('/path/to/shared/model_cache/plip').iterdir():
  if path.is_file() and path.suffix in ['.json','.txt']:bindings[str(path)]=sha(path)
 for directory in [a.project/'pathotme',a.pgvl/'methods',a.pgvl/'common',a.pgvl/'clip',Path(__file__).parent]:
  for path in directory.rglob('*.py'):bindings[str(path)]=sha(path)
 for path in [a.pgvl/'train.py',a.project/'scripts/run_vila_guided.py',bankroot/'MANIFEST.json',bankroot/'source.json',Path(ready['tme_csv'])]:bindings[str(path)]=sha(path)
 for directory in [a.output,a.prepared]:
  for path in directory.rglob('*'):
   if path.is_file():bindings[str(path)]=sha(path)
 launch={'cohort':a.cohort,'output':str(a.output),'data_readiness':ready,'plans':plans,'file_sha256':bindings,'contracts':contracts,'scope':'Seven architectures; two encoders; five 16-shot folds; each native fit followed by zero/actual/shuffled TME','counts':{'smokes':14,'fold_allocations':70,'native_fits':70,'adapter_fits':210,'condition_fold_evaluations':280},'dispatch_stages':{'wave_A':['vila_mil','mgpath'],'wave_B':['focus','muse','hive_mil','dyko','mscpt']},'generator':'GPT 6 ASTRA','feature_panel':'shared_core62_v1','no_external_evaluation':True}
 launch['identity']=identity(launch);dump(a.output/'launch.json',launch);print(json.dumps({'launch':str(a.output/'launch.json'),'identity':launch['identity'],'counts':launch['counts']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',choices=['crc','blca'],required=True)
 for k in ['project','pgvl','results','prepared','concepts','output']:p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
