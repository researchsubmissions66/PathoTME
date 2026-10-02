"""Encode both frozen cohort concept banks once with native pinned TITAN on CPU."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main(a):
 sys.path.insert(0,str(a.pgvl))
 import torch
 from common.backbones.factory import _load_titan
 torch.set_num_threads(2)
 revision='dac6773d9961cfc75503440676ff157a2c6e8d2e'
 if a.weights.name!=revision:raise ValueError('Pinned TITAN revision required')
 if a.output.exists():raise FileExistsError(a.output)
 source={c:a.banks/c/'dyko/knowledge_bank.json' for c in ['crc','blca']}
 banks={c:json.loads(p.read_text()) for c,p in source.items()}
 for bank in banks.values():
  if bank['generator']!='GPT 6 ASTRA' or len(bank['concepts'])!=64:raise ValueError('Bank identity mismatch')
 inputs={str(p):sha(p) for p in [Path(__file__),*source.values()]}
 print('Hashing pinned TITAN source and weights',flush=True)
 for p in a.weights.rglob('*'):
  if p.is_file():inputs[str(p)]=sha(p)
 cache=Path(os.environ['HF_MODULES_CACHE'])/'transformers_modules'/revision
 cache.mkdir(parents=True,exist_ok=True)
 for p in a.weights.glob('*.py'):shutil.copyfile(p,cache/p.name)
 started=time.monotonic()
 print('Loading native TITAN text encoder on login CPU',flush=True)
 encoder=_load_titan(str(a.weights),'cpu',revision=revision,local_files_only=True).freeze()
 a.output.mkdir(parents=True)
 for cohort,bank in banks.items():
  texts=[x['text'] for x in bank['concepts']]
  tokens=encoder.text.tokenize(texts)
  tokenizer=encoder.raw_model.text_encoder.tokenizer.tokenizer
  raw=tokenizer(texts,add_special_tokens=False,padding=False,truncation=False)['input_ids']
  if max(map(len,raw))+2>tokens.input_ids.shape[1]:raise ValueError('Concept truncation')
  chunks=[]
  with torch.no_grad():
   for start in range(0,64,8):
    chunks.append(encoder.encode_text(texts[start:start+8],normalize=True).cpu().float())
    print(cohort,'encoded',start+8,'/64',flush=True)
  value=torch.cat(chunks)
  if value.shape!=(64,768) or not torch.isfinite(value).all():raise ValueError('Invalid concept tensor')
  torch.testing.assert_close(value.norm(dim=1),torch.ones(64),atol=1e-5,rtol=1e-5)
  out=a.output/cohort;out.mkdir();path=out/'concepts.pt';torch.save(value,path)
  record={'status':'completed','generator':'GPT 6 ASTRA','source_bank_path':str(source[cohort]),'source_bank_sha256':sha(source[cohort]),'tensor_path':str(path),'tensor_sha256':sha(path),'shape':[64,768],'normalized':True,'encoder':'TITAN','revision':revision,'device':'cpu','token_counts':list(map(len,raw)),'input_sha256':inputs,'elapsed_since_model_load_start':time.monotonic()-started,'patient_data_used':False}
  (out/'encoding.json').write_text(json.dumps(record,indent=2)+'\n')
 print('Both concept banks encoded',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['banks','weights','output','pgvl']:p.add_argument('--'+key,type=Path,required=True)
 main(p.parse_args())
