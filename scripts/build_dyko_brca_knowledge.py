#!/usr/bin/env python3
"""Encode a frozen, authored BRCA bank with the pinned native TITAN tower."""
import argparse, json, sys, time, os, shutil
from pathlib import Path
sys.path[:0] = ['/path/to/PGVL-Gym','/path/to/PathoTME']
from pathotme.locked_tcga import sha, identity, atomic_json


def build(source, output):
    import torch
    from common.backbones.factory import _load_titan
    torch.set_num_threads(2)
    bank = json.loads(source.read_text())
    entries = bank['concepts']
    if len(entries) != 64 or len({e['text'] for e in entries}) != 64:
        raise ValueError('64 distinct frozen morphology concepts required')
    if output.exists(): raise FileExistsError(output)
    model_path = Path(bank['encoder']['checkpoint'])
    inputs = {str(source): sha(source), str(Path(__file__).resolve()):sha(__file__)}
    for path in model_path.rglob('*'):
        if path.is_file(): inputs[str(path)] = sha(path)
    # Transformers local-directory caching can omit second-level relative imports.
    # Populate only our private module cache from byte-verified pinned sources.
    cache = Path(os.environ['HF_MODULES_CACHE']) / 'transformers_modules' / model_path.name
    cache.mkdir(parents=True, exist_ok=True)
    for src in model_path.glob('*.py'):
        dest = cache / src.name
        if dest.exists() and sha(dest) != sha(src):
            raise ValueError('private TITAN cache contains unexpected code')
        shutil.copyfile(src, dest)
    before = time.monotonic()
    print('Loading pinned TITAN on CPU for 64 authored BRCA concepts', flush=True)
    encoder = _load_titan(str(model_path), 'cpu', revision=bank['encoder']['revision'], local_files_only=True).freeze()
    texts = [e['text'] for e in entries]
    # Use the bundle's actual native tokenization and encoding interfaces.
    tokens = encoder.text.tokenize(texts)
    ids = tokens.input_ids
    raw_tokenizer = encoder.raw_model.text_encoder.tokenizer.tokenizer
    raw = raw_tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']
    if max(map(len,raw)) + 2 > ids.shape[1]:
        raise ValueError('BRCA concepts must encode in full without truncation')
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), 8):
            chunks.append(encoder.encode_text(texts[start:start+8], normalize=True).cpu().float())
            print(f'Encoded {min(start+8,len(texts))}/{len(texts)} concepts', flush=True)
    embeddings = torch.cat(chunks)
    if embeddings.shape != (64,768) or not torch.isfinite(embeddings).all():
        raise ValueError('invalid TITAN concept tensor')
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(64), atol=1e-5, rtol=1e-5)
    output.mkdir(parents=True)
    tensor = output/'text_embeddings_BRCA.pt'
    torch.save(embeddings, tensor)
    atomic_json(output/'encoding.json', dict(status='completed', source_sha256=inputs,
        source_bank_identity=identity(bank), tensor_path=str(tensor), tensor_sha256=sha(tensor),
        shape=[64,768], dtype='float32', normalized=True, token_ids=ids.tolist(),
        untruncated_content_token_counts=list(map(len,raw)), encoder=bank['encoder'],
        device='cpu', wall_seconds=time.monotonic()-before, patient_data_used=False,
        independent_pathology_review=False, provenance='assistant-authored text; actual pinned TITAN embedding call'))
    print(json.dumps({'status':'completed','shape':[64,768],'tensor_sha256':sha(tensor)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:build(a.source.resolve(),a.output.resolve())
    else:print(json.dumps({'source':str(a.source),'output':str(a.output),'would_encode':64}))
