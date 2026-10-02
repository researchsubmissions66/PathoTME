#!/usr/bin/env python3
"""Check native bank parsers, ordered roles and untruncated local token budgets."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(bundle, pgvl, plip_tokenizer):
    sys.path.insert(0, str(pgvl))
    from common.prompts.focus import load_focus_prompt_bank
    from common.prompts.vila_mil import load_vila_prompt_bank
    from common.prompts.muse import load_muse_prompt_csv
    from methods.mgpath.prompts import load_prompt_bank as mgload
    from methods.dyko.prompts import load_prompt_bank as dyload
    from methods.hive_mil.prompts import load_hierarchical_prompts
    from tokenizers import Tokenizer
    spec = importlib.util.spec_from_file_location('local_clip_bpe', pgvl/'clip/simple_tokenizer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clip = module.SimpleTokenizer()
    plip = Tokenizer.from_file(str(plip_tokenizer))
    plip.no_padding()
    plip.no_truncation()
    manifest = json.loads((bundle/'MANIFEST.json').read_text())
    source = json.loads((bundle/'source.json').read_text())
    assert sha(bundle/'source.json') == manifest['source_sha256']
    original_template = json.loads((pgvl/'text_prompts/muse/class_template.json').read_text())['template']
    results = []
    for record in manifest['records']:
        path = bundle/record['file']
        assert sha(path) == record['sha256']
        labels = record['class_order']
        label_dict = {l:i for i,l in enumerate(labels)}
        method, role = record['method'], record['role']
        texts, context, limit, parser = [], 16, 77, 'native_parser'
        if method in ('vila_mil', 'focus'):
            loader = load_vila_prompt_bank if method == 'vila_mil' else load_focus_prompt_bank
            bank = loader(path, class_names=labels, file_class_names=labels,
                record=dict(provenance='generated', source='assistant_conversation', copied_from_upstream=False))
            texts = list(bank.prompts)
        elif method == 'mgpath':
            texts = list(mgload(path, label_dict, labels+labels).prompts)
        elif method == 'muse' and role == 'training_retrieval_bank':
            texts = list(load_muse_prompt_csv(path, expected_rows=300,
                expected_sha256=record['sha256']).descriptions)
            assert len(set(texts)) == 300
            context = 0
        elif method == 'muse':
            value = json.loads(path.read_text())
            texts = value['rendered_prompts']
            expected_names = [source['classes'][l]['name'] for l in labels]
            assert value['classnames'] == expected_names
            assert texts == [original_template.format(n) for n in expected_names]
            context, parser = 0, 'exact_existing_class_template'
        elif method == 'hive_mil':
            value = load_hierarchical_prompts(path, label_dict)
            assert list(value) == labels
            texts = [f'{term} : {explanation}' for branch in value.values()
                     for term, explanation in branch.items()]
            for label, branch in value.items():
                keys = list(branch)
                assert len(keys) == 16
                for i, key in enumerate(keys[4:]):
                    assert key.startswith(keys[i//3] + ' / ')
        elif method == 'mscpt':
            value = json.loads(path.read_text())
            assert path.parent.name == 'description' and list(value) == labels
            assert record['selector_policy'] == 'existing_mean_low_description_fallback_no_separate_selector'
            for branch in value.values():
                assert list(branch) == ['small_mag', 'big_mag']
                assert all(len(v) == 10 for v in branch.values())
                texts.extend(t for v in branch.values() for t in v)
            # Conservative 12-token prefix probe on both banks. Only big_mag has this prefix at runtime.
            context, limit, parser = 12, 64, 'native_json_shape_and_directory_contract'
        elif method == 'dyko' and role == 'class_descriptions':
            names = [source['classes'][l]['name'] for l in labels]
            texts = list(dyload(path, label_dict, names, labels).descriptions)
        elif method == 'dyko':
            value = json.loads(path.read_text())
            texts = [c['text'] for c in value['concepts']]
            assert len(texts) == len(set(texts)) == 64
            assert all('class' not in c and 'label' not in c for c in value['concepts'])
            context, parser = 0, 'unlabeled_knowledge_schema_TITAN_encoding_pending'
        else:
            raise ValueError(method)
        assert len(texts) == record['content_count']
        assert all(isinstance(t, str) and t.strip() for t in texts)
        prefix = ' '.join(['X']*context)
        consumed = [f'{prefix} {t}' if context else t for t in texts]
        clip_counts = [len(clip.encode(t))+2 for t in consumed]
        plip_counts = [len(plip.encode(t, add_special_tokens=True).ids) for t in consumed]
        for name, counts in [('clip-rn50', clip_counts), ('plip', plip_counts)]:
            if max(counts) > limit:
                index = counts.index(max(counts))
                raise ValueError(f'{record["file"]}: {name} {max(counts)} > {limit}: {texts[index]}')
        results.append(dict(file=record['file'], sha256=record['sha256'], parser=parser,
            strings=len(texts), context_probe=context, context_limit=limit,
            clip_rn50_max_tokens=max(clip_counts), plip_max_tokens=max(plip_counts),
            untruncated=True))
    assert len(results) == 20
    assert 'torch' not in sys.modules
    return dict(status='passed', native_asset_files=len(results),
        text_occurrences=sum(r['strings'] for r in results), generator=manifest['generator'],
        manifest_sha256=sha(bundle/'MANIFEST.json'), validator_sha256=sha(__file__),
        clip_bpe_code_sha256=sha(pgvl/'clip/simple_tokenizer.py'),
        clip_bpe_vocabulary_sha256=sha(pgvl/'clip/bpe_simple_vocab_16e6.txt.gz'),
        plip_tokenizer_sha256=sha(plip_tokenizer), results=results,
        scope='Native data parsers where available; full term/explanation/context token checks with cached tokenizers. No model construction or encoder weights.',
        titan_concept_encoding='pending', independent_pathology_review='not_attested',
        runtime_integration='pending', gpu_jobs_submitted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--pgvl-root', type=Path, required=True)
    p.add_argument('--plip-tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = validate(a.bundle.resolve(), a.pgvl_root.resolve(), a.plip_tokenizer.resolve())
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','native_asset_files','text_occurrences','generator']}))
