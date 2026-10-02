#!/usr/bin/env python3
"""Export separate CRC/BLCA native banks from versioned authored data.

No model imports, training registration, feature access, API calls or submission.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def csv_bytes(rows):
    buffer = io.StringIO(newline='')
    csv.writer(buffer, lineterminator='\n').writerows(rows)
    return buffer.getvalue().encode()


def build(source):
    data = json.loads(source)
    files, records = {}, []
    templates = data['templates']

    def add(cohort, method, filename, value, role, slots, *, csv_format=False, **extra):
        relative = f'{cohort}/{method}/{filename}'
        content = csv_bytes(value) if csv_format else json_bytes(value)
        files[relative] = content
        records.append(dict(file=relative, cohort=cohort, method=method, role=role,
            generator=data['generator'], provenance='generated', copied_from_upstream=False,
            class_order=data['cohorts'][cohort]['classes'], sha256=digest(content),
            slots=slots, content_count=len(slots), **extra))

    for cohort, task in data['cohorts'].items():
        labels = task['classes']
        classes = [data['classes'][label] for label in labels]
        for method, indices in [('vila_mil', (0, 2)), ('mgpath', (0, 4)), ('focus', (0, 1))]:
            rows, slots = [], []
            for scale in ('low', 'high'):
                for index, (label, cls) in enumerate(zip(labels, classes)):
                    a, b = indices if scale == 'low' else (0, 3)
                    text = templates['positional'].format(scale=scale, name=cls['name'],
                        first=cls[scale][a], second=cls[scale][b])
                    rows.append([text])
                    slots.append(dict(row=len(rows)-1, class_index=index, label=label,
                        scale=scale, clause_indices=[a, b]))
            add(cohort, method, 'two_scale.csv', rows, 'ordered_class_scale_descriptions',
                slots, csv_format=True)

        names = [cls['name'] for cls in classes]
        add(cohort, 'muse', 'class_semantics.json',
            dict(classnames=names, template=templates['muse_class'],
                rendered_prompts=[templates['muse_class'].format(name=n) for n in names]),
            'inference_class_semantics', [dict(class_index=i, label=l) for i,l in enumerate(labels)])
        for index, (label, cls) in enumerate(zip(labels, classes)):
            clauses = [(scale, i, text) for scale in ('low', 'high')
                       for i, text in enumerate(cls[scale])]
            pairs = list(itertools.combinations(range(len(clauses)), 2))
            pairs.sort(key=lambda p: digest(json_bytes([cohort, label, list(p)])))
            rows, slots = [['', '0']], []
            for i, pair in enumerate(pairs[:300]):
                first, second = [clauses[p] for p in pair]
                text = templates['retrieval'].format(name=cls['name'], first=first[2], second=second[2])
                rows.append([str(i), text])
                slots.append(dict(row=i+1, class_index=index, label=label,
                    clauses=[[first[0], first[1]], [second[0], second[1]]]))
            if len(slots) != 300 or len({r[1] for r in rows[1:]}) != 300:
                raise ValueError('MUSE requires 300 unique rows per class in this extension')
            add(cohort, 'muse', f'generated_new_{index}.csv', rows, 'training_retrieval_bank',
                slots, csv_format=True, construction='deterministic_authored_clause_pairs',
                cardinality_basis='matches_existing_PathoTME_NSCLC_and_BRCA_300_per_class')

        hierarchy, slots = {}, []
        for index, (label, cls) in enumerate(zip(labels, classes)):
            branch = {}
            for i, term in enumerate(cls['coarse_terms']):
                branch[term] = templates['hive'].format(name=cls['name'], clause=cls['low'][i])
                slots.append(dict(label=label, class_index=index, index=i, term=term,
                    scale='5x', parent_index=i, clause=['low', i]))
            for i, term in enumerate(cls['fine_terms']):
                parent = i // 3
                key = cls['coarse_terms'][parent] + ' / ' + term
                branch[key] = templates['hive'].format(name=cls['name'], clause=cls['high'][i])
                slots.append(dict(label=label, class_index=index, index=i+4, term=key,
                    scale='20x', parent_index=parent, clause=['high', i]))
            hierarchy[label] = branch
        add(cohort, 'hive_mil', 'hierarchy.json', hierarchy, 'four_coarse_twelve_fine', slots)

        descriptions, slots = {}, []
        for index, (label, cls) in enumerate(zip(labels, classes)):
            descriptions[label] = {}
            for role, scale in [('small_mag', 'low'), ('big_mag', 'high')]:
                descriptions[label][role] = [templates['single'].format(
                    scale=scale, name=cls['name'], clause=clause) for clause in cls[scale][:10]]
                slots.extend(dict(label=label, class_index=index, role=role, index=i,
                    clause=[scale, i]) for i in range(10))
        add(cohort, 'mscpt', f'description/TCGA_{cohort.upper()}.json', descriptions, 'multiscale_description_graph', slots,
            cardinality_basis='10_low_and_10_high_per_class_as_existing_BRCA_extension',
            selector_policy='existing_mean_low_description_fallback_no_separate_selector')

        rows, slots = [['Prompt', 'Classnames']], []
        for index, (label, cls) in enumerate(zip(labels, classes)):
            rows.append([templates['dyko_class'].format(name=cls['name'],
                first=cls['high'][0], second=cls['high'][3]), cls['name']])
            slots.append(dict(row=index+1, class_index=index, label=label,
                source_classname=cls['name'], clauses=[['high', 0], ['high', 3]]))
        add(cohort, 'dyko', 'class_descriptions.csv', rows, 'class_descriptions', slots, csv_format=True)
        concepts = []
        for group, texts in {**data['knowledge_cohort'][cohort], **data['knowledge_common']}.items():
            for i, text in enumerate(texts):
                concepts.append(dict(id=f'{group}_{i+1:02d}', group=group, text=text,
                    role='unlabeled_visual_concept'))
        if len(concepts) != 64 or len({x['text'] for x in concepts}) != 64:
            raise ValueError('64 unique DyKo concepts required')
        add(cohort, 'dyko', 'knowledge_bank.json', dict(version=data['version'],
            generator=data['generator'], cohort=task['cohort'], concepts=concepts,
            concept_count=64, embedding_status='not_encoded',
            embedding_requirement='same_pinned_TITAN_revision_and_native_text_encoding_as_PathoTME_BRCA',
            note='DyKo text concepts are model inputs, distinct from the fixed 62 numeric TME measurements.'),
            'unlabeled_visual_knowledge', [dict(index=i, id=c['id'], group=c['group'])
                for i,c in enumerate(concepts)])

    manifest = dict(version=data['version'], generator=data['generator'],
        source_sha256=digest(source), exporter_sha256=digest(Path(__file__).read_bytes()),
        authorship=data['authorship'], construction=data['construction'], selection=data['selection'],
        sources=data['sources'], cohorts=data['cohorts'], records=records,
        status='authored_banks_runtime_integration_pending',
        controls='Exact same bank bytes for native/zero/actual/shuffled and PLIP/CLIP-RN50 within each method/cohort.',
        no_prompt_producer_axis=True, upstream_banks_reused=False, expert_review_attested=False,
        gpu_smokes_run=False, new_cohort_jobs_submitted=False,
        runtime_note='Existing NSCLC/BRCA-only contracts remain frozen. New scoped cohort configuration and DyKo embedding are required before training.')
    files['source.json'] = source
    files['MANIFEST.json'] = json_bytes(manifest)
    return files, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files, manifest = build(args.source.read_bytes())
    # Check the entire bundle before writing. Existing differing bytes are never replaced.
    for name, content in files.items():
        path = args.output / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f'Differing existing asset: {path}')
    for name, content in files.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as handle:
                handle.write(content)
    print(json.dumps(dict(files=len(files), native_asset_files=len(manifest['records']),
        prompt_occurrences=sum(r['content_count'] for r in manifest['records']),
        generator=manifest['generator'], status=manifest['status'])))


if __name__ == '__main__':
    main()
