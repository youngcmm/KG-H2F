"""Describe current category memberships using unique visits and explicit denominators."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cohort_multilabel'
CATEGORIES = ['sleep_dermatologic', 'sleep_emotional', 'sleep_fatigue_deficiency',
              'sleep_neurologic', 'sleep_oncologic', 'sleep_urinary']
FIELDS = {'disease': '西医诊断', 'tcm_pattern': '四诊对应证候_标准名称',
          'tongue_pulse': '舌苔脉象_标准名称', 'herb': '标准处方药材名称'}


def tokens(value):
    return {s.strip() for s in re.split(r'[,，;；、]+', str(value or ''))
            if s.strip() and s.strip() not in ('无', 'nan')}


def export(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    visits, records, hashes = {}, [], {}
    for category in CATEGORIES:
        for split in ('train', 'test'):
            p = ROOT / 'data/broad_category_splits_standardized_6' / category / f'{split}.json'
            hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            rows = json.loads(p.read_text(encoding='utf-8-sig'))
            seen = set()
            for row in rows:
                key = str(row['门诊号']).strip()
                if not key or key in seen:
                    raise ValueError(f'Missing or duplicate visit within {p}')
                seen.add(key)
                entity = {kind: tokens(row.get(field)) for kind, field in FIELDS.items()}
                visit = visits.setdefault(key, {'split': split, 'categories': set(),
                    'diagnosis_text': str(row.get(FIELDS['disease'], '')),
                    'entities': {kind: set() for kind in FIELDS}})
                if visit['split'] != split:
                    raise ValueError('A unique visit appears in both train and test')
                if visit['diagnosis_text'] != str(row.get(FIELDS['disease'], '')):
                    raise ValueError('A shared visit has inconsistent diagnoses across categories')
                visit['categories'].add(category)
                for kind in FIELDS:
                    visit['entities'][kind].update(entity[kind])
                records.append({'category': category, 'split': split, 'entities': entity})
    marginals, diagnosis = [], []
    for category in CATEGORIES:
        group = [v for v in visits.values() if category in v['categories']]
        for split in ('all', 'train', 'test'):
            subset = [v for v in group if split == 'all' or v['split'] == split]
            denominator = sum(split == 'all' or v['split'] == split for v in visits.values())
            single = sum(len(v['categories']) == 1 for v in subset)
            marginals.append({'category': category, 'split': split, 'visits': len(subset),
                'single_label_visits': single, 'multilabel_visits': len(subset)-single,
                'unique_cohort_denominator': denominator, 'cohort_percent': 100*len(subset)/denominator})
        term_counts = Counter(t for v in group for t in v['entities']['disease'] if t != '睡眠障碍')
        for term, count in sorted(term_counts.items(), key=lambda item: (-item[1], item[0])):
            diagnosis.append({'category': category, 'diagnosis': term, 'visits': count,
                              'category_visits': len(group), 'category_percent': 100*count/len(group)})
    pairs = [{'category_a': a, 'category_b': b,
              'shared_visits': sum(a in v['categories'] and b in v['categories'] for v in visits.values())}
             for a in CATEGORIES for b in CATEGORIES]
    memberships = Counter(tuple(sorted(v['categories'])) for v in visits.values())
    patterns = [{'categories': '|'.join(c), 'number_of_labels': len(c), 'visits': n}
                for c,n in sorted(memberships.items(), key=lambda item: (-item[1], item[0]))]
    nodes_path = ROOT / 'outputs/knowledge_graph/sleep_ehr_tcm_kg_nodes.csv'
    edges_path = ROOT / 'outputs/knowledge_graph/sleep_ehr_tcm_kg_edges.csv'
    with nodes_path.open(encoding='utf-8-sig', newline='') as f:
        nodes = list(csv.DictReader(f))
    node_counts = Counter(n['node_type'] for n in nodes)
    vocab = {kind: {n['node_name'] for n in nodes if n['node_type'] == kind} for kind in FIELDS}
    with edges_path.open(encoding='utf-8-sig', newline='') as f:
        relations = Counter(row['relation'] for row in csv.DictReader(f))
    for p in (nodes_path, edges_path):
        hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    export('kg_counts.csv', [{'type': 'node', 'name': k, 'count': v} for k,v in node_counts.items()] +
           [{'type': 'relation', 'name': k, 'count': v} for k,v in relations.items()])
    coverage = []

    def cover(scope, kind, test_sets, names):
        unique = set().union(*test_sets)
        total = sum(len(s) for s in test_sets)
        seen = sum(len(s & names) for s in test_sets)
        coverage.append({'scope': scope, 'entity': kind, 'holdout_unique': len(unique),
            'seen_unique': len(unique & names), 'holdout_associations': total, 'seen_associations': seen,
            'unique_percent': 100*len(unique & names)/len(unique), 'association_percent': 100*seen/total})
    for kind in FIELDS:
        cover('overall', kind, [v['entities'][kind] for v in visits.values() if v['split']=='test'], vocab[kind])
    for category in CATEGORIES:
        training = [r['entities']['herb'] for r in records if r['category']==category and r['split']=='train']
        testing = [r['entities']['herb'] for r in records if r['category']==category and r['split']=='test']
        cover(category, 'herb', testing, set().union(*training))
    diagnostic_labels = defaultdict(set)
    for v in visits.values():
        diagnostic_labels[v['diagnosis_text']].add(tuple(sorted(v['categories'])))
    shared_emotional_neurologic = [v for v in visits.values()
        if {'sleep_emotional', 'sleep_neurologic'} <= v['categories']]
    summary = {'record_memberships': len(records), 'unique_visits': len(visits),
        'unique_train_visits': sum(v['split']=='train' for v in visits.values()),
        'unique_test_visits': sum(v['split']=='test' for v in visits.values()),
        'train_memberships': sum(r['split']=='train' for r in records),
        'test_memberships': sum(r['split']=='test' for r in records),
        'label_multiplicity': dict(sorted(Counter(len(v['categories']) for v in visits.values()).items())),
        'all_have_sleep_disorder': all('睡眠障碍' in v['entities']['disease'] for v in visits.values()),
        'distinct_complete_diagnosis_strings': len(diagnostic_labels),
        'diagnosis_strings_with_inconsistent_labels': sum(len(labels)>1 for labels in diagnostic_labels.values()),
        'shared_emotional_neurologic_visits': len(shared_emotional_neurologic),
        'shared_sleep_neurasthenia_only': sum(v['entities']['disease']=={'睡眠障碍','神经衰弱'}
            for v in shared_emotional_neurologic),
        'shared_sleep_dizziness_anxiety_only': sum(v['entities']['disease']=={'睡眠障碍','头晕和眩晕','焦虑状态'}
            for v in shared_emotional_neurologic),
        'kg_nodes': sum(node_counts.values()), 'kg_edges': sum(relations.values())}
    assert sum(row['visits']*row['number_of_labels'] for row in patterns) == len(records)
    assert node_counts['visit'] == summary['train_memberships']
    assert node_counts['patient'] == summary['unique_train_visits']
    export('category_membership.csv', marginals)
    export('category_overlap.csv', pairs)
    export('membership_patterns.csv', patterns)
    export('diagnosis_composition.csv', diagnosis)
    export('entity_coverage.csv', coverage)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
