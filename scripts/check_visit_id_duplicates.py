"""Check 门诊号 uniqueness within files, across categories, and across train/test.

Uses only the Python standard library. Inputs are read-only; reports are local.
String IDs are stripped of surrounding whitespace; leading zeros are preserved.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check(data_dir: Path, output_dir: Path) -> dict:
    files = sorted(p for p in data_dir.glob('*/*.json') if p.name in ('train.json', 'test.json'))
    if not files:
        raise FileNotFoundError(f'No category train.json/test.json files found: {data_dir}')
    for category in {p.parent for p in files}:
        for split in ('train', 'test'):
            if not (category / f'{split}.json').is_file():
                raise FileNotFoundError(category / f'{split}.json')

    occurrences = defaultdict(Counter)
    file_ids, file_rows = {}, []
    file_info = {}
    for path in files:
        records = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(records, dict):
            records = records.get('data', records.get('records'))
        if not isinstance(records, list):
            raise ValueError(f'Expected a record list: {path}')
        relative = path.relative_to(data_dir).as_posix()
        counts = Counter()
        missing = 0
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f'{relative}, record {index}: expected an object')
            value = record.get('门诊号')
            if value is None or (isinstance(value, str) and not value.strip()):
                missing += 1
                continue
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError(f'{relative}, record {index}: 门诊号 must be a string or integer')
            visit = str(value).strip()
            counts[visit] += 1
            occurrences[visit][relative] += 1
        file_ids[relative] = set(counts)
        file_info[relative] = (path.parent.name, path.stem)
        file_rows.append({'file': relative, 'records': len(records), 'missing_visit_ids': missing,
            'unique_visit_ids': len(counts), 'duplicated_visit_ids': sum(n > 1 for n in counts.values()),
            'extra_duplicate_records': sum(n - 1 for n in counts.values())})

    pairs = []
    for left, right in combinations(file_ids, 2):
        overlap = len(file_ids[left] & file_ids[right])
        pairs.append({'file_a': left, 'file_b': right,
            'same_category': file_info[left][0] == file_info[right][0],
            'same_split': file_info[left][1] == file_info[right][1],
            'shared_visit_ids': overlap})
    duplicates = []
    for visit, locations in sorted(occurrences.items()):
        if sum(locations.values()) <= 1:
            continue
        categories = {file_info[name][0] for name in locations}
        splits = {file_info[name][1] for name in locations}
        duplicates.append({'visit_id': visit, 'total_records': sum(locations.values()),
            'file_count': len(locations), 'within_file_duplicate': any(n > 1 for n in locations.values()),
            'cross_category': len(categories) > 1, 'cross_train_test': len(splits) > 1,
            'locations': json.dumps(dict(locations), ensure_ascii=False)})
    train_ids = set().union(*(ids for name, ids in file_ids.items() if file_info[name][1] == 'train'))
    test_ids = set().union(*(ids for name, ids in file_ids.items() if file_info[name][1] == 'test'))
    summary = {
        'data_dir': str(data_dir.resolve()), 'files_checked': len(files),
        'total_records': sum(row['records'] for row in file_rows),
        'missing_visit_id_records': sum(row['missing_visit_ids'] for row in file_rows),
        'global_unique_visit_ids': len(occurrences),
        'global_duplicated_visit_ids': len(duplicates),
        'global_extra_duplicate_records': sum(sum(c.values()) - 1 for c in occurrences.values()),
        'within_file_duplicated_visit_ids': sum(row['within_file_duplicate'] for row in duplicates),
        'cross_file_duplicated_visit_ids': sum(row['file_count'] > 1 for row in duplicates),
        'cross_category_duplicated_visit_ids': sum(row['cross_category'] for row in duplicates),
        'train_unique_visit_ids': len(train_ids), 'test_unique_visit_ids': len(test_ids),
        'global_train_test_shared_visit_ids': len(train_ids & test_ids),
        'same_category_train_test_overlap': {
            file_info[row['file_a']][0]: row['shared_visit_ids']
            for row in pairs if row['same_category'] and not row['same_split']},
        'all_records_have_globally_unique_visit_ids': not duplicates and not any(
            row['missing_visit_ids'] for row in file_rows),
        'note': 'Counts are unique IDs unless named records; overlap categories are not additive. '
                'Duplicate IDs indicate repeated identifiers, not necessarily identical clinical records.'}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'by_file.csv', file_rows, list(file_rows[0]))
    write_csv(output_dir / 'file_pair_overlap.csv', pairs,
              ['file_a', 'file_b', 'same_category', 'same_split', 'shared_visit_ids'])
    write_csv(output_dir / 'duplicate_visit_ids.csv', duplicates,
              ['visit_id', 'total_records', 'file_count', 'within_file_duplicate',
               'cross_category', 'cross_train_test', 'locations'])
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/broad_category_splits_standardized_6')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/visit_id_audit')
    args = parser.parse_args()
    summary = check(args.data_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Reports saved to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
