"""Frozen training-only herb-pool ablation, followed by existing temporal evaluation.

Run discovery and evaluation as separate processes. No test files are read (or
hashed) by discovery. All-489 is the complete eligible mother table.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import basic_discover as bd
from category import CATEGORIES, DEFAULT_DATA_DIR, load_prescriptions, load_scores

DEFAULT_OUT = ROOT / 'outputs/formula_strategy_comparison'
FIXED = ('ULTRA-400', 'Frequency-400', 'All-489')
METRICS = ('mean_test_support', 'median_test_support',
           'mean_formula_enrichment', 'median_formula_enrichment')


def log(message):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {message}', flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig',
                 compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 0}
                 if str(path).endswith('.gz') else None)


def bit_tidsets(records):
    result = {}
    for i, record in enumerate(records):
        bit = 1 << i
        for herb in record:
            result[herb] = result.get(herb, 0) | bit
    return result


def bit_count(formula, tidsets):
    bits = None
    for herb in formula:
        value = tidsets.get(herb, 0)
        bits = value if bits is None else bits & value
        if not bits:
            return 0
    return bits.bit_count() if bits is not None else 0


def enumerate_mother(records, names, min_count=5):
    """Exact Eclat intersections; antimonotone support pruning is lossless."""
    tids = bit_tidsets(records)
    initial = [(i, tids.get(h, 0)) for i, h in enumerate(names)
               if tids.get(h, 0).bit_count() >= min_count]
    formulas, counts = [], []

    def extend(prefix, extensions):
        for k, (herb, bits) in enumerate(extensions):
            formula = prefix + (herb,)
            if len(formula) >= 3:
                formulas.append(formula + (-1,) * (4 - len(formula)))
                counts.append(bits.bit_count())
            if len(formula) < 4:
                remaining = []
                for other, other_bits in extensions[k + 1:]:
                    joint = bits & other_bits
                    if joint.bit_count() >= min_count:
                        remaining.append((other, joint))
                if remaining:
                    extend(formula, remaining)
    extend((), initial)
    return np.asarray(formulas, dtype=np.int16), np.asarray(counts, dtype=np.int32)


def jaccard(left, right):
    return len(left & right) / len(left | right) if left | right else 1.0


def discover(args):
    out = args.output_dir
    if (out / 'discovery_manifest.json').exists():
        raise FileExistsError('Discovery is frozen. Use a new output directory for a new design.')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'mother').mkdir(exist_ok=True)
    (out / 'frozen').mkdir(exist_ok=True)
    rankings = {cat: load_scores(args.score_dir, cat) for cat in CATEGORIES}
    names = sorted(rankings[next(iter(CATEGORIES))].herb_name)
    assert len(names) == 489 and len(set(names)) == 489
    assert all(set(r.herb_name) == set(names) for r in rankings.values())
    assert all(r['model'].eq('ULTRA-50G-targeted').all() for r in rankings.values())
    name_to_id = {h: i for i, h in enumerate(names)}
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(6)]
    selected, inventories, overlaps, frequencies = [], [], [], []
    all_pools = np.empty((6, args.repeats + 3, 489), dtype=bool)
    inputs = {}
    for ci, (cat, ranking) in enumerate(rankings.items()):
        path = args.data_dir / cat / 'train.json'
        inputs[str(path.resolve())] = sha(path)
        score_path = args.score_dir / f'comorbidity_{cat}_selected_herb_scores.csv'
        inputs[str(score_path.resolve())] = sha(score_path)
        records = load_prescriptions(args.data_dir, cat, 'train')
        unknown = set().union(*records) - set(names)
        assert not unknown, f'Unexpected training herbs: {unknown}'
        freq = Counter(h for record in records for h in record)
        frequency_order = sorted(names, key=lambda h: (-freq[h], h))
        frequency_rank = {h: i + 1 for i, h in enumerate(frequency_order)}
        ranks = dict(zip(ranking.herb_name, ranking.model_rank))
        for h in names:
            frequencies.append(dict(category=cat, herb_name=h, train_count=freq[h],
                                    frequency_rank=frequency_rank[h], ultra_rank=ranks[h]))
        log(f'{cat}: enumerating all eligible 3-4 herb formulas from {len(records):,} training records')
        ids, counts = enumerate_mother(records, names)
        sizes = (ids >= 0).sum(axis=1)
        herb_scores = np.array([1 - (ranks[h] - 1) / 489 for h in names] + [0.0])
        mean_ranks = herb_scores[ids].sum(axis=1) / sizes
        formula_scores = np.sqrt(mean_ranks * np.log1p(counts) / np.log1p(counts.max()))
        strings = np.array(['|'.join(names[i] for i in row if i >= 0) for row in ids])
        order = np.lexsort((strings, sizes, -counts, -formula_scores))
        ids, counts, sizes, strings, formula_scores = (
            a[order] for a in (ids, counts, sizes, strings, formula_scores))
        master = pd.DataFrame(dict(category=cat, category_cn=CATEGORIES[cat],
            formula_id=np.arange(len(ids)), rank=np.arange(1, len(ids) + 1),
            base_formula=strings, formula_size=sizes, formula_score=formula_scores,
            train_count=counts, train_total=len(records), train_support=counts / len(records)))
        save_csv(master, out / 'mother' / f'{cat}.csv.gz')
        pools = []
        for herbs in (ranking.herb_name.iloc[:400], frequency_order[:400], names):
            mask = np.zeros(490, dtype=bool)
            mask[[name_to_id[h] for h in herbs]] = True
            mask[-1] = True  # padding for 3-herb formulas
            pools.append(mask)
        for rep in range(args.repeats):
            mask = np.zeros(490, dtype=bool)
            mask[rngs[ci].choice(489, 400, replace=False)] = True
            mask[-1] = True
            pools.append(mask)
        formula_sets, pool_sets = [], []
        for pi, mask in enumerate(pools):
            strategy = FIXED[pi] if pi < 3 else 'Random-400'
            repeat = 0 if pi < 3 else pi - 2
            eligible = mask[ids].all(axis=1)
            chosen = np.flatnonzero(eligible)[:30]
            assert len(chosen) == 30
            all_pools[ci, pi] = mask[:489]
            pool_sets.append(set(np.flatnonzero(mask[:489])))
            formula_sets.append(set(chosen))
            block = master.iloc[chosen].copy()
            block['mother_rank'] = block['rank']
            block['rank'] = np.arange(1, 31)
            block.insert(0, 'repeat', repeat)
            block.insert(0, 'strategy', strategy)
            selected.append(block)
            inventories.append(dict(category=cat, strategy=strategy, repeat=repeat,
                pool_size=int(mask[:489].sum()), eligible_formulas=int(eligible.sum()),
                eligible_3herb=int((eligible & (sizes == 3)).sum()),
                eligible_4herb=int((eligible & (sizes == 4)).sum()),
                mother_formulas=len(ids), coverage=float(eligible.mean()), selected_formulas=30))
        pairs = list(combinations(range(3), 2)) + [(i, j) for j in range(3, len(pools)) for i in range(3)]
        for i, j in pairs:
            overlaps.append(dict(category=cat, strategy_a=FIXED[i],
                strategy_b=FIXED[j] if j < 3 else 'Random-400',
                repeat=0 if j < 3 else j - 2,
                herb_pool_jaccard=jaccard(pool_sets[i], pool_sets[j]),
                top30_formula_jaccard=jaccard(formula_sets[i], formula_sets[j])))
        log(f'{cat}: froze {len(master):,} mother formulas and {len(pools)} Top-30 lists')
    save_csv(pd.concat(selected, ignore_index=True), out / 'frozen/top30_selection.csv.gz')
    save_csv(pd.DataFrame(inventories), out / 'frozen/candidate_inventory.csv')
    save_csv(pd.DataFrame(overlaps), out / 'frozen/jaccard_by_repeat.csv')
    save_csv(pd.DataFrame(frequencies), out / 'frozen/herb_frequencies_and_ranks.csv')
    np.savez_compressed(out / 'frozen/herb_pools.npz', pools=all_pools,
                        categories=np.array(list(CATEGORIES)), herbs=np.array(names),
                        strategies=np.array(list(FIXED) + ['Random-400'] * args.repeats),
                        repeats=np.array([0, 0, 0] + list(range(1, args.repeats + 1))))
    outputs = {str(p.relative_to(out)): sha(p) for sub in ('mother', 'frozen')
               for p in sorted((out / sub).iterdir())}
    manifest = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        model='ULTRA-50G-targeted', input_sha256=inputs, output_sha256=outputs,
        seed=args.seed, random_repeats=args.repeats, random_generator='numpy PCG64; SeedSequence.spawn(6)',
        category_random_pools='independent per category; paired repeat IDs for macro statistics',
        frequency_pool='category-specific prescription presence counts; herb-name ascending ties',
        universe_size=489, pool_size=400, min_size=3, max_size=4, min_count=5, top_k=30,
        formula_score='sqrt(mean(1-(ULTRA_full_rank-1)/489) * log1p(train_count)/log1p(category_mother_max_count))',
        ranking_ties=['train_count descending', 'formula_size ascending', 'canonical herb names ascending'],
        shared_scoring=True, temporal_files_accessed=False,
        data_dir=str(args.data_dir.resolve()), score_dir=str(args.score_dir.resolve()),
        code_sha256=sha(Path(__file__)))
    save_json(out / 'discovery_manifest.json', manifest)
    log('DISCOVERY FROZEN. Temporal data have not been read.')


def verify_frozen(out):
    manifest = json.loads((out / 'discovery_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['output_sha256'].items():
        assert sha(out / relative) == expected, f'Frozen artifact changed: {relative}'
    for absolute, expected in manifest['input_sha256'].items():
        assert sha(absolute) == expected, f'Discovery input changed: {absolute}'
    return manifest


def evaluate(args):
    out = args.output_dir
    manifest = verify_frozen(out)
    data_dir = Path(manifest['data_dir'])
    temporal = out / 'temporal'
    temporal.mkdir(exist_ok=True)
    selections = pd.read_csv(out / 'frozen/top30_selection.csv.gz')
    selected_results, qc = [], []
    test_inputs = {str((data_dir / cat / 'test.json').resolve()): sha(data_dir / cat / 'test.json')
                   for cat in CATEGORIES}
    # Existing validate() and its formulas remain unchanged. Exact integer
    # bitsets accelerate only the two tidset helpers; parity is verified below.
    for cat in CATEGORIES:
        master = pd.read_csv(out / 'mother' / f'{cat}.csv.gz')
        log(f'{cat}: evaluating all {len(master):,} mother candidates using basic_discover.validate')
        empty_bootstrap = master.iloc[:0]
        sample = master.sample(min(100, len(master)), random_state=42)
        reference = bd.validate(sample, empty_bootstrap, data_dir, 0)
        with patch.object(bd, 'build_tidsets', bit_tidsets), patch.object(bd, 'count_from_tidsets', bit_count):
            accelerated = bd.validate(sample, empty_bootstrap, data_dir, 0)
            pd.testing.assert_frame_equal(reference, accelerated, check_exact=True)
            result = bd.validate(master, empty_bootstrap, data_dir, 0)
        result.insert(2, 'formula_id', master.formula_id.to_numpy())
        # No per-formula bootstrap requested: remove legacy zero placeholders.
        result = result.drop(columns=['formula_enrichment_ci_lower', 'formula_enrichment_ci_upper', 'bootstrap_validated'])
        result['train_count'] = master.train_count.to_numpy()
        save_csv(result, temporal / f'{cat}_all_candidates.csv.gz')
        chosen = selections.loc[selections.category.eq(cat)]
        evidence_cols = ['formula_id', 'test_count', 'test_total', 'test_support',
                         'test_support_ci_lower', 'test_support_ci_upper', 'formula_enrichment']
        selected_results.append(chosen.merge(result[evidence_cols], on='formula_id', validate='many_to_one'))
        qc.append(dict(check=f'{cat}: 100 formulas bitset vs original set-based validate', passed=True))
        qc.append(dict(check=f'{cat}: all mother candidates evaluated', passed=len(result) == len(master)))
        log(f'{cat}: complete')
    top = pd.concat(selected_results, ignore_index=True)
    assert len(top) == 6 * 30 * (3 + manifest['random_repeats'])
    save_csv(top, temporal / 'top30_evaluated.csv.gz')
    save_csv(top.loc[top.strategy.ne('Random-400')], temporal / 'top30_deterministic.csv')
    qc.append(dict(check='All category/strategy/repeat groups have exactly 30 formulas',
                   passed=bool(top.groupby(['category', 'strategy', 'repeat']).size().eq(30).all())))
    save_csv(pd.DataFrame(qc), out / 'QC_evaluation.csv')
    save_json(out / 'evaluation_manifest.json', dict(evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        discovery_manifest_sha256=sha(out / 'discovery_manifest.json'), test_input_sha256=test_inputs,
        evaluator='basic_discover.validate; exact bitset tidset backend; parity against original verified',
        evaluator_sha256=sha(ROOT / 'basic_discover.py'), loader_sha256=sha(ROOT / 'category.py'),
        enrichment='(target_test_count/target_test_total + 1e-8)/(sum_category_counts/sum_category_totals + 1e-8)',
        denominator='category-membership weighted; cross-category visit duplicates retained as in existing evaluator',
        zero_occurrence='EPS inherited: if absent from all categories enrichment equals 1',
        per_formula_bootstrap=False))
    summarize(out)


def summarize(out):
    top = pd.read_csv(out / 'temporal/top30_evaluated.csv.gz')
    per = top.groupby(['category', 'strategy', 'repeat'], sort=False).agg(
        mean_test_support=('test_support', 'mean'), median_test_support=('test_support', 'median'),
        mean_formula_enrichment=('formula_enrichment', 'mean'),
        median_formula_enrichment=('formula_enrichment', 'median')).reset_index()
    macro = per.groupby(['strategy', 'repeat'], sort=False)[list(METRICS)].mean().reset_index()
    macro.insert(0, 'category', 'MACRO')
    per = pd.concat([per, macro], ignore_index=True)
    save_csv(per, out / 'metrics_by_repeat.csv')
    summary = []
    for (cat, strategy), group in per.groupby(['category', 'strategy'], sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy()
            row = dict(category=cat, strategy=strategy, metric=metric,
                       value=float(values.mean()), repeats=len(values),
                       empirical_lower=float(np.quantile(values, .025)) if len(values) > 1 else np.nan,
                       empirical_upper=float(np.quantile(values, .975)) if len(values) > 1 else np.nan)
            summary.append(row)
    summary = pd.DataFrame(summary)
    save_csv(summary, out / 'metric_summary.csv')
    percentiles = []
    for cat in list(CATEGORIES) + ['MACRO']:
        group = per[per.category.eq(cat)]
        for metric in METRICS:
            ultra = group.loc[group.strategy.eq('ULTRA-400'), metric].iloc[0]
            random = group.loc[group.strategy.eq('Random-400'), metric].to_numpy()
            equal = np.isclose(random, ultra, rtol=0, atol=1e-12)
            below = (random < ultra) & ~equal
            above = (random > ultra) & ~equal
            percentiles.append(dict(category=cat, metric=metric, ultra_value=ultra,
                random_mean=random.mean(), random_empirical_lower=np.quantile(random, .025),
                random_empirical_upper=np.quantile(random, .975), n_random=len(random),
                n_random_below=int(below.sum()), n_random_equal=int(equal.sum()), n_random_above=int(above.sum()),
                ultra_empirical_percentile=100 * (below.sum() + .5 * equal.sum()) / len(random),
                ultra_percentile_strict=100 * below.mean(),
                ultra_percentile_weak=100 * (below.sum() + equal.sum()) / len(random)))
    save_csv(pd.DataFrame(percentiles), out / 'ultra_random_percentiles.csv')
    overlap = pd.read_csv(out / 'frozen/jaccard_by_repeat.csv')
    ovmetrics = ['herb_pool_jaccard', 'top30_formula_jaccard']
    ovmacro = overlap.groupby(['strategy_a', 'strategy_b', 'repeat'])[ovmetrics].mean().reset_index()
    ovmacro.insert(0, 'category', 'MACRO')
    overlap = pd.concat([overlap, ovmacro], ignore_index=True)
    save_csv(overlap, out / 'jaccard_by_repeat.csv')
    ovrows = []
    for keys, group in overlap.groupby(['category', 'strategy_a', 'strategy_b']):
        for metric in ovmetrics:
            values = group[metric].to_numpy()
            ovrows.append(dict(zip(['category', 'strategy_a', 'strategy_b'], keys), metric=metric,
                value=values.mean(), repeats=len(values),
                empirical_lower=np.quantile(values, .025) if len(values) > 1 else np.nan,
                empirical_upper=np.quantile(values, .975) if len(values) > 1 else np.nan))
    save_csv(pd.DataFrame(ovrows), out / 'jaccard_summary.csv')
    write_report(out, summary, pd.DataFrame(percentiles))
    log('Summaries, empirical intervals, percentiles and Jaccard comparisons saved')


def write_report(out, summary, percentiles):
    manifest = json.loads((out / 'discovery_manifest.json').read_text(encoding='utf-8'))
    rows = ['# 药材池策略比较：冻结候选与时间外评价', '',
        '本实验为药材池消融：四种策略共用 ULTRA 基础方评分，比较筛选药材池带来的变化。', '',
        '## 预先固定的方法', '',
        '- 模型为现有 ULTRA-50G-targeted；词表为六类共同的完整 489 味药材。',
        '- 母表仅从各类 train.json 穷尽生成无序、不重复的 3–4 味组合；每张处方贡献一次支持，类别内支持次数 ≥5。',
        '- ULTRA-400：类别内 ULTRA 前400。Frequency-400：类别内建模处方药材出现次数前400，同频按药名升序；零频药材可补齐400。',
        f'- Random-400：每类从489味均匀无放回抽400味，重复{manifest["random_repeats"]}次，seed={manifest["seed"]}；各类随机流独立，同一重复编号汇总六类macro。',
        '- All-489：完整词表，覆盖全部母表。所有策略均从母表过滤，只保留药味全部在池内的方。',
        '- 统一评分：sqrt(mean(1-(ULTRA完整排名-1)/489) × log(1+建模支持次数)/log(1+该类母表最大支持次数))。',
        '- 排序依次按综合分降序、支持次数降序、方大小升序、规范化药名串升序，取30方；三味和四味合并排名。',
        '- 原 basic_discover.py 会按候选池大小归一化排名；本实验固定分母489，确保不同池使用相同正值评分。因此本实验不是原Top-400排序的逐行复现，也不是各策略独立排名能力的比较。',
        '- discovery 与 evaluate 分进程执行；所有母表、药材池和Top-30在访问test.json前落盘并保存SHA-256。评价前核验输入和候选文件哈希。', '',
        '## 时间外指标与随机汇总', '',
        '- 使用 basic_discover.validate 原有指标计算代码；仅将集合交集替换为等价整数位集加速，并逐类与原函数100个样本作完全一致性检查。评价覆盖母表所有候选，因而覆盖所有策略候选。',
        '- 出现率为该类别时间外处方包含全部方内药味的比例；类别富集=(类别出现率+1e-8)/(六类合并出现率+1e-8)。',
        '- 沿用现有分母：六类处方成员记录相加，不跨类别按门诊号去重；故这是按类别成员记录加权的富集。全六类都不出现时，原EPS定义给出富集1。',
        '- 每个类别/策略/重复对Top-30分别计算出现率均值、中位数、富集倍数均值、中位数；macro为六个类别对应统计量的等权均值（中位数的macro不是180方的合并中位数）。',
        '- Random-400先逐次计算类别指标及macro，再报告500次的均值和2.5%/97.5%分位数（线性插值）；这是随机药材池经验区间，不是患者抽样置信区间。',
        '- ULTRA经验百分位=100×(随机值低于ULTRA的次数+0.5×相等次数)/500；相等容差1e-12，同时保存严格/弱百分位以及低于、相等、高于次数。',
        '- Jaccard=交集大小/并集大小；药材池按药名集合，基础方按无序药味集合。Random与其他各策略逐次配对，macro先按重复取六类均值再算区间。', '',
        '## 六类与macro结果', '',
        '出现率以百分数表示，富集为倍数；Random行括号为95%经验区间。', '',
        '| 类别 | 策略 | 平均出现率 % | 中位出现率 % | 平均富集 | 中位富集 |',
        '|---|---|---:|---:|---:|---:|']
    for cat in list(CATEGORIES) + ['MACRO']:
        for strategy in ('ULTRA-400', 'Frequency-400', 'Random-400', 'All-489'):
            sub = summary[summary.category.eq(cat) & summary.strategy.eq(strategy)].set_index('metric')
            cells = []
            for metric in METRICS:
                r = sub.loc[metric]
                scale = 100 if 'support' in metric else 1
                cell = f'{r.value * scale:.3f}'
                if strategy == 'Random-400':
                    cell += f' [{r.empirical_lower * scale:.3f}, {r.empirical_upper * scale:.3f}]'
                cells.append(cell)
            rows.append('| ' + ' | '.join([CATEGORIES.get(cat, cat), strategy] + cells) + ' |')
    rows += ['', '## ULTRA 相对随机分布的经验百分位', '',
             '| 类别 | 平均出现率 | 中位出现率 | 平均富集 | 中位富集 |', '|---|---:|---:|---:|---:|']
    for cat in list(CATEGORIES) + ['MACRO']:
        sub = percentiles[percentiles.category.eq(cat)].set_index('metric')
        rows.append('| ' + ' | '.join([CATEGORIES.get(cat, cat)] +
            [f'{sub.loc[m, "ultra_empirical_percentile"]:.1f}' for m in METRICS]) + ' |')
    rows += ['', '## 文件与复现', '',
        '- mother/*.csv.gz：每类完整基础方母表、训练支持次数、统一评分。',
        '- frozen/top30_selection.csv.gz：读取时间外数据之前冻结的全部90,540条选择记录。',
        '- frozen/herb_pools.npz：六类×503池×489味布尔矩阵，含药名、类别、策略、重复编号。',
        '- frozen/herb_frequencies_and_ranks.csv：建模频次及ULTRA排名；candidate_inventory.csv：候选数量和母表覆盖率。',
        '- temporal/*_all_candidates.csv.gz：全部母表候选的时间外评价；top30_evaluated.csv.gz：全部策略及重复的Top-30评价。',
        '- metric_summary.csv、metrics_by_repeat.csv、ultra_random_percentiles.csv：完整汇总和随机分布。',
        '- jaccard_summary.csv、jaccard_by_repeat.csv：六类及macro的策略间重叠度。',
        '```powershell',
        'python scripts/compare_formula_discovery_strategies.py discover',
        'python scripts/compare_formula_discovery_strategies.py evaluate',
        '```', '',
        '已冻结输出禁止原地重跑发现；需用 --output-dir 指定新目录。分析不修改原数据、已有评价结果或投稿图表。']
    (out / 'REPORT.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['discover', 'evaluate', 'summarize'])
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--score-dir', type=Path, default=ROOT / 'outputs/result_v1/selected_scores')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--repeats', type=int, default=500)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be at least 2')
    if args.phase == 'discover':
        # Runtime guard makes accidental temporal reads during discovery fail.
        def training_only(data_dir, category, split):
            if split != 'train':
                raise RuntimeError('Temporal access is forbidden during discovery')
            return load_prescriptions(data_dir, category, split)
        with patch.object(bd, 'load_prescriptions', training_only):
            discover(args)
    elif args.phase == 'evaluate':
        evaluate(args)
    else:
        summarize(args.output_dir)


if __name__ == '__main__':
    main()
