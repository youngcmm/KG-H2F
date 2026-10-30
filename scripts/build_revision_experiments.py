"""CSV-first revision experiments: training-only selection, cohort table, full Top-30."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from category import CATEGORIES, DEFAULT_DATA_DIR, load_prescriptions, load_scores, split_herbs
from basic_discover import wilson_ci
from compare_formula_discovery_strategies import enumerate_mother, bit_tidsets, bit_count, sha, save_csv, save_json

OUT = ROOT / 'outputs/revision_experiments'
SOURCE = ROOT / 'submission/source_data'
SCORES = ROOT / 'outputs/result_v1/selected_scores'
STRATEGIES = ['Composite', 'Support-only', 'Support-same-pool']


def export(frame, name):
    save_csv(frame, OUT / f'{name}.csv')
    save_csv(frame, SOURCE / f'revision_{name}.csv')


def discover():
    frozen = OUT / 'formula_selection.csv'
    if frozen.exists():
        raise FileExistsError('Selection already frozen; do not overwrite.')
    records = {c: load_prescriptions(DEFAULT_DATA_DIR,c,'train') for c in CATEGORIES}
    names = sorted(set().union(*(set().union(*rs) for rs in records.values())))
    selections, inventories = [], []
    hashes = {str(DEFAULT_DATA_DIR / c / 'train.json'): sha(DEFAULT_DATA_DIR / c / 'train.json') for c in CATEGORIES}
    # Pure baseline never reads scores; candidate universe is observed training herbs.
    mothers = {}
    for cat, rs in records.items():
        print(f'{cat}: enumerate training-only formulas',flush=True)
        ids, counts = enumerate_mother(rs,names)
        strings = ['|'.join(names[i] for i in row if i>=0) for row in ids]
        master = pd.DataFrame(dict(category=cat,base_formula=strings,formula_size=(ids>=0).sum(axis=1),
                                   train_count=counts,train_total=len(rs),train_support=counts/len(rs)))
        mothers[cat] = master
        chosen = master.sort_values(['train_count','formula_size','base_formula'],ascending=[False,True,True]).head(30).copy()
        chosen['strategy']='Support-only'; chosen['rank']=np.arange(1,31)
        chosen['selection_score']=chosen.train_count
        selections.append(chosen)
        inventories.append(dict(category=cat,all_training_candidates=len(master),training_herbs=len(set().union(*rs))))
    pure = pd.concat(selections,ignore_index=True)
    save_csv(pure,OUT/'support_only_frozen.csv')
    pure_hash=sha(OUT/'support_only_frozen.csv')
    # Reconstruct exactly the existing Top-400 composite, normalization denominator 400.
    for cat, master in mothers.items():
        scores=load_scores(SCORES,cat).head(400)
        ranks=dict(zip(scores.herb_name,scores.model_rank))
        hashes[str(SCORES/f'comorbidity_{cat}_selected_herb_scores.csv')]=sha(SCORES/f'comorbidity_{cat}_selected_herb_scores.csv')
        pool=set(ranks)
        eligible=master[master.base_formula.map(lambda f:set(f.split('|'))<=pool)].copy()
        eligible['original_formula']=eligible.base_formula.map(lambda f:'|'.join(sorted(f.split('|'),key=ranks.get)))
        eligible['selection_score']=eligible.base_formula.map(lambda f:np.mean([1-(ranks[h]-1)/400 for h in f.split('|')]))
        eligible['selection_score']=np.sqrt(eligible.selection_score*np.log1p(eligible.train_count)/np.log1p(eligible.train_count.max()))
        composite=eligible.sort_values(['selection_score','train_count','formula_size','original_formula'],ascending=[False,False,True,True]).head(30).copy()
        composite['strategy']='Composite'; composite['rank']=np.arange(1,31); selections.append(composite)
        same=eligible.sort_values(['train_count','formula_size','base_formula'],ascending=[False,True,True]).head(30).copy()
        same['strategy']='Support-same-pool'; same['rank']=np.arange(1,31); same['selection_score']=same.train_count; selections.append(same)
        next(r for r in inventories if r['category']==cat)['model_pool_candidates']=len(eligible)
    export(pd.concat(selections,ignore_index=True),'formula_selection')
    export(pd.DataFrame(inventories),'selection_inventory')
    save_json(OUT/'selection_manifest.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        training_only=True,temporal_data_accessed=False,observed_training_vocabulary=len(names),
        support_only_definition='All training-observed herbs; 3-4 herbs; support >=5; sort count desc, size asc, canonical name asc; no scores or model pool',
        composite_definition='Original basic_discover.py model Top-400; rank normalization 400; original rank-ordered name tie breaker',
        support_same_pool_definition='Secondary ranking ablation in model Top-400; not a model-free baseline',
        input_sha256=hashes,selection_sha256=sha(frozen),pure_baseline_sha256=pure_hash,code_sha256=sha(Path(__file__))))


MISSING={'','none','null','nan','na','n/a','未知','不详','未填写','未记录','缺失'}
def missing(v):
    return v is None or str(v).strip().lower() in MISSING


def tokens(v):
    if missing(v): return set()
    return {t.strip() for t in re.split(r'[,，;；、|]+',str(v)) if not missing(t) and t.strip()!='无'}


def age_value(v):
    if missing(v): return np.nan
    s=str(v).strip()
    if re.fullmatch(r'\d+(?:\.\d+)?\s*(?:岁|周岁)?',s):
        a=float(re.search(r'\d+(?:\.\d+)?',s)[0]); return a if 0<=a<=120 else np.nan
    if re.fullmatch(r'\d+(?:\.\d+)?\s*(?:月|个月)',s):
        return float(re.search(r'\d+(?:\.\d+)?',s)[0])/12
    return np.nan


def cohorts():
    visits={}; hashes={}; field_names=set()
    for c in CATEGORIES:
        for split in ['train','test']:
            p=DEFAULT_DATA_DIR/c/f'{split}.json'; hashes[str(p)]=sha(p)
            rows=json.loads(p.read_text(encoding='utf-8-sig')); seen=set()
            for r in rows:
                field_names.update(r)
                key=str(r['门诊号']).strip()
                assert key and key not in seen; seen.add(key)
                if key in visits:
                    assert visits[key]['split']==split
                    for f in ['年龄','西医诊断','四诊对应证候_标准名称','标准处方药材名称','就诊时间']:
                        assert visits[key]['row'].get(f)==r.get(f),f'Conflicting {f}'
                    visits[key]['cats'].add(c)
                else: visits[key]=dict(split=split,row=r,cats={c})
    sex_fields=[f for f in field_names if f.lower() in ['性别','患者性别','sex','gender']]
    fields={'age':'年龄','visit_date':'就诊时间','diagnosis':'西医诊断','tcm_diagnosis':'中医诊断',
            'syndrome':'四诊对应证候_标准名称','tongue_pulse':'舌苔脉象_标准名称',
            'prescription':'标准处方药材名称','patient_id':'患者个人id'}
    if sex_fields: fields['sex']=sex_fields[0]
    summary=[]; terms=[]; missing_rows=[]; sizes=[]; age_bins=[]
    for scope in ['overall',*CATEGORIES]:
        for split in ['train','test']:
            group=[v for v in visits.values() if v['split']==split and (scope=='overall' or scope in v['cats'])]
            rs=[v['row'] for v in group]; n=len(rs)
            ages=np.array([age_value(r.get('年龄')) for r in rs]); a=ages[np.isfinite(ages)]
            herb_sizes=np.array([len(split_herbs(r.get('标准处方药材名称'))) for r in rs]); valid_sizes=herb_sizes[herb_sizes>0]
            dates=pd.to_datetime(pd.Series([r.get('就诊时间') for r in rs]),errors='coerce')
            patients={str(r['患者个人id']) for r in rs if not missing(r.get('患者个人id'))}
            row=dict(scope=scope,split=split,n_visits=n,n_distinct_patient_ids=len(patients),
                category_memberships=sum(len(v['cats']) for v in group) if scope=='overall' else n,
                multilabel_visits=sum(len(v['cats'])>1 for v in group),
                date_min=str(dates.min().date()),date_max=str(dates.max().date()),
                age_n=len(a),age_mean=a.mean() if len(a) else np.nan,age_sd=a.std(ddof=1) if len(a)>1 else np.nan,
                age_median=np.median(a) if len(a) else np.nan,age_q1=np.quantile(a,.25) if len(a) else np.nan,
                age_q3=np.quantile(a,.75) if len(a) else np.nan,herb_n=len(valid_sizes),
                herb_mean=valid_sizes.mean(),herb_sd=valid_sizes.std(ddof=1),herb_median=np.median(valid_sizes),
                herb_q1=np.quantile(valid_sizes,.25),herb_q3=np.quantile(valid_sizes,.75),sex_available=bool(sex_fields))
            summary.append(row)
            for field,key in fields.items():
                raw_missing=sum(missing(r.get(key)) for r in rs)
                usable=sum(np.isfinite(age_value(r.get(key))) if field=='age' else
                    not pd.isna(pd.to_datetime(r.get(key),errors='coerce')) if field=='visit_date' else
                    bool(tokens(r.get(key))) if field in ['diagnosis','syndrome','tongue_pulse','prescription','tcm_diagnosis'] else
                    not missing(r.get(key)) for r in rs)
                missing_rows.append(dict(scope=scope,split=split,field=field,source_field=key,n=n,
                    field_available=True,raw_missing_n=raw_missing,unusable_n=n-usable,missing_percent=100*(n-usable)/n))
            if not sex_fields:
                missing_rows.append(dict(scope=scope,split=split,field='sex',source_field='',n=n,
                    field_available=False,raw_missing_n=n,unusable_n=n,missing_percent=100.))
            for field,key in [('diagnosis','西医诊断'),('syndrome','四诊对应证候_标准名称')]:
                counts=Counter(t for r in rs for t in tokens(r.get(key)))
                for rank,(term,count) in enumerate(sorted(counts.items(),key=lambda x:(-x[1],x[0])),1):
                    terms.append(dict(scope=scope,split=split,field=field,term=term,rank=rank,count=count,n=n,percent=100*count/n))
            for size,count in sorted(Counter(herb_sizes).items()):
                sizes.append(dict(scope=scope,split=split,herb_count=size,count=count,n=n,percent=100*count/n))
            for lo,hi in [(0,18),(18,40),(40,60),(60,80),(80,121)]:
                count=int(((ages>=lo)&(ages<hi)).sum())
                age_bins.append(dict(scope=scope,split=split,age_band=f'{lo}-{hi-1}',count=count,n=n,percent=100*count/n))
    for name,rows in [('cohort_summary',summary),('cohort_terms',terms),('cohort_missingness',missing_rows),('cohort_herb_counts',sizes),('cohort_age_bands',age_bins)]:
        export(pd.DataFrame(rows),name)
    term_frame=pd.DataFrame(terms)
    for field in ['diagnosis','syndrome']:
        top=term_frame[(term_frame.scope=='overall')&(term_frame.split=='train')&
            (term_frame.field==field)&(term_frame.term!='睡眠障碍')].sort_values('rank').head(7)
        labels=pd.DataFrame(dict(field=field,id=[f'{"D" if field=="diagnosis" else "P"}{i+1}' for i in range(len(top))],term=top.term.to_list()))
        export(labels,f'{field}_figure_labels')
    save_json(OUT/'cohort_manifest.json',dict(input_sha256=hashes,fields=sorted(field_names),sex_fields=sex_fields,
        unit='unique visit ID overall; one visit per category within strata; category labels overlap',
        age_parser='numeric years, optional 岁/周岁; numeric months converted to years; valid range 0-120; unparseable excluded',
        missingness='empty/null markers, empty token sets, invalid ages/dates; absent sex distinguished from field-level missingness'))
    make_cohort_table()


def make_cohort_table():
    s=pd.read_csv(OUT/'cohort_summary.csv'); t=pd.read_csv(OUT/'cohort_terms.csv'); m=pd.read_csv(OUT/'cohort_missingness.csv')
    rows=[]
    for scope in ['overall',*CATEGORIES]:
        group=s[s.scope.eq(scope)].set_index('split')
        def add(label,fn):
            rows.append(dict(scope=scope,characteristic=label,modeling=fn('train'),temporal_holdout=fn('test')))
        add('就诊数',lambda p:f'{int(group.loc[p,"n_visits"]):,}')
        add('实际就诊日期',lambda p:f'{group.loc[p,"date_min"]} — {group.loc[p,"date_max"]}')
        if scope=='overall':
            add('类别归属记录数（含跨类重复）',lambda p:f'{int(group.loc[p,"category_memberships"]):,}')
            for cat,cn in CATEGORIES.items():
                add(f'{cn}，n (%)',lambda p,c=cat:f'{int(s[(s.scope==c)&(s.split==p)].n_visits.iloc[0]):,} ({100*s[(s.scope==c)&(s.split==p)].n_visits.iloc[0]/group.loc[p,"n_visits"]:.2f}%)')
        add('可用年龄，n',lambda p:str(int(group.loc[p,'age_n'])))
        add('年龄，均值 ± SD（岁）',lambda p:f'{group.loc[p,"age_mean"]:.2f} ± {group.loc[p,"age_sd"]:.2f}')
        add('年龄，中位数 [Q1, Q3]',lambda p:f'{group.loc[p,"age_median"]:.1f} [{group.loc[p,"age_q1"]:.1f}, {group.loc[p,"age_q3"]:.1f}]')
        add('性别',lambda p:'源数据未提供' if not group.loc[p,'sex_available'] else '见字段分布')
        add('处方药味数，中位数 [Q1, Q3]',lambda p:f'{group.loc[p,"herb_median"]:.0f} [{group.loc[p,"herb_q1"]:.0f}, {group.loc[p,"herb_q3"]:.0f}]')
        add('处方药味数，均值 ± SD',lambda p:f'{group.loc[p,"herb_mean"]:.2f} ± {group.loc[p,"herb_sd"]:.2f}')
        for field,title in [('diagnosis','主要诊断'),('syndrome','证候')]:
            pool=t[t.scope.eq(scope)&t.field.eq(field)&t.term.ne('睡眠障碍')]
            # Display union of top five in each period; retain all terms in source CSV.
            chosen=set(pool.sort_values('rank').groupby('split').head(5).term)
            for term in sorted(chosen):
                def term_cell(p):
                    r=pool[pool.split.eq(p)&pool.term.eq(term)]
                    count=int(r['count'].iloc[0]) if len(r) else 0
                    return f'{count:,} ({100*count/group.loc[p,"n_visits"]:.2f}%)'
                add(f'{title}：{term}，n (%)',term_cell)
        field_labels=dict(age='年龄',visit_date='就诊时间',diagnosis='西医诊断',tcm_diagnosis='中医诊断',
            syndrome='标准证候',tongue_pulse='标准舌脉',prescription='标准处方',patient_id='患者标识',sex='性别（未提供）')
        for f in m.field.unique():
            add(f'关键字段不可用率：{field_labels[f]}',lambda p:f'{m[(m.scope==scope)&(m.split==p)&(m.field==f)].missing_percent.iloc[0]:.2f}%')
    export(pd.DataFrame(rows),'cohort_table')


def evaluate():
    manifest=json.loads((OUT/'selection_manifest.json').read_text(encoding='utf-8'))
    assert sha(OUT/'formula_selection.csv')==manifest['selection_sha256']
    for p,h in manifest['input_sha256'].items(): assert sha(p)==h
    selected=pd.read_csv(OUT/'formula_selection.csv')
    old=pd.read_csv(ROOT/'outputs/result_v1/temporal_validation/candidate_base_formulas_temporal_validation.csv')
    old=old[old['rank']<=30].copy(); old['canonical']=old.base_formula.map(lambda f:'|'.join(sorted(f.split('|'))))
    original=selected[selected.strategy.eq('Composite')].sort_values(['category','rank'])
    reference=old.sort_values(['category','rank'])
    assert original.base_formula.tolist()==reference.canonical.tolist(),'Original Top-30 selection mismatch'
    tests={c:load_prescriptions(DEFAULT_DATA_DIR,c,'test') for c in CATEGORIES}
    tids={c:bit_tidsets(rs) for c,rs in tests.items()}; totals={c:len(rs) for c,rs in tests.items()}
    results=[]
    for r in selected.to_dict('records'):
        f=r['base_formula'].split('|'); counts={c:bit_count(f,ts) for c,ts in tids.items()}
        n=totals[r['category']]; k=counts[r['category']]; rate=k/n; overall=sum(counts.values())/sum(totals.values())
        lo,hi=wilson_ci(k,n)
        results.append(dict(r,test_count=k,test_total=n,test_support=rate,test_support_ci_lower=lo,
            test_support_ci_upper=hi,formula_enrichment=(rate+1e-8)/(overall+1e-8),
            overall_membership_count=sum(counts.values()),overall_membership_total=sum(totals.values())))
    result=pd.DataFrame(results)
    current=result[result.strategy.eq('Composite')].sort_values(['category','rank'])
    for col in ['test_count','test_total','test_support','formula_enrichment']:
        np.testing.assert_allclose(current[col],reference[col],rtol=1e-10,atol=1e-12)
    export(result,'formula_top30_evaluated')
    full=old.copy(); full['base_formula']=full.pop('canonical')
    full['candidate_id']=[f'F{list(CATEGORIES).index(c)+1}-{r:02}' for c,r in zip(full.category,full['rank'])]
    export(full,'composite_top30_complete')
    stats=[]
    for (cat,strategy),g in result.groupby(['category','strategy']):
        row=dict(category=cat,strategy=strategy,n=len(g),n_occurs=int((g.test_count>0).sum()),
            n_rate_ge_5pct=int((g.test_support>=.05).sum()),n_rate_ci_ge_5pct=int((g.test_support_ci_lower>=.05).sum()),
            n_enrichment_gt1=int((g.formula_enrichment>1).sum()))
        for col in ['test_support','formula_enrichment']:
            for stat,fun in [('mean',np.mean),('median',np.median),('min',np.min),('q1',lambda a:np.quantile(a,.25)),('q3',lambda a:np.quantile(a,.75)),('max',np.max)]:
                row[f'{col}_{stat}']=fun(g[col])
        stats.append(row)
    stats=pd.DataFrame(stats); export(stats,'formula_summary')
    macro=stats.groupby('strategy').mean(numeric_only=True).reset_index(); export(macro,'formula_macro')
    overlap=[]
    for strategy in STRATEGIES:
        for a,b in combinations(CATEGORIES,2):
            aa=set(result[(result.strategy==strategy)&(result.category==a)].base_formula)
            bb=set(result[(result.strategy==strategy)&(result.category==b)].base_formula)
            overlap.append(dict(strategy=strategy,category_a=a,category_b=b,intersection=len(aa&bb),union=len(aa|bb),jaccard=len(aa&bb)/len(aa|bb)))
    export(pd.DataFrame(overlap),'cross_category_overlap')
    pairs=[]
    for c in CATEGORIES:
        for a,b in combinations(STRATEGIES,2):
            aa=set(result[(result.strategy==a)&(result.category==c)].base_formula)
            bb=set(result[(result.strategy==b)&(result.category==c)].base_formula)
            pairs.append(dict(category=c,strategy_a=a,strategy_b=b,intersection=len(aa&bb),union=len(aa|bb),jaccard=len(aa&bb)/len(aa|bb)))
    export(pd.DataFrame(pairs),'ranking_overlap')
    membership=full.groupby('base_formula').agg(category_count=('category','nunique'),categories=('category',lambda x:'|'.join(sorted(x)))).reset_index()
    export(membership,'formula_memberships')
    save_json(OUT/'evaluation_manifest.json',dict(selection_sha256=sha(OUT/'formula_selection.csv'),
        original_top30_exact_match=True,original_temporal_metrics_match=True,
        enrichment_denominator='sum of six category memberships, including shared visits; retained for comparability',
        uncertainty='Boxplots are candidate distributions, not training SD or confidence intervals. Original per-formula enrichment bootstrap intervals retained separately.',
        test_sha256={c:sha(DEFAULT_DATA_DIR/c/'test.json') for c in CATEGORIES}))


def stability():
    from run_revision_stability import module
    d=OUT/'stability'; evaluator=module('09_consolidate_model_comparison.py')
    rows=pd.read_csv(d/'kge_metrics_by_seed.csv').to_dict('records')
    positive=pd.read_csv(ROOT/'outputs/splits/positive_edges_with_split.csv')
    positive=positive[positive.edge_type.eq('category-herb')&positive.label.eq(1)]
    sampled={s:pd.read_csv(ROOT/f'outputs/splits/{s}_edges.csv') for s in ['valid','test']}
    for seed in [42,52,62]:
        folder=d/f'ultra_seed_{seed}'
        scores=pd.read_csv(folder/'03_入选模型六类药材完整排序.csv')
        scores=scores.rename(columns={'共病节点ID':'source','药材节点ID':'target','ULTRA原始分数':'score'})
        rows.extend(dict(r,seed=seed,run_kind='independent target adaptation') for r in evaluator.evaluate('ULTRA-50G-targeted',scores,positive,sampled))
    frame=pd.DataFrame(rows)
    assert frame.groupby(['model','split']).seed.nunique().eq(3).all()
    export(frame,'model_metrics_by_seed')
    metrics=['MRR','Hits@1','Hits@3','Hits@10','AUROC_sampled_1to5','AUPRC_sampled_1to5']
    stats=[]
    for (model,split),g in frame.groupby(['model','split']):
        for m in metrics:
            stats.append(dict(model=model,split=split,metric=m,n_runs=len(g),mean=g[m].mean(),sd=g[m].std(ddof=1),min=g[m].min(),max=g[m].max()))
    export(pd.DataFrame(stats),'model_stability_summary')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['discover','cohorts','evaluate','stability','make_cohort_table']);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);SOURCE.mkdir(parents=True,exist_ok=True)
    globals()[args.phase]()


if __name__=='__main__': main()
