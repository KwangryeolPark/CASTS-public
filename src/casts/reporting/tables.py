from __future__ import annotations

from pathlib import Path
import math
import warnings
import numpy as np
import pandas as pd

METHOD_MAP = {
    'Chronological 60/20/20': 'Chronological 60/20/20',
    'Random point stratified 60/20/20': 'Point-level stratified random',
    'Random stratified bins': 'Bin-level stratified random',
    'CASTS-main': 'CASTS-05',
    'CASTS-lambda-0.05': 'CASTS-05',
    'CASTS-lambda-0.35': 'CASTS-35',
}
METHOD_ORDER = ['Chronological 60/20/20','Point-level stratified random','Bin-level stratified random','CASTS-05','CASTS-35']
DETECTOR_METHODS = ['Chronological 60/20/20','Bin-level stratified random','CASTS-05','CASTS-35']
MODEL_ORDER = ['CNN1D','GRU','LSTM']
FAMILY_ORDER = ['NAB','SKAB','SMD','SMAP','MSL']


def method_display(name: str) -> str:
    if name.startswith('CASTS-lambda-'):
        value = name.split('CASTS-lambda-')[1]
        return f'CASTS-{float(value):g}' if value not in {'0.05','0.35'} else {'0.05':'CASTS-05','0.35':'CASTS-35'}[value]
    return METHOD_MAP.get(name, name)


def write_csv_and_tex(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f'{name}.csv', index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        latex = df.to_latex(index=False, float_format=lambda x: f'{x:.3f}')
    (output_dir / f'{name}.tex').write_text(latex, encoding='utf-8')


def dataset_manifest(results_root: Path) -> pd.DataFrame:
    inv = pd.read_csv(results_root / 'raw' / 'dataset_inventory.csv')
    columns = [
        'dataset_id',
        'benchmark',
        'source_file',
        'length',
        'n_features',
        'anomaly_count',
        'anomaly_rate',
        'included',
        'exclusion_reason',
    ]
    missing = set(columns) - set(inv.columns)
    if missing:
        raise ValueError(
            f'dataset inventory is missing columns: {sorted(missing)}'
        )
    return (
        inv[columns]
        .sort_values(['benchmark', 'dataset_id'])
        .reset_index(drop=True)
    )


def table1(results_root: Path) -> pd.DataFrame:
    data = dataset_manifest(results_root)
    rows = []
    for fam in FAMILY_ORDER:
        g = data[data['benchmark'] == fam]
        rows.append({
            'Benchmark': fam,
            'Series': int(len(g)),
            'Median length': int(math.floor(float(g['length'].median()) + 0.5)),
            'Features': int(math.floor(float(g['n_features'].median()) + 0.5)),
            'Median anomaly rate': round(float(g['anomaly_rate'].median()), 3),
        })
    rows.append({'Benchmark':'Total','Series':int(len(data)),'Median length':'-','Features':'-','Median anomaly rate':'-'})
    return pd.DataFrame(rows)


def split_method_summaries(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(
        results_root
        / "summaries"
        / "generated_split_method_summary.csv"
    )

    summary = summary[
        summary["method"].isin(METHOD_ORDER)
    ].copy()

    summary["method"] = pd.Categorical(
        summary["method"],
        METHOD_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("method")

    table2 = summary.rename(
        columns={
            "method": "Method",
            "mean_anomaly_difference":
                "Mean anomaly difference",
            "mean_contiguous_count":
                "Mean contiguous count",
        }
    )[
        [
            "Method",
            "Mean anomaly difference",
            "Mean contiguous count",
            "split_rows",
            "specs",
        ]
    ]

    table3 = summary.rename(
        columns={
            "method": "Method",
            "both_class_rate": "Both-class rate",
            "crossing_rate": "Crossing rate",
        }
    )[
        [
            "Method",
            "Both-class rate",
            "Crossing rate",
            "split_rows",
            "specs",
        ]
    ]

    return table2, table3

def lambda_summary(results_root: Path) -> pd.DataFrame:
    fam = pd.read_csv(results_root / 'raw' / 'lambda_summary_by_family.csv')
    rows = []
    for lam, g in fam.groupby('lambda_cont'):
        rows.append({
            'lambda_cont': float(lam),
            'Anomaly difference': np.average(g['mean_label_rate_error'], weights=g['split_rows']),
            'Contiguous count': np.average(g['mean_segments'], weights=g['split_rows']),
            'split_rows': int(g['split_rows'].sum()),
            'specs': int(g['n_specs'].sum()),
        })
    return pd.DataFrame(rows).sort_values('lambda_cont').reset_index(drop=True)


def family_tables(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fam = pd.read_csv(results_root / 'raw' / 'lambda_summary_by_family.csv')
    laudit = pd.read_csv(results_root / 'raw' / 'lambda_audit_by_split.csv')
    keep = fam[fam['lambda_cont'].isin([0.05, 0.35])].copy()
    keep['Config'] = keep['lambda_cont'].map({0.05:'CASTS-05',0.35:'CASTS-35'})
    both = laudit[laudit['lambda_cont'].isin([0.05,0.35])].groupby(['family','lambda_cont'], as_index=False).agg(
        **{'Both-class rate':('split_has_both_classes','mean'), 'Crossing rate':('has_any_episode_leakage','mean')}
    )
    merged = keep.merge(both, on=['family','lambda_cont'], how='left')
    merged['family'] = pd.Categorical(merged['family'], FAMILY_ORDER, ordered=True)
    merged = merged.sort_values(['family','lambda_cont'])
    t9 = merged.rename(columns={'family':'Family','mean_label_rate_error':'Anomaly difference','mean_segments':'Contiguous count'})[['Family','Config','Anomaly difference','Contiguous count','Both-class rate','Crossing rate','split_rows','n_specs']]
    t12 = t9[['Family','Config','Both-class rate','Crossing rate','split_rows','n_specs']].copy()
    pivot_anom = fam.pivot(index='lambda_cont', columns='family', values='mean_label_rate_error').reset_index().rename(columns={'lambda_cont':'lambda'})
    pivot_cont = fam.pivot(index='lambda_cont', columns='family', values='mean_segments').reset_index().rename(columns={'lambda_cont':'lambda'})
    pivot_anom = pivot_anom[['lambda'] + FAMILY_ORDER]
    pivot_cont = pivot_cont[['lambda'] + FAMILY_ORDER]
    return t9, t12, pivot_anom, pivot_cont, fam


def common_77(results_root: Path) -> pd.DataFrame:
    df = pd.read_csv(results_root / 'raw' / 'detector_model_metrics_summary.csv')
    df['method_public'] = df['method'].map(method_display)
    methods = DETECTOR_METHODS
    models = MODEL_ORDER
    d = df[df['method_public'].isin(methods) & df['model'].isin(models)].copy()
    valid = d[np.isfinite(d['mean_auroc']) & np.isfinite(d['mean_average_precision'])]
    counts = valid.groupby('dataset').apply(lambda g: len(set(zip(g['method_public'], g['model'])))).reset_index(name='valid_combinations')
    common = counts[counts['valid_combinations'] == len(methods) * len(models)]['dataset'].sort_values().tolist()
    # meta = d[d['dataset'].isin(common)].groupby('dataset', as_index=False).agg(source=('source','first'))
    # meta['valid_combinations'] = len(methods) * len(models)
    meta = pd.DataFrame({'dataset': common})
    meta['valid_combinations'] = len(methods) * len(models)
    return meta.sort_values('dataset').reset_index(drop=True)


def detector_summaries(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(results_root / 'raw' / 'detector_model_metrics_summary.csv')
    df['Method'] = df['method'].map(method_display)
    common = set(common_77(results_root)['dataset'])
    d = df[df['dataset'].isin(common) & df['Method'].isin(DETECTOR_METHODS) & df['model'].isin(MODEL_ORDER)].copy()
    method = d.groupby('Method', as_index=False).agg(
        BAcc=('mean_balanced_accuracy','mean'), F1=('mean_f1','mean'), AUROC=('mean_auroc','mean'), AP=('mean_average_precision','mean'),
        datasets=('dataset','nunique'), models=('model','nunique'), rows=('dataset','size')
    ).set_index('Method').loc[DETECTOR_METHODS].reset_index()
    by_model = d.groupby(['model','Method'], as_index=False).agg(
        BAcc=('mean_balanced_accuracy','mean'), F1=('mean_f1','mean'), AUROC=('mean_auroc','mean'), AP=('mean_average_precision','mean'), datasets=('dataset','nunique'), rows=('dataset','size')
    )
    by_model['model'] = pd.Categorical(by_model['model'], MODEL_ORDER, ordered=True)
    by_model['Method'] = pd.Categorical(by_model['Method'], DETECTOR_METHODS, ordered=True)
    by_model = by_model.sort_values(['model','Method']).reset_index(drop=True).rename(columns={'model':'Model'})
    return method, by_model

def generated_lambda_audit(results_root: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        results_root / 'raw' / 'generated_lambda_audit_by_split.csv'
    )
    raw = raw[raw['method'].astype(str).str.startswith('CASTS')].copy()

    required = {
        'dataset',
        'benchmark',
        'method',
        'lambda_cont',
        'seed',
        'replicate',
        'anomaly_proportion_difference',
        'contiguous_count',
        'both_class',
        'has_any_episode_crossing',
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f'generated lambda audit is missing columns: {sorted(missing)}'
        )
    return raw


def lambda_summary(results_root: Path) -> pd.DataFrame:
    raw = generated_lambda_audit(results_root)

    summary = (
        raw.groupby('lambda_cont', as_index=False)
        .agg(
            **{
                'Anomaly difference': (
                    'anomaly_proportion_difference',
                    'mean',
                ),
                'Contiguous count': ('contiguous_count', 'mean'),
                'split_rows': ('dataset', 'size'),
            }
        )
        .sort_values('lambda_cont')
        .reset_index(drop=True)
    )

    specs = (
        raw[
            ['dataset', 'lambda_cont', 'seed', 'replicate']
        ]
        .drop_duplicates()
        .groupby('lambda_cont', as_index=False)
        .size()
        .rename(columns={'size': 'specs'})
    )

    return summary.merge(specs, on='lambda_cont', how='left')


def lambda_family_summary(results_root: Path) -> pd.DataFrame:
    raw = generated_lambda_audit(results_root)

    family = (
        raw.groupby(['benchmark', 'lambda_cont'], as_index=False)
        .agg(
            **{
                'Anomaly difference': (
                    'anomaly_proportion_difference',
                    'mean',
                ),
                'Contiguous count': ('contiguous_count', 'mean'),
                'Both-class rate': ('both_class', 'mean'),
                'split_rows': ('dataset', 'size'),
            }
        )
    )

    spec_columns = [
        'dataset',
        'benchmark',
        'lambda_cont',
        'seed',
        'replicate',
        'has_any_episode_crossing',
    ]
    specs = raw[spec_columns].drop_duplicates()

    spec_summary = (
        specs.groupby(['benchmark', 'lambda_cont'], as_index=False)
        .agg(
            **{
                'Crossing rate': (
                    'has_any_episode_crossing',
                    'mean',
                ),
                'n_specs': ('dataset', 'size'),
            }
        )
    )

    return family.merge(
        spec_summary,
        on=['benchmark', 'lambda_cont'],
        how='left',
    )


def family_tables(
    results_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fam = lambda_family_summary(results_root)

    keep = fam[fam['lambda_cont'].isin([0.05, 0.35])].copy()
    keep['Config'] = keep['lambda_cont'].map(
        {0.05: 'CASTS-05', 0.35: 'CASTS-35'}
    )
    keep['benchmark'] = pd.Categorical(
        keep['benchmark'],
        FAMILY_ORDER,
        ordered=True,
    )
    keep = keep.sort_values(['benchmark', 'lambda_cont'])

    t9 = keep.rename(columns={'benchmark': 'Family'})[
        [
            'Family',
            'Config',
            'Anomaly difference',
            'Contiguous count',
            'Both-class rate',
            'Crossing rate',
            'split_rows',
            'n_specs',
        ]
    ]
    t12 = t9[
        [
            'Family',
            'Config',
            'Both-class rate',
            'Crossing rate',
            'split_rows',
            'n_specs',
        ]
    ].copy()

    t10 = (
        fam.pivot(
            index='lambda_cont',
            columns='benchmark',
            values='Anomaly difference',
        )
        .reset_index()
        .rename(columns={'lambda_cont': 'lambda'})
    )
    t11 = (
        fam.pivot(
            index='lambda_cont',
            columns='benchmark',
            values='Contiguous count',
        )
        .reset_index()
        .rename(columns={'lambda_cont': 'lambda'})
    )

    t10 = t10[['lambda'] + FAMILY_ORDER]
    t11 = t11[['lambda'] + FAMILY_ORDER]

    return t9, t12, t10, t11, fam