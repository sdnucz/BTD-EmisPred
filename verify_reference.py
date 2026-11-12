"""Audit archived results and export reviewer figures without fitting models."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np
import pandas as pd
from data.dataset import load_data, TARGET
from frozen_partitions import load_frozen
from predict import predict_frame
from plot_review_results import validate, metrics, build

ROOT = Path(__file__).resolve().parent
ATOL = 1e-7


def read(path):
    return json.loads(Path(path).read_text())


def attach_records(frame, raw):
    if frame.record_id.isna().any() or not frame.record_id.isin(raw.record_id).all():
        raise ValueError('Unknown record ID in reference predictions.')
    matched = raw.set_index('record_id').loc[frame.record_id]
    np.testing.assert_array_equal(frame[TARGET], matched[TARGET])
    frame = frame.copy()
    frame['canonical_smiles'] = matched.canonical_smiles.to_numpy()
    return frame


def verify(output, figures=False, font_dir=None):
    if output.exists():
        raise FileExistsError(output)
    reference = ROOT / 'reference'
    for name, expected in read(reference / 'SHA256SUMS.json').items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Archived file checksum differs: {name}')
    data = ROOT / 'data/data/nir2_emission_dataset.csv'
    raw = load_data(data)
    config = read(ROOT / 'configs/paper.json')
    saved = load_frozen(ROOT / 'splits/Frozen_Partitions.json', data,
                        raw.canonical_smiles.tolist(), config)
    params = read(reference / 'Estimator_Parameters.json')
    if set(params) != set(saved['contexts']):
        raise ValueError('Candidate parameters and partition contexts differ.')
    names = set(config['search_spaces'])
    choices = pd.read_csv(reference / 'Outer_Algorithm_Selections.csv')
    oof = attach_records(pd.read_csv(reference / 'Procedure_Outer_OOF.csv'), raw)
    test = attach_records(pd.read_csv(reference / 'Independent_Test.csv'), raw)
    fitted = attach_records(pd.read_csv(reference / 'Training_Fit.csv'), raw)
    candidates = attach_records(pd.read_csv(reference / 'Eight_Algorithm_Outer_OOF.csv'), raw)
    final_inner = attach_records(pd.read_csv(reference / 'Final_Inner_OOF.csv'), raw)
    for frame, ids in [(oof, saved['holdout']['fit_record_ids']),
                       (test, saved['holdout']['valid_record_ids']),
                       (fitted, saved['holdout']['fit_record_ids'])]:
        if sorted(frame.record_id) != sorted(ids):
            raise ValueError('Prediction coverage differs from the frozen split.')
    validate(oof, test, candidates, choices)
    for i, context in enumerate(saved['outer_order'], 1):
        ids = saved['contexts'][context]['partition']['valid_record_ids']
        if sorted(oof.loc[oof.outer_fold.eq(i), 'record_id']) != sorted(ids):
            raise ValueError('OOF fold differs from the frozen assignment.')
    for context, group in params.items():
        if set(group['candidates']) != names:
            raise ValueError(f'Incomplete candidates: {context}')
        scores = {name: float(np.mean(row['inner_fold_rmse']))
                  for name, row in group['candidates'].items()}
        for name, row in group['candidates'].items():
            np.testing.assert_allclose(scores[name], row['mean_inner_RMSE'], rtol=0, atol=ATOL)
        selected = group['selected']
        np.testing.assert_allclose(scores[selected['algorithm']], min(scores.values()), rtol=0, atol=ATOL)
        if context != 'final':
            fold = saved['outer_order'].index(context) + 1
            row = choices.loc[choices.outer_fold.eq(fold)].iloc[0]
            if row.algorithm != selected['algorithm']:
                raise ValueError('Outer choice differs from the parameter archive.')
            np.testing.assert_allclose(row.mean_inner_RMSE, scores[row.algorithm], rtol=0, atol=ATOL)
    if final_inner.duplicated(['algorithm', 'record_id']).any() or set(final_inner.algorithm) != names:
        raise ValueError('Invalid final inner prediction keys.')
    for name, part in final_inner.groupby('algorithm'):
        for fold, split in enumerate(saved['contexts']['final']['inner'], 1):
            rows = part[part.inner_fold.eq(fold)]
            if sorted(rows.record_id) != sorted(split['valid_record_ids']):
                raise ValueError('Final inner-fold coverage differs.')
            np.testing.assert_allclose(metrics(rows)['RMSE'],
                params['final']['candidates'][name]['inner_fold_rmse'][fold-1], rtol=0, atol=ATOL)
    trials = pd.read_csv(reference / 'Optuna_Trials.csv')
    counts = trials.groupby(['context', 'algorithm']).size()
    expected = {(context, name) for context in params for name in names - {'GBR'}}
    if set(counts.index) != expected or not counts.eq(50).all():
        raise ValueError('Historical trial counts differ from seven algorithms x 50 trials.')
    model = ROOT / 'pretrained'
    predicted = predict_frame(raw, model, threads=1).set_index('record_id')
    differences = {}
    for name, frame in [('training_fit', fitted), ('independent_test', test)]:
        p = predicted.loc[frame.record_id, 'prediction_nm'].to_numpy()
        np.testing.assert_allclose(p, frame.prediction_nm, rtol=0, atol=ATOL)
        differences[name] = float(np.max(abs(p-frame.prediction_nm.to_numpy())))
    for name in ['External13.csv', 'SI6_Predictions.csv']:
        frame = pd.read_csv(reference / name)
        p = predict_frame(frame, model, threads=1).prediction_nm
        np.testing.assert_allclose(p, frame.prediction_nm, rtol=0, atol=ATOL)
        differences[name] = float(np.max(abs(p-frame.prediction_nm)))
    output.mkdir(parents=True)
    run = output / 'figure_inputs'
    (run / 'final').mkdir(parents=True)
    for frame, path in [(oof, 'Nested_OOF_Predictions.csv'), (test, 'final/Predictions.csv'),
                        (candidates, 'Candidate_Outer_OOF_Predictions.csv'),
                        (choices, 'Outer_Algorithm_Selections.csv')]:
        frame.to_csv(run / path, index=False)
    for context in saved['outer_order']:
        folder = run / context
        folder.mkdir(parents=True)
        pd.DataFrame(params[context]['candidates'].values()).to_csv(folder / 'Candidate_Inner_Scores.csv', index=False)
    rows = []
    for name, frame in [('Nested outer OOF', oof), ('Independent test', test)]:
        for subset, part in [('all', frame), ('experimental >= 1000 nm', frame[frame[TARGET].ge(1000)])]:
            rows.append(dict(evaluation=name, subset=subset, **metrics(part)))
    pd.DataFrame(rows).to_csv(output / 'Evaluation_Metrics.csv', index=False)
    report = dict(passed=True, tolerance_nm=ATOL, dataset_records=len(raw),
        unique_molecules=raw.canonical_smiles.nunique(), solvent_categories=raw.Solvent.nunique(),
        all_partitions_verified=True, selected_OOF_matches_candidate=True,
        historical_search_attempts=len(trials), final_algorithm=params['final']['selected']['algorithm'],
        model_prediction_max_difference_nm=differences,
        training_or_search_repeated=False, OOF_check='Archived coverage and selected-candidate consistency; not a new outer refit')
    (output / 'Verification.json').write_text(json.dumps(report, indent=2)+'\n')
    if figures:
        build(run, output / 'figures', font_dir)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--font-dir', type=Path)
    args = parser.parse_args()
    verify(args.output, args.figures, args.font_dir)
