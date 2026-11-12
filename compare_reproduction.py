"""Check freshly fitted outputs against frozen reference values; never train on them."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from predict import predict_frame

ROOT = Path(__file__).resolve().parent
PREDICTION_ATOL = 1e-7  # nm; fixed before comparison


def read(path):
    return json.loads(Path(path).read_text())


def selector_signature(state):
    keys = ['fit_record_ids', 'selected_features', 'after_variance', 'after_correlation',
            'rfe_ranking', 'protected_features', 'protected_training_nonzero_counts']
    obj = {key: state[key] for key in keys}
    obj['solvent_categories'] = sorted(state['solvent_categories'])
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compare_frame(actual, expected, keys):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f'Duplicate prediction keys: {keys}')
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[keys], expected[keys], check_dtype=False)
    for key in ['outer_fold', 'inner_fold', 'λem (nm)']:
        if key in expected and key not in keys:
            np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_allclose(actual.prediction_nm, expected.prediction_nm,
                               rtol=0, atol=PREDICTION_ATOL)
    return dict(records=len(actual), max_abs_prediction_difference_nm=float(
        np.max(np.abs(actual.prediction_nm.to_numpy() - expected.prediction_nm.to_numpy()))))


def verify(run, reference):
    run, reference = Path(run), Path(reference)
    if not (run / 'COMPLETED.json').exists():
        raise ValueError('The complete nested run has not finished.')
    selectors = read(reference / 'Selectors.json')
    parameters = read(ROOT / 'configs/Estimator_Parameters.json')
    selector_count = score_count = 0
    for context, states in selectors.items():
        parent = run / context
        for path, expected in [(parent / 'Selector.json', states['parent'])] + [
                (parent / 'inner' / f'{i:02d}' / 'Selector.json', s)
                for i, s in enumerate(states['inner'])]:
            state = read(path)
            if selector_signature(state) != expected['sequence_sha256']:
                raise ValueError(f'Feature sequence or partition differs: {path}')
            for key in ['rfe_scaler_mean', 'rfe_scaler_scale']:
                np.testing.assert_allclose(state[key], expected[key], rtol=0, atol=1e-10)
            selector_count += 1
        selected = read(parent / 'Selected_Algorithm.json')
        if selected['algorithm'] != parameters[context]['selected']['algorithm']:
            raise ValueError(f'Selected algorithm differs: {context}')
        for name, expected in parameters[context]['candidates'].items():
            actual = read(parent / 'candidates' / name / 'Best_Parameters.json')
            if actual['parameters'] != expected['parameters']:
                raise ValueError(f'Candidate parameters differ: {context}/{name}')
            for key in ['mean_inner_RMSE', 'inner_fold_rmse']:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=PREDICTION_ATOL)
            score_count += 1
    checks = {}
    for output, frozen, keys in [
        ('Nested_OOF_Predictions.csv', 'Procedure_Outer_OOF.csv', ['record_id']),
        ('final/Predictions.csv', 'Independent_Test.csv', ['record_id']),
        ('final/Training_Fit_Predictions.csv', 'Training_Fit.csv', ['record_id']),
        ('Candidate_Outer_OOF_Predictions.csv', 'Eight_Algorithm_Outer_OOF.csv', ['algorithm', 'record_id']),
    ]:
        checks[frozen] = compare_frame(pd.read_csv(run / output), pd.read_csv(reference / frozen), keys)
    inner = []
    for name in parameters['final']['candidates']:
        frame = pd.read_csv(run / 'final/candidates' / name / 'Inner_OOF_Predictions.csv')
        frame['algorithm'] = name
        inner.append(frame)
    checks['Final_Inner_OOF.csv'] = compare_frame(pd.concat(inner),
        pd.read_csv(reference / 'Final_Inner_OOF.csv'), ['algorithm', 'record_id'])
    for name in ['External13.csv', 'SI6_Predictions.csv']:
        expected = pd.read_csv(reference / name)
        actual = predict_frame(expected, run / 'final', threads=1)
        # Row identity is retained by predict_frame, including repeated structures.
        expected = expected.assign(check_row_id=np.arange(len(expected)))
        actual = actual.assign(check_row_id=np.arange(len(actual)))
        checks[name] = compare_frame(actual, expected, ['check_row_id'])
    return dict(passed=True, prediction_tolerance_nm=PREDICTION_ATOL,
        selectors_verified=selector_count, candidate_parameter_sets_verified=score_count,
        final_algorithm=read(run / 'final/Selected_Algorithm.json')['algorithm'],
        mode=read(run / 'Reproduction_Mode.json'), prediction_checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--reference', default=str(ROOT / 'reference'))
    args = parser.parse_args()
    report = verify(args.run, args.reference)
    (Path(args.run) / 'Reproduction_Verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
