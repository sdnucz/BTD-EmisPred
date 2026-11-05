"""Selection and prediction-boundary checks with a small synthetic dataset."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data.dataset import TARGET
from train import fit_context, select_candidate


class TrainingTests(unittest.TestCase):
    def test_exact_tie_uses_predefined_algorithm_order(self):
        rows = [{'algorithm': name, 'mean_inner_RMSE': 1.0}
                for name in ['CAT', 'XGB']]
        self.assertEqual(select_candidate(rows)['algorithm'], 'XGB')

    def test_exports_and_selection_do_not_depend_on_test_labels(self):
        rng = np.random.default_rng(12)
        x = pd.DataFrame(rng.normal(size=(18, 8)),
                         columns=[f'Morgan_{i}' for i in range(8)])
        raw = pd.DataFrame({
            'record_id': np.arange(18),
            'canonical_smiles': [f'molecule_{i}' for i in range(18)],
            'Solvent': ['A', 'B'] * 6 + ['C'] * 6,
            TARGET: 600 + 50 * x.Morgan_0,
        })
        config = dict(seed=48, threads=1, inner_folds=3, stratify_bins=2,
                      selected_features=3, correlation_threshold=0.95,
                      rfe_estimators=5, rfe_depth=3, rfe_step=2,
                      morgan_radius=2, morgan_bits=8, n_trials=1,
                      export_candidate_predictions=True,
                      search_spaces={'GBR': {'n_estimators': 5, 'max_depth': 1},
                                     'KNN': {'n_neighbors': 2}})
        fit, valid = np.arange(12), np.arange(12, 18)
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / 'outer', Path(temp) / 'final'
            fit_context(x, raw, fit, valid, config, first, compare_candidates=True)
            altered = raw.copy()
            altered.loc[valid, TARGET] += 10000
            fit_context(x, altered, fit, valid, config, second)
            choice = json.loads((first / 'Selected_Algorithm.json').read_text())
            self.assertEqual(choice, json.loads((second / 'Selected_Algorithm.json').read_text()))
            selected = pd.read_csv(first / 'Predictions.csv')
            np.testing.assert_array_equal(selected.prediction_nm,
                                          pd.read_csv(second / 'Predictions.csv').prediction_nm)
            self.assertTrue(selected.unknown_solvent.all())
            for name in config['search_spaces']:
                folder = first / 'candidates' / name
                self.assertTrue((folder / 'Predictions.csv').exists())
                inner = pd.read_csv(folder / 'Inner_OOF_Predictions.csv')
                self.assertEqual(sorted(inner.record_id), fit.tolist())
                self.assertFalse(inner.record_id.duplicated().any())
                params = json.loads((folder / 'Best_Parameters.json').read_text())
                rmses = [np.sqrt(np.mean((part[TARGET] - part.prediction_nm) ** 2))
                         for _, part in inner.groupby('inner_fold')]
                np.testing.assert_allclose(rmses, params['inner_fold_rmse'])
                # Final test predictions are produced only for the selected algorithm.
                self.assertFalse((second / 'candidates' / name / 'Predictions.csv').exists())
            chosen = pd.read_csv(first / 'candidates' / choice['algorithm'] / 'Predictions.csv')
            np.testing.assert_array_equal(chosen.prediction_nm, selected.prediction_nm)
            training = pd.read_csv(second / 'Training_Fit_Predictions.csv')
            self.assertEqual(training.record_id.tolist(), fit.tolist())


if __name__ == '__main__':
    unittest.main()
