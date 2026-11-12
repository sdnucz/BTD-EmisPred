import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from data.dataset import TARGET
from train import ALGORITHMS, fit_context, load_config

ROOT = Path(__file__).resolve().parents[1]


class SearchTests(unittest.TestCase):
    def test_paper_searches_all_eight_candidates(self):
        config = load_config(ROOT / 'configs/paper.json')
        self.assertEqual(set(config['search_spaces']), set(ALGORITHMS))
        self.assertEqual(config['n_trials'], 50)
        self.assertNotIn('fixed_gbr_parameters', config)

    def test_old_fixed_gbr_configuration_is_rejected(self):
        config = json.loads((ROOT / 'configs/paper.json').read_text())
        config['fixed_gbr_parameters'] = {k: 1 for k in config['search_spaces']['GBR']}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_gbr_performs_search_and_saves_trial_records(self):
        raw = pd.DataFrame({'record_id': range(12),
            'canonical_smiles': [f'molecule_{i}' for i in range(12)],
            'Solvent': ['THF', 'MTBE', 'DCM'] * 4,
            TARGET: np.arange(12, dtype=float) * 2 + 500})
        x = pd.DataFrame({'Morgan_0': np.arange(12, dtype=float),
                          'Morgan_1': np.arange(12, dtype=float) ** 2})
        cfg = dict(seed=2, threads=1, inner_folds=3, stratify_bins=2,
            selected_features=2, correlation_threshold=1.0,
            rfe_estimators=3, rfe_depth=2, rfe_step=1,
            morgan_radius=2, morgan_bits=2, n_trials=2,
            export_candidate_predictions=True,
            search_spaces={'GBR': {'n_estimators': {'kind':'int','low':2,'high':3},
                                    'max_depth': 1}})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            fit_context(x, raw, list(range(9)), list(range(9,12)), cfg, out, True)
            result = json.loads((out / 'candidates/GBR/Best_Parameters.json').read_text())
            self.assertEqual(result['trial_attempts'], 2)
            self.assertEqual(len(pd.read_csv(out / 'candidates/GBR/Optuna_Trials.csv')), 2)
            self.assertEqual(len(result['inner_fold_rmse']), 3)


if __name__ == '__main__':
    unittest.main()
