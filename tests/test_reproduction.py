import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data.dataset import TARGET, fit_selector
from train import fit_context
from compare_reproduction import compare_frame


class ReproductionTests(unittest.TestCase):
    def test_comparison_rejects_changed_predictions_and_duplicate_ids(self):
        expected = pd.DataFrame({'record_id': [0, 1], 'prediction_nm': [500., 600.]})
        self.assertEqual(compare_frame(expected, expected, ['record_id'])['records'], 2)
        changed = expected.copy()
        changed.loc[0, 'prediction_nm'] += 0.01
        with self.assertRaises(AssertionError):
            compare_frame(changed, expected, ['record_id'])
        with self.assertRaises(ValueError):
            compare_frame(pd.concat([expected, expected]), expected, ['record_id'])

    def setUp(self):
        self.raw = pd.DataFrame({'record_id': range(12),
            'canonical_smiles': [f'molecule_{i}' for i in range(12)],
            'Solvent': ['THF', 'MTBE', 'DCM'] * 4,
            TARGET: np.arange(12, dtype=float) * 2 + 500})
        self.x = pd.DataFrame({'Morgan_0': np.arange(12, dtype=float),
                               'Morgan_1': np.arange(12, dtype=float) ** 2})
        self.cfg = dict(seed=2, threads=1, inner_folds=3, stratify_bins=2,
            selected_features=2, correlation_threshold=1.0,
            rfe_estimators=3, rfe_depth=2, rfe_step=1,
            morgan_radius=2, morgan_bits=2, n_trials=2,
            export_candidate_predictions=True,
            search_spaces={'GBR': {'n_estimators': {'kind':'int','low':2,'high':3},
                                    'max_depth': 1}},
            fixed_gbr_parameters={'n_estimators':2,'max_depth':1})

    def test_solvent_columns_follow_saved_numeric_order(self):
        self.cfg['solvent_feature_order']=['Solvent_MTBE','Solvent_DCM','Solvent_THF']
        _,_,state=fit_selector(self.x,self.raw,self.cfg)
        actual=[x for x in state['after_variance'] if x.startswith('Solvent_')]
        self.assertEqual(actual,self.cfg['solvent_feature_order'])

    def test_replay_refits_and_rescores_without_reference_values(self):
        self.cfg.pop('fixed_gbr_parameters')
        params={'GBR':{'n_estimators':3,'max_depth':1}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'run'
            fit_context(self.x,self.raw,list(range(9)),list(range(9,12)),self.cfg,p,True,
                        replay_parameters=params)
            row=json.loads((p/'candidates/GBR/Best_Parameters.json').read_text())
            pred=pd.read_csv(p/'Predictions.csv')
            self.assertEqual(row['parameters'],params['GBR'])
            self.assertEqual(row['trial_attempts'],0)
            self.assertTrue(np.isfinite(row['mean_inner_RMSE']))
            self.assertEqual(len(pred),3)
            self.assertTrue((p/'Model.joblib').is_file())


if __name__=='__main__':
    unittest.main()
