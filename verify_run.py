"""Verify all eight searches completed in each nested-CV and final context."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from train import ALGORITHMS, select_candidate


def verify(run):
    run = Path(run)
    config = json.loads((run/'Study_Config.json').read_text())
    if 'fixed_gbr_parameters' in config or config['n_trials'] != 50:
        raise ValueError('Expected fresh 50-trial searches for every algorithm.')
    contexts = [f'outer/{i:02d}' for i in range(config['outer_folds'])] + ['final']
    count = 0
    for context in contexts:
        folder = run/context
        rows = []
        for name in ALGORITHMS:
            base = folder/'candidates'/name
            trials = pd.read_csv(base/'Optuna_Trials.csv')
            row = json.loads((base/'Best_Parameters.json').read_text())
            if len(trials) != 50 or row['trial_attempts'] != 50:
                raise ValueError(f'Incomplete search: {context}/{name}')
            complete = trials[trials.state.eq('COMPLETE')]
            if complete.empty or not np.isclose(complete.value.min(), row['mean_inner_RMSE'], rtol=0, atol=1e-8):
                raise ValueError(f'Best trial does not match selection: {context}/{name}')
            if len(row['inner_fold_rmse']) != config['inner_folds']:
                raise ValueError(f'Missing inner-fold scores: {context}/{name}')
            if not np.isclose(np.mean(row['inner_fold_rmse']), row['mean_inner_RMSE'], rtol=0, atol=1e-8):
                raise ValueError(f'Incorrect mean inner score: {context}/{name}')
            rows.append(row)
            count += len(trials)
        expected = select_candidate(rows)
        actual = json.loads((folder/'Selected_Algorithm.json').read_text())
        if actual['algorithm'] != expected['algorithm'] or actual['parameters'] != expected['parameters']:
            raise ValueError(f'Incorrect selected candidate: {context}')
    done = json.loads((run/'COMPLETED.json').read_text())
    if done['trial_attempts'] != count or count != len(contexts)*8*50:
        raise ValueError('Trial totals do not match.')
    return dict(passed=True, contexts=len(contexts), algorithms=8,
                trials_per_algorithm=50, total_trials=count,
                selected_algorithm=done['selected_algorithm'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2))
