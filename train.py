"""Molecule-grouped nested algorithm selection, without residual correction."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, multiprocessing, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any
import joblib, numpy as np, pandas as pd, optuna
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from data.dataset import TARGET, load_data, featurize, group_holdout, group_folds, check_partition, inner_splits, prepare_fold
from emission_project.utils import compute_metrics

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def select_candidate(rows):
    names = [row['algorithm'] for row in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError('Missing or duplicate candidate algorithms.')
    for row in rows:
        if row['algorithm'] not in ALGORITHMS or not np.isfinite(row['mean_inner_RMSE']) or row['mean_inner_RMSE'] < 0:
            raise ValueError('Invalid candidate selection score.')
    return min(rows, key=lambda row: (row['mean_inner_RMSE'], row['algorithm']))

def checked_fit_predict(model, x_train, y_train, x_valid):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_valid), dtype=float)
    if any((issubclass(item.category, ConvergenceWarning) for item in recorded)):
        raise FloatingPointError('Candidate model did not converge.')
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite candidate predictions.')
    return prediction

def fit_context(structural, raw, fit, valid, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fit = np.asarray(fit, dtype=int)
    valid = np.asarray(valid, dtype=int)
    check_partition(raw, fit, valid)
    write_json(output / 'Partition.json', {'fit_record_ids': fit.tolist(), 'valid_record_ids': valid.tolist()})
    with threadpool_limits(limits=config['threads']):
        cached = []
        for number, (a, b) in enumerate(inner_splits(raw, fit, config)):
            fold = prepare_fold(structural, raw, a, b, config)
            folder = output / 'inner' / f'{number:02d}'
            folder.mkdir(parents=True)
            write_json(folder / 'Selector.json', fold['state'])
            write_json(folder / 'Partition.json', {'fit_record_ids': a.tolist(), 'valid_record_ids': b.tolist()})
            cached.append(fold)
        candidates = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for name in config['search_spaces']:
            folder = output / 'candidates' / name
            folder.mkdir(parents=True)
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config['seed']))

            def objective(trial):
                build_model(trial, name, config['seed'], config['threads'], config['search_spaces'])
                scores = []
                try:
                    for number, fold in enumerate(cached):
                        estimator = build_model(optuna.trial.FixedTrial(trial.params), name, None if config['seed'] is None else config['seed'] + number, config['threads'], config['search_spaces'])
                        pred = checked_fit_predict(estimator, fold['x_train'], fold['y_train'], fold['x_valid'])
                        scores.append(float(np.sqrt(np.mean((pred - fold['y_valid']) ** 2))))
                except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                    trial.set_user_attr('invalid_reason', str(exc))
                    raise optuna.TrialPruned(str(exc)) from exc
                trial.set_user_attr('inner_fold_rmse', scores)
                return float(np.mean(scores))
            study.optimize(objective, n_trials=config['n_trials'])
            study.trials_dataframe().to_csv(folder / 'Optuna_Trials.csv', index=False)
            successful = sum((t.state == optuna.trial.TrialState.COMPLETE for t in study.trials))
            if not successful:
                raise ValueError(f'No successful trials for {name}.')
            row = dict(algorithm=name, mean_inner_RMSE=float(study.best_value), best_trial=study.best_trial.number, parameters=study.best_params, inner_fold_rmse=study.best_trial.user_attrs['inner_fold_rmse'], trial_attempts=len(study.trials), successful_trials=successful)
            candidates.append(row)
            write_json(folder / 'Best_Parameters.json', row)
            print(f'{output.name}: {name}, inner RMSE={study.best_value:.4f}', flush=True)
        choice = select_candidate(candidates)
        parent = prepare_fold(structural, raw, fit, valid, config)
        model = build_model(optuna.trial.FixedTrial(choice['parameters']), choice['algorithm'], config['seed'], config['threads'], config['search_spaces'])
        prediction = checked_fit_predict(model, parent['x_train'], parent['y_train'], parent['x_valid'])
        joblib.dump(model, output / 'Model.joblib', compress=3)
        if choice['algorithm'] == 'XGB':
            model.save_model(output / 'XGB_Final_Model.json')
        parent['state']['fingerprint_config'] = {'morgan_radius': config['morgan_radius'], 'morgan_bits': config['morgan_bits']}
        write_json(output / 'Selector.json', parent['state'])
        write_json(output / 'Selected_Algorithm.json', choice)
        pd.DataFrame(candidates).drop(columns=['parameters', 'inner_fold_rmse']).to_csv(output / 'Candidate_Inner_Scores.csv', index=False)
        result = raw.iloc[valid][['record_id', 'canonical_smiles', 'Solvent', TARGET]].copy()
        result['prediction_nm'] = prediction
        result['unknown_solvent'] = parent['unknown_solvent']
        result.to_csv(output / 'Predictions.csv', index=False)
        reloaded = joblib.load(output / 'Model.joblib')
        np.testing.assert_allclose(reloaded.predict(parent['x_valid']), prediction, rtol=0, atol=1e-09)
        completion = dict(algorithm=choice['algorithm'], candidates=candidates, trial_attempts=sum((row['trial_attempts'] for row in candidates)), validation_records=len(valid), metrics=compute_metrics(parent['y_valid'], prediction))
        write_json(output / 'COMPLETED.json', completion)
        return completion

def evaluation_rows(frame, label):
    rows = []
    for subset, part in [('all', frame), ('true_ge1000nm', frame[frame[TARGET] >= 1000])]:
        if len(part) >= 2:
            rows.append(dict(evaluation=label, subset=subset, n=len(part), n_molecules=part.canonical_smiles.nunique(), **compute_metrics(part[TARGET].to_numpy(float), part.prediction_nm.to_numpy(float))))
    return rows

def train(data, output, config):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if config['residual_correction'] is not False:
        raise ValueError('This release has no residual correction.')
    raw = load_data(data)
    fit, test = group_holdout(raw[TARGET].to_numpy(), raw.canonical_smiles.to_numpy(), config['seed'], config['test_size'], config['stratify_bins'])
    check_partition(raw, fit, test)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'Study_Config.json', config)
    split = raw.copy()
    split['split'] = 'train'
    split.loc[test, 'split'] = 'test'
    split.to_csv(output / 'Split_Assignments.csv', index=False)
    structural = featurize(raw, [], radius=config['morgan_radius'], bits=config['morgan_bits'])
    outer = [(fit[a], fit[b]) for a, b in group_folds(raw.iloc[fit][TARGET].to_numpy(), raw.iloc[fit].canonical_smiles.to_numpy(), config['seed'], config['outer_folds'], config['stratify_bins'])]
    coverage = np.concatenate([b for _, b in outer])
    if sorted(coverage.tolist()) != sorted(fit.tolist()):
        raise ValueError('Incomplete outer OOF coverage.')
    for a, b in outer:
        check_partition(raw, a, b)
    write_json(output / 'CV_Partitions.json', {'outer_folds': [{'fold': i + 1, 'fit_record_ids': a.tolist(), 'valid_record_ids': b.tolist()} for i, (a, b) in enumerate(outer)]})
    source_files = [Path(data), Path(__file__), Path(__file__).parent / 'predict.py', Path(__file__).parent / 'data/dataset.py', Path(__file__).parent / 'emission_project/utils.py']
    source_hashes = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    write_json(output / 'Source_Hashes.json', source_hashes)
    write_json(output / 'Environment.json', {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'rdkit', 'optuna', 'catboost', 'lightgbm']})
    with ProcessPoolExecutor(max_workers=config['outer_jobs'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(fit_context, structural, raw, a, b, config, output / 'outer' / f'{i:02d}'): i for i, (a, b) in enumerate(outer)}
        for future in as_completed(futures):
            future.result()
    pieces = []
    choices = []
    for i in range(len(outer)):
        folder = output / 'outer' / f'{i:02d}'
        part = pd.read_csv(folder / 'Predictions.csv')
        part['outer_fold'] = i + 1
        pieces.append(part)
        row = json.loads((folder / 'Selected_Algorithm.json').read_text())
        choices.append({'outer_fold': i + 1, 'algorithm': row['algorithm'], 'mean_inner_RMSE': row['mean_inner_RMSE']})
    oof = pd.concat(pieces).sort_values('record_id').reset_index(drop=True)
    if oof.record_id.tolist() != sorted(fit.tolist()) or oof.record_id.duplicated().any():
        raise ValueError('Incomplete or duplicate outer predictions.')
    if oof.groupby('canonical_smiles').outer_fold.nunique().max() != 1:
        raise ValueError('A molecule occurs in multiple outer validation folds.')
    oof.to_csv(output / 'Nested_OOF_Predictions.csv', index=False)
    pd.DataFrame(choices).to_csv(output / 'Outer_Algorithm_Selections.csv', index=False)
    final = fit_context(structural, raw, fit, test, config, output / 'final')
    heldout = pd.read_csv(output / 'final/Predictions.csv')
    pd.DataFrame(evaluation_rows(oof, 'outer_OOF') + evaluation_rows(heldout, 'historical_test')).to_csv(output / 'Evaluation_Metrics.csv', index=False)
    for path, expected in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed during training.')
    write_json(output / 'COMPLETED.json', {'outer_folds': len(outer), 'OOF_records': len(oof), 'training_records': len(fit), 'test_records': len(test), 'selected_algorithm': final['algorithm'], 'trial_attempts': (len(outer) + 1) * len(config['search_spaces']) * config['n_trials'], 'residual_correction': False})
    hashes = {str(f.relative_to(output)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(output.rglob('*')) if f.is_file()}
    write_json(output / 'Artifact_Hashes.json', hashes)
    return final
ALGORITHMS = ['XGB','CAT','LGBM','RF','GBR','KNN','KRR','SVR']

def load_config(path):
    config=json.loads(Path(path).read_text())
    required=['test_size','outer_folds','inner_folds','stratify_bins','selected_features',
              'correlation_threshold','rfe_estimators','rfe_depth','rfe_step','n_trials',
              'morgan_radius','morgan_bits','search_spaces']
    missing=[key for key in required if key not in config]
    if missing:raise ValueError('Configuration missing: '+', '.join(missing))
    config={**config,'seed':config.get('seed'),'threads':config.get('threads',1),
            'outer_jobs':config.get('outer_jobs',1),'residual_correction':False}
    for key in ['stratify_bins','selected_features','rfe_estimators','rfe_step','n_trials',
                'morgan_radius','morgan_bits','threads','outer_jobs']:
        if not isinstance(config[key],int) or isinstance(config[key],bool) or config[key]<1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ['outer_folds','inner_folds']:
        if not isinstance(config[key],int) or config[key]<2:raise ValueError(f'{key} must be at least two.')
    if not 0<float(config['test_size'])<1 or not 0<float(config['correlation_threshold'])<=1:
        raise ValueError('Invalid test_size or correlation_threshold.')
    if config['rfe_depth'] is not None and (not isinstance(config['rfe_depth'],int) or config['rfe_depth']<1):
        raise ValueError('rfe_depth must be null or a positive integer.')
    if config['seed'] is not None and (not isinstance(config['seed'],int) or isinstance(config['seed'],bool)):
        raise ValueError('seed must be null or an integer.')
    if not isinstance(config['search_spaces'],dict) or not config['search_spaces']:
        raise ValueError('search_spaces must define candidate algorithms.')
    if any(name not in ALGORITHMS or not isinstance(space,dict) for name,space in config['search_spaces'].items()):
        raise ValueError('Unsupported candidate algorithm or parameter specification.')
    return config

def draw_parameters(trial,space):
    params={}
    for name,spec in space.items():
        if not isinstance(spec,dict):
            params[name]=spec
        elif spec.get('kind')=='int':
            params[name]=trial.suggest_int(name,spec['low'],spec['high'],
                                          step=spec.get('step',1),log=spec.get('log',False))
        elif spec.get('kind')=='float':
            params[name]=trial.suggest_float(name,spec['low'],spec['high'],
                                            step=spec.get('step'),log=spec.get('log',False))
        elif spec.get('kind')=='categorical':
            params[name]=trial.suggest_categorical(name,spec['choices'])
        else:
            raise ValueError(f'Unknown search specification for {name}.')
    return params

def build_model(trial,name,seed,threads,search_spaces):
    params=draw_parameters(trial,search_spaces[name])
    if name=='XGB':
        options=dict(objective='reg:squarederror',tree_method='hist',device='cpu',
                     verbosity=0,n_jobs=threads,random_state=seed)
        return XGBRegressor(**{**options,**params})
    if name=='CAT':
        options=dict(loss_function='RMSE',verbose=False,thread_count=threads,allow_writing_files=False)
        if seed is not None:options['random_seed']=seed
        return CatBoostRegressor(**{**options,**params})
    if name=='LGBM':
        return LGBMRegressor(**{'verbosity':-1,'n_jobs':threads,'random_state':seed,**params})
    if name=='RF':
        return RandomForestRegressor(**{'n_jobs':threads,'random_state':seed,**params})
    if name=='GBR':
        return GradientBoostingRegressor(**{'random_state':seed,**params})
    if name=='KNN':estimator=KNeighborsRegressor(**{'n_jobs':threads,**params})
    elif name=='KRR':estimator=KernelRidge(**params)
    elif name=='SVR':estimator=SVR(**params)
    else:raise ValueError(f'Unsupported algorithm: {name}')
    return Pipeline([('scaler',StandardScaler()),('model',estimator)])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--config',required=True,help='Your local configuration; not supplied with the repository.')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    train(args.data,args.output,load_config(args.config))

if __name__=='__main__':
    main()
