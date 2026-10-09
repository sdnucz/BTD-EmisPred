"""Batch emission predictions with the selector saved during training."""
from pathlib import Path
import argparse,json
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from data.dataset import featurize,feature_matrix
from emission_project.utils import robust_read_csv

def predict_frame(frame,model_directory,threads=1,feature_config=None):
    directory=Path(model_directory)
    state=json.loads((directory/'Selector.json').read_text())
    columns=state['selected_features']
    if not columns or len(columns)!=len(set(columns)):
        raise ValueError('Selected feature names are missing or duplicated.')
    features=feature_config or state.get('fingerprint_config')
    if not features:
        raise ValueError('Fingerprint configuration is required for this older selector.')
    structural=featurize(frame,[],radius=features['morgan_radius'],bits=features['morgan_bits'])
    x,unknown=feature_matrix(structural,frame.Solvent,state['solvent_categories'])
    if not set(columns)<=set(x.columns):
        raise ValueError('Saved features cannot be reconstructed from this input.')
    if (directory/'XGB_Final_Model.json').exists():
        model=XGBRegressor();model.load_model(directory/'XGB_Final_Model.json')
        model.set_params(n_jobs=threads,device='cpu')
    else:
        model=joblib.load(directory/'Model.joblib')
    with threadpool_limits(limits=threads):
        predicted=np.asarray(model.predict(x[columns]),dtype=float)
    if len(predicted)!=len(frame) or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite or incomplete predictions.')
    result=frame.copy();result['prediction_nm']=predicted;result['unknown_solvent']=unknown
    return result
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--config',help='Local feature settings for an older saved selector.')
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if args.threads<1:parser.error('Use a positive thread count.')
    output=Path(args.output)
    if output.exists():raise FileExistsError(f'Output already exists: {output}')
    frame=robust_read_csv(Path(args.input))
    features=json.loads(Path(args.config).read_text()) if args.config else None
    result=predict_frame(frame,args.model,args.threads,features)
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
if __name__=='__main__':
    main()
