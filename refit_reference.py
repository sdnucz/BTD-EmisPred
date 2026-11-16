"""Refit archived candidate parameters on the fixed training partitions."""
import argparse
import os
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='New output directory')
    parser.add_argument('--context',default='all',help='all, final, or outer/00 through outer/09')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    threads=json.loads((root/'configs/paper.json').read_text())['threads']
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]=str(threads)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:threads])
    import train as training
    from data.dataset import load_data,featurize
    from frozen_partitions import load_frozen
    root=Path(__file__).resolve().parent
    data=root/'data/data/nir2_emission_dataset.csv'
    config=training.load_config(root/'configs/paper.json')
    partitions=root/'splits/Frozen_Partitions.json'
    parameters=root/'reference/Estimator_Parameters.json'
    if args.context=='all':
        training.train(data,args.output,config,partitions,parameters)
    else:
        raw=load_data(data)
        frozen=load_frozen(partitions,data,raw.canonical_smiles.tolist(),config)
        if args.context not in frozen['contexts']:
            parser.error('Unknown context')
        structural=featurize(raw,[],radius=config['morgan_radius'],bits=config['morgan_bits'])
        part=frozen['contexts'][args.context]['partition']
        replay=None if parameters is None else {
            name:row['parameters'] for name,row in json.loads(parameters.read_text())[args.context]['candidates'].items()}
        training.fit_context(structural,raw,part['fit_record_ids'],part['valid_record_ids'],
            config,args.output,compare_candidates=True,frozen=frozen,replay_parameters=replay)
    training.write_json(Path(args.output)/'Reproduction_Mode.json',dict(mode='refit',context=args.context,
        searches_repeated=False,selectors_refitted=True,
        predictions_refitted=True,reference_predictions_used_for_fitting=False))


if __name__=='__main__':
    main()
