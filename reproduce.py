"""Run eight-algorithm nested selection from the supplied data and frozen folds."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New output directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    threads = json.loads((root/'configs/paper.json').read_text())['threads']
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = str(threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:threads])
    import train as training
    config = training.load_config(root/'configs/paper.json')
    if set(config['search_spaces']) != set(training.ALGORITHMS) or config['n_trials'] != 50:
        raise ValueError('This entry requires all eight algorithms and 50 trials each.')
    training.train(root/'data/data/nir2_emission_dataset.csv', args.output, config,
                   root/'splits/Frozen_Partitions.json')
    from verify_run import verify
    report = verify(Path(args.output))
    training.write_json(Path(args.output)/'Search_Verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
