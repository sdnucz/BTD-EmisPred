"""Maximum sparse-count Morgan Tanimoto similarity to training structures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from data.dataset import load_data
from frozen_partitions import load_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/data/nir2_emission_dataset.csv'))
    parser.add_argument('--partitions', type=Path, default=Path('splits/Frozen_Partitions.json'))
    parser.add_argument('--config', type=Path, default=Path('configs/paper.json'))
    parser.add_argument('--queries', type=Path, default=Path('data/similarity_queries.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs/count_similarity'))
    args = parser.parse_args()
    raw = load_data(args.data)
    config = json.loads(args.config.read_text())
    partitions = load_frozen(args.partitions, args.data, raw.canonical_smiles.tolist(), config)
    training = raw.iloc[partitions['holdout']['fit_record_ids']]
    reference = training.drop_duplicates('canonical_smiles').reset_index(drop=True)
    queries = pd.read_csv(args.queries, dtype=str)
    required = ['record_id', 'compound', 'cohort', 'SMILES']
    if any(c not in queries for c in required) or queries.empty:
        raise ValueError('Queries require nonempty record_id, compound, cohort and SMILES columns.')
    if queries[required].isna().any().any() or queries.record_id.duplicated().any():
        raise ValueError('Query values must be present and record_id must be unique.')
    molecules = [Chem.MolFromSmiles(s) for s in queries.SMILES]
    for record_id, mol in zip(queries.record_id, molecules):
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f'Invalid or empty query SMILES: {record_id}')
    # Sparse count fingerprints retain environment multiplicities without folding
    # them into a fixed-length bit vector. Solvent features are not included.
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=False)
    fingerprints = [generator.GetSparseCountFingerprint(Chem.MolFromSmiles(s))
                    for s in reference.canonical_smiles]
    rows = []
    for row, mol in zip(queries.itertuples(index=False), molecules):
        fingerprint = generator.GetSparseCountFingerprint(mol)
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints))
        nearest = int(np.argmax(similarities))
        neighbor = reference.iloc[nearest]
        canonical = Chem.MolToSmiles(mol, canonical=True)
        rows.append(dict(record_id=row.record_id, compound=row.compound, cohort=row.cohort,
                         canonical_smiles=canonical,
                         max_sparse_count_tanimoto=float(similarities[nearest]),
                         nearest_training_record_id=int(neighbor.record_id),
                         nearest_training_canonical_smiles=neighbor.canonical_smiles,
                         exact_structure_in_training=canonical in set(reference.canonical_smiles)))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / 'Count_similarity.csv', index=False)
    metadata = dict(fingerprint='Sparse count Morgan', radius=2, include_chirality=False,
                    similarity='RDKit Tanimoto; maximum over unique training structures',
                    rdkit_version=rdBase.rdkitVersion, training_records=len(training),
                    training_unique_structures=len(reference), queries=len(queries),
                    input_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in [('dataset', args.data),
                                                     ('partitions', args.partitions),
                                                     ('queries', args.queries)]})
    (args.output / 'Count_similarity_info.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Computed {len(queries)} queries against {len(reference)} training structures.')


if __name__ == '__main__':
    main()
