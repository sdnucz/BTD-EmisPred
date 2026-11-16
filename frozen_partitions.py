"""Validate and consume the exact, archived molecule-grouped partitions."""
import hashlib
import json
from pathlib import Path


def validate_partitions(saved, groups, outer_count, inner_count):
    def check(part, parent):
        a,b=part['fit_record_ids'],part['valid_record_ids']
        if (not a or not b or any(type(x) is not int for x in a+b)
                or len(set(a+b))!=len(a+b) or set(a+b)!=set(parent)):
            raise ValueError('Invalid IDs, overlap, or incomplete parent coverage.')
        if {groups[i] for i in a} & {groups[i] for i in b}:
            raise ValueError('Molecules cross a training/validation boundary.')
    check(saved['holdout'],range(len(groups)))
    train=saved['holdout']['fit_record_ids']
    if saved['contexts']['final']['partition']!=saved['holdout']:
        raise ValueError('Final fit/test partition differs from holdout.')
    order=saved['outer_order']
    if (len(order)!=outer_count or len(set(order))!=outer_count
            or set(saved['contexts'])!=set(order)|{'final'}):
        raise ValueError('Unexpected outer contexts.')
    outer_valid=[]
    for name,context in saved['contexts'].items():
        parent=context['partition']['fit_record_ids']
        if name!='final':
            check(context['partition'],train)
            outer_valid.extend(context['partition']['valid_record_ids'])
        if len(context['inner'])!=inner_count:
            raise ValueError('Unexpected inner fold count.')
        coverage=[]
        for split in context['inner']:
            check(split,parent)
            coverage.extend(split['valid_record_ids'])
        if sorted(coverage)!=sorted(parent):
            raise ValueError('Inner validation coverage is incomplete or duplicated.')
    if sorted(outer_valid)!=sorted(train):
        raise ValueError('Outer validation coverage is incomplete or duplicated.')


def load_frozen(path, data, groups, config):
    path=Path(path)
    expected=path.with_name('Dataset_SHA256.txt').read_text().strip().split()[0]
    if hashlib.sha256(Path(data).read_bytes()).hexdigest()!=expected:
        raise ValueError('Dataset bytes do not match the frozen partition archive.')
    saved=json.loads(path.read_text())
    validate_partitions(saved,groups,config['outer_folds'],config['inner_folds'])
    return saved


def inner_for_parent(saved,parent):
    matches=[c['inner'] for c in saved['contexts'].values()
             if c['partition']['fit_record_ids']==list(parent)]
    if len(matches)!=1:
        raise ValueError('No unique frozen context for this ordered training set.')
    return matches[0]
