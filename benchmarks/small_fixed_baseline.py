"""Restore the recorded matrix baseline for reproducible research comparisons."""
from contextlib import contextmanager
import json
from pathlib import Path
import fast_moss.small_matrices as small
from fast_moss.ordered_matrices import CONFIGS as ORDERED

@contextmanager
def previous_runtime():
    snapshot=json.loads(Path('results/small_fixed_baseline.json').read_text())
    if ORDERED!={tuple(r['shape']):tuple(r['config']) for r in snapshot['ordered_configs']}:
        raise ValueError('Ordered matrix baseline changed')
    saved,saved_shapes=small.CONFIGS,small.SHAPES
    small.CONFIGS={tuple(r['shape']):tuple(r['config']) for r in snapshot['small_configs']}
    small.SHAPES=set(small.CONFIGS)
    try:yield
    finally:small.CONFIGS,small.SHAPES=saved,saved_shapes
