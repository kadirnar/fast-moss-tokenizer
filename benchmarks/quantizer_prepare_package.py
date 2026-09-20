"""Verify all runtime wheel bytes and import v2/LFQ outside the source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from fast_moss.quantizer_prepare import SOURCE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('wheel', type=Path)
    p.add_argument('--output', default='results/v2_package.json')
    args = p.parse_args()
    root = Path.cwd()
    source_hash = hashlib.sha256(SOURCE.encode()).hexdigest()
    report = {'scope':'v2 and supported LFQ preparation wheel/source byte audit and isolated import',
              'wheel_sha256':hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
              'source_sha256':source_hash, 'records':[]}
    with zipfile.ZipFile(args.wheel) as archive:
        for path in sorted(Path('fast_moss').glob('*')):
            if path.suffix not in ('.py','.json'):
                continue
            value = path.read_bytes()
            assert archive.read(str(path)) == value, path
            report['records'].append({'path':str(path), 'sha256':hashlib.sha256(value).hexdigest(), 'wheel_match':True})
        with tempfile.TemporaryDirectory(prefix='moss-prepare-wheel-') as temporary:
            archive.extractall(temporary)
            code = ('import sys,hashlib;sys.path.insert(0,'+repr(temporary)+');'
                    'import fast_moss.quantizer_prepare as kernel;'
                    'assert kernel.__file__.startswith('+repr(temporary)+');'
                    'assert hashlib.sha256(kernel.SOURCE.encode()).hexdigest()=='+repr(source_hash)+';'
                    'import fast_moss.v2 as v2;'
                    'import fast_moss.v2_pointwise as pointwise;'
                    'from fast_moss import load_model_v2;'
                    'assert v2.__file__.startswith('+repr(temporary)+');'
                    'assert pointwise.__file__.startswith('+repr(temporary)+');'
                    'assert v2.load_model is load_model_v2;'
                    'assert v2.MODEL_ID=="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2";'
                    'print("isolated import passed")')
            subprocess.run([str(root/'.venv/bin/python'),'-I','-c',code],cwd=temporary,check=True)
    report.update(all_match=True,isolated_import=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(report['records']),'package files')


if __name__=='__main__':
    main()
