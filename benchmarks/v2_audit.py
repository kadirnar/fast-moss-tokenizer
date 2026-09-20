"""Cross-check v2 published results, source hashes, package bytes and README table."""
import hashlib
import json
from pathlib import Path
import re
import statistics

from fast_moss.v2_loading import MODEL_ID, REVISION


def main():
    checks = []
    def check(name, value):
        assert value, name
        checks.append(name)
    def read(name):
        return json.loads(Path('results',name+'.json').read_text())
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def scope(name, report):
        check(name+' checkpoint',report['model_id']==MODEL_ID and report['revision']==REVISION)
        check(name+' native precision',report['dtype_policy']=={
            'encoder':'torch.float32','decoder':'torch.float32','quantizer':'torch.float32',
            'compute_dtype':'bf16','codec_weight_dtype':'fp32'})
        check(name+' backend',report['attention_implementation']=='sdpa')
        check(name+' true stereo',report['source']['channels']==2 and report['source']['sampling_rate']==48000
              and report['source']['original_sampling_rate']==44100)
        check(name+' all exact',report['all_exact'])
        for path in ['fast_moss/v2.py','fast_moss/v2_loading.py','fast_moss/quantizer_prepare.py',
                     'fast_moss/quantizer.py','fast_moss/normalization.py','fast_moss/graphs.py',
                     'fast_moss/storage_epoch.py','fast_moss/loading.py']:
            check(name+' runtime bytes '+path,report['runtime_sha256'][path]==digest(path))
    timing = read('v2_compare')
    scope('timing',timing)
    check('timing complete',len(timing['cases'])==4 and timing['rounds']==3 and timing['graph_warmups']==200)
    readme = Path('README.md').read_text()
    for case in timing['cases']:
        label = str((case['batch'],case['frames']))
        check(label+' stereo shape',case['input_shape']==[case['batch'],2,case['frames']*3840])
        check(label+' rounds',len(case['rounds'])==9)
        check(label+' bits',all(c['bits_equal'] for r in case['rounds'] for k in ['checks','restored'] for c in r[k]))
        for mode,median in case['medians_ms'].items():
            samples = [s for r in case['rounds'] if r['mode']==mode for s in r['timing']['wall_ms_samples']]
            check(label+mode+' sample count',len(samples)==30)
            check(label+mode+' median',median==statistics.median(samples))
        for mode,ratio in case['speedups'].items():
            check(label+mode+' ratio',ratio==case['medians_ms'][mode]/case['medians_ms']['optimized_graph'])
        check(label+' cache dispatch',all(r['counts']['linear_modules']==760 and r['counts']['conv_modules']==66
              and r['counts']['quantizer_modules']==32 and r['counts']['prepare_calls']>0
              for r in case['rounds'] if r['mode']=='optimized_graph'))
        t,s = case['medians_ms'],case['speedups']
        row = (f"| {case['batch']} | {case['frames']*80} ms | {t['original_eager']:.3f} ms | "
               f"{t['original_graph_adapter']:.3f} ms | **{t['optimized_graph']:.3f} ms** | "
               f"**{s['original_eager']:.2f}×** | {s['original_graph_adapter']:.2f}× |")
        check(label+' README table',row in readme)
    corpus = read('v2_fidelity')
    scope('fidelity',corpus)
    check('14 offline cases',len(corpus['cases'])==14)
    for case in corpus['cases']:
        check(case['name']+' checks',set(case['checks'])=={'eager','fixed_adapter','graph','changed_graph','replayed_original','restored'}
              and all(c['bits_equal'] for v in case['checks'].values() for c in v))
        check(case['name']+' dispatch',case['counts']['conv_modules']==66 and case['counts']['prepare_calls']>0)
    for batch in [1,2]:
        report = read(f'v2_streaming_b{batch}')
        scope(f'streaming {batch}',report)
        check(f'streaming {batch} length',report['batch']==batch and report['frames']==162 and report['seconds']==12.96)
        case = report['cases'][0]
        check(f'streaming {batch} bits',all(c['bits_equal'] for v in case['checks'].values() for c in v))
        check(f'streaming {batch} dispatch',case['counts']['prepare_calls']==162*32 and case['counts']['linear_calls']==162*760)
        check(f'streaming {batch} samples',case['checks']['decode'][0]['elements']==batch*2*162*3840)
    profile = read('v2_profile')
    scope('profile',profile)
    for mode, record in profile['modes'].items():
        check(mode+' profile bits',all(c['bits_equal'] for c in record['checks']))
        check(mode+' kernel counts',sum(k['calls_per_replay'] for k in record['kernels'])==record['kernel_count_per_replay'])
        check(mode+' kernel durations',abs(sum(k['ms_per_replay'] for k in record['kernels'])-record['kernel_ms_per_replay'])<1.e-9)
    package = read('v2_package')
    check('package and isolated import',package['all_match'] and package['isolated_import'])
    paths = sorted(str(p) for p in Path('fast_moss').glob('*') if p.suffix in ('.py','.json'))
    check('all package files',sorted(r['path'] for r in package['records'])==paths)
    for record in package['records']:
        check(record['path']+' packaged source',record['sha256']==digest(record['path']) and record['wheel_match'])
    tests = Path('results/v2_full_tests.txt').read_text()
    match = re.search(r'(\d+) passed in ([\d.]+)s',tests)
    check('full suite',match is not None and int(match[1])>=1146 and 'FAILED' not in tests)
    check('focused v2 suite','11 passed in' in Path('results/v2_runtime_tests.txt').read_text())
    changes = [p for p,h in timing['runtime_sha256'].items() if digest(p)!=h]
    # Only a convenience loader export was added after timing began; all
    # numerical runtime files above must match their recorded bytes exactly.
    check('timed runtime unchanged',changes in ([],['fast_moss/__init__.py']))
    report = {'all_passed':True,'check_count':len(checks),'checks':checks,
              'full_tests_passed':int(match[1]),'test_seconds':float(match[2]),
              'package_files':len(paths),'non_numerical_changes_since_timing':changes}
    Path('results/v2_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(checks),'checks;',match[1],'tests;',len(paths),'package files')


if __name__=='__main__':
    main()
