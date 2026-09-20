"""Cross-check preparation research evidence without launching GPU work."""
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.quantizer_prepare import SOURCE
from fast_moss.loading import REVISION


def main():
    checks = []
    def check(name, value):
        assert value, name
        checks.append(name)
    def read(name):
        return json.loads(Path('results', name+'.json').read_text())
    source = hashlib.sha256(SOURCE.encode()).hexdigest()
    probe = read('quantizer_prepare_probe')
    model = read('full_quantizer_prepare')
    cute = read('full_quantizer_prepare_cute')
    for name, report in [('probe',probe),('model',model),('cute',cute)]:
        check(name+' summary', report['all_exact'])
        check(name+' kernel source', report['resources']['source_sha256']==source)
        check(name+' binary', report['resources']['cubin_sha256']==probe['resources']['cubin_sha256'])
        check(name+' no local/shared storage', report['resources']['local_bytes']==report['resources']['shared_bytes']==0)
    check('synthetic cases', len(probe['stress'])==9)
    check('synthetic intermediate bits and layouts', all(c['bits_equal'] and c['strides_equal'] and c['different_bits']==0
                                                        for r in probe['stress'] for c in r['checks']))
    check('component geometries', len(probe['records'])==7)
    for row in probe['records']:
        label = str(row['shape'])
        check(label+' component rounds', len(row['rounds'])==10)
        for method, median in row['medians_ms'].items():
            check(label+method+' component median', median==statistics.median(v for r in row['rounds']
                                        if r['backend']==method for v in r['samples_ms']))
        check(label+' component ratio', row['speedup']==row['medians_ms']['current']/row['medians_ms']['candidate'])
    for name, report in [('model',model),('cute',cute)]:
        check(name+' original model scope', report['revision']==REVISION and not report['enabled_in_runtime'])
        check(name+' all corpus cases', len(report['cases'])==48)
        check(name+' preparation coverage', sum(c['preparation_checks'] for c in report['cases'])==6144)
        check(name+' selected path', all(c['candidate_calls']==160 and c['fallback_calls']==0 for c in report['cases']))
        check(name+' full codec and restored bits', all(c['bits_equal'] for r in report['cases']
                                for values in r['checks'].values() for c in values))
    check('four matched timing geometries', len(model['timings'])==4)
    for row in model['timings']:
        label = str((row['batch'],row['frames']))
        check(label+' five paired rounds', len(row['rounds'])==10)
        check(label+' timed outputs exact', all(c['bits_equal'] for r in row['rounds']
                                               for v in r['results'].values() for c in v['checks']))
        for direction, medians in row['medians_ms'].items():
            for method, median in medians.items():
                check(label+direction+method+' median', median==statistics.median(v for r in row['rounds'] if r['mode']==method
                                                                                   for v in r['results'][direction]['wall_ms']))
            check(label+direction+' speedup', row['speedups'][direction]==medians['current']/medians['candidate'])
    for batch in [1,2]:
        report = read(f'full_quantizer_prepare_streaming_b{batch}')
        check(f'stream {batch} scope', report['frames']==162 and report['batch']==batch and report['chunk_frames']==1
              and report['quantizer_prepare_research'] and not report['enabled_in_runtime'])
        check(f'stream {batch} bits', report['graph_exact'] and all(all(v['per_chunk_bits_equal'])
                                                                  for v in report['results'].values()))
        check(f'stream {batch} active encoder', report['quantizer_prepare_counts'][0]['candidate_calls']>0)
        check(f'stream {batch} unchanged decoder path', report['quantizer_prepare_counts'][1]['candidate_calls']==0)
    for frames in [1,3]:
        report = read(f'full_quantizer_prepare_profile_f{frames}')
        check(f'profile {frames} scope', report['quantizer_prepare_research'] and not report['enabled_in_runtime'])
        check(f'profile {frames} 32 fusions', report['results']['encode']['kernel_summary']['quantizer_prepare_per_replay']==32)
        check(f'profile {frames} no decoder preparation', report['results']['decode']['kernel_summary']['quantizer_prepare_per_replay']==0)
        prior = read(f'full_residual_async_profile_f{frames}')
        check(f'profile {frames} removes 160 encoder launches',
              prior['results']['encode']['kernel_summary']['kernels_per_replay']
              - report['results']['encode']['kernel_summary']['kernels_per_replay']==160)
        check(f'profile {frames} decoder launches unchanged',
              prior['results']['decode']['kernel_summary']['kernels_per_replay']
              == report['results']['decode']['kernel_summary']['kernels_per_replay'])
    check('24 focused tests', '24 passed' in Path('results/quantizer_prepare_tests.txt').read_text())
    package = read('residual_async_package')
    check('33 supported runtime files unchanged', len(package['records'])==33 and all(
        hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in package['records']))
    report = {'scope':'research exact quantizer preparation, supported runtime unchanged; integration pending',
              'previous_commit':'9e9ec47', 'source_sha256':source, 'all_passed':True,
              'check_count':len(checks), 'checks':checks}
    Path('results/quantizer_prepare_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(checks),'checks')


if __name__ == '__main__':
    main()
