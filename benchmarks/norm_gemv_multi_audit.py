"""Audit the rejected multi-output experiment and direct codec roundtrip."""
import hashlib,json,statistics
from pathlib import Path
from benchmarks.norm_gemv_multi import SOURCE
from fast_moss.loading import REVISION


def main():
    checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    def read(name):return json.loads(Path('results',name+'.json').read_text())
    source=hashlib.sha256(SOURCE.encode()).hexdigest()
    probe=read('norm_gemv_multi_probe');confirm=read('norm_gemv_multi_confirm');resources=read('norm_gemv_multi_resources')
    for name,r in [('probe',probe),('confirm',confirm)]:
        check(name+' exact',r['all_exact'] and r['revision']==REVISION)
        check(name+' source',r['source_sha256']==source)
        for row in r['records']:
            for ring in row['rings']:
                label=f'{name} {row["mode"]} ring {ring["length"]}'
                check(label+' sample bits',all(t['bits_equal'] for t in ring['rounds']))
                for method,v in ring['medians_ms'].items():
                    check(label+method+' median',v==statistics.median(t['per_call_ms'] for t in ring['rounds'] if t['backend']==method))
                check(label+' ratios',all(abs(v-ring['medians_ms']['current']/ring['medians_ms'][k])<1e-12 for k,v in ring['speedups'].items()))
    check('244 captured schedules',sum(len(r['trials']) for r in probe['records'])==244)
    check('captured normalization and output bits',all(t['bits_equal'] for r in probe['records'] for t in r['trials']))
    check('12 finalists',sum(len(r['configs']) for r in confirm['records'])==12)
    check('308 stress comparisons',sum(2*len(r['checks']) for r in confirm['records'])==308)
    check('stress bits',all(c['eager'] and c['graph'] for r in confirm['records'] for c in r['checks']))
    check('no distinct-weight candidate above 0.5 percent',all(max(r['rings'][1]['speedups'].values())<=1.005 for r in confirm['records']))
    check('compiler provenance',resources['source_sha256']==source and len(resources['records'])==12 and not resources['enabled_in_runtime'])
    for row in resources['records']:
        label=str((row['mode'],row['config']))
        prior=next(r for r in confirm['records'] if r['mode']==row['mode'])
        check(label+' resources',row['matches_confirmation'] and row['resources']==prior['resources'][json.dumps(row['config'])])
        check(label+' instructions',bool(row['sass_async_copies'])==bool(row['config'][6]) and row['sass_local_load_store']==row['sass_matrix_instructions']==0)
    check('17 focused tests','17 passed' in Path('results/norm_gemv_multi_tests.txt').read_text())
    package=read('residual_async_package')
    check('33 supported runtime files unchanged',len(package['records'])==33 and all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in package['records']))
    report=read('full_codec_roundtrip')
    check('roundtrip model and scope',report['revision']==REVISION and report['sampling_rate']==24000 and report['channels']==1 and report['dtype']=='float32' and not report['tf32'] and report['quantizers']==32)
    check('roundtrip protocol',report['all_exact'] and report['rounds']==3 and report['graph_warmups']==200 and len(report['cases'])==2)
    for row in report['cases']:
        label=f'roundtrip {row["frames"]}'
        check(label+' duration',row['batch']==1 and row['audio_seconds']==row['frames']*.08 and len(row['rounds'])==9)
        check(label+' bits',all(c['bits_equal'] for t in row['rounds'] for k in ('checks','restored') for c in t[k]))
        for mode,median in row['medians_ms'].items():
            check(label+mode+' median',median==statistics.median(v for t in row['rounds'] if t['mode']==mode for v in t['timing']['wall_ms_samples']))
        check(label+' speedups',all(abs(v-row['medians_ms'][m]/row['medians_ms']['optimized_graph'])<1e-12 for m,v in row['speedups'].items()))
    result={'scope':'multi-output normalization/projection research rejected for production; supported runtime unchanged; direct encode-decode timing audit','previous_commit':'beb5c32','all_passed':True,'check_count':len(checks),'checks':checks,'source_sha256':source}
    Path('results/norm_gemv_multi_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',len(checks),'checks')

if __name__=='__main__':main()
