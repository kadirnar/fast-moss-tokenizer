"""Audit the supported residual staging integration from recorded evidence."""
import hashlib,json,re
from pathlib import Path
from benchmarks.residual_gemv_async import SOURCE as RESEARCH_SOURCE
from fast_moss.residual_async import CONFIG,SOURCE


def main():
    checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    def read(name):return json.loads(Path('results',name+'.json').read_text())
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    check('source matches research',SOURCE==RESEARCH_SOURCE)
    source=hashlib.sha256(SOURCE.encode()).hexdigest()
    selection=read('residual_gemv_async_confirm');selected={}
    for row in selection['records']:
        ratios=row['rings'][-1]['speedups'];best=max(ratios,key=ratios.get)
        if row['shape'][2]==5120 and ratios[best]>1.005:selected[5120]=tuple(json.loads(best))
    check('production schedule matches selected',selected=={5120:CONFIG})
    configs={'5120':list(CONFIG)}
    for name in ['full_residual_async_runtime','full_residual_async_cute']:
        r=read(name)
        check(name+' corpus',r['all_exact'] and len(r['cases'])==48 and r['enabled_in_runtime'] and r['configs']==configs)
        check(name+' output bits',all(v['bits_equal'] for case in r['cases'] for mode in case['checks'].values() for v in mode))
        check(name+' candidate exercised',sum(v['candidate_calls'] for v in r['cases'])>0)
        check(name+' shape restriction',all((v['candidate_calls']>0)==(v['shape']==[1,1,1920]) for v in r['cases']))
    r=read('full_residual_async_runtime')
    check('context protocol',r['timing_rounds']==7 and r['extra_warmup_replays']==200 and len(r['timings'])==4)
    for t in r['timings']:
        check('round count '+str((t['batch'],t['frames'])),len(t['rounds'])==14)
        for row in t['rounds']:
            label=str((t['batch'],t['frames'],row['round'],row['backend']))
            check('timing dispatch '+label,(row['candidate_calls']>0)==(row['backend']=='candidate' and (t['batch'],t['frames'])==(1,1)))
            check('timing bits '+label,all(c['bits_equal'] for v in row['results'].values() for c in v['checks']))
        for name,v in t['medians_ms'].items():check('timing ratio '+str((t['batch'],t['frames'],name)),abs(t['speedups'][name]-v['current']/v['candidate'])<1e-12)
    r=read('full_residual_async_paired')
    check('stable pairs',r['all_exact'] and r['enabled_in_runtime'] and len(r['records'])==3 and r['configs']==configs)
    for row in r['records']:
        label=str((row['batch'],row['frames']))
        check('paired storage '+label,row['stable_storage'] and row['restored_bits_equal'] and len(row['samples'])==160)
        check('paired dispatch '+label,row['counts']['current']['candidate_calls']==0 and (row['counts']['candidate']['candidate_calls']>0)==((row['batch'],row['frames'])==(1,1)))
        for name,v in row['medians_ms'].items():
            check('paired ratio '+str((row['batch'],row['frames'],name)),abs(row['speedups'][name]-v['current']/v['candidate'])<1e-12 and len(row['paired_speedups'][name])==40 and row['candidate_wins'][name]==sum(x>1 for x in row['paired_speedups'][name]))
    for batch in (1,2):
        r=read(f'full_residual_async_streaming_b{batch}');old=read(f'full_norm_async_streaming_b{batch}')
        check('stream bits '+str(batch),r['graph_exact'] and r['frames']==162 and all(len(v['per_chunk_bits_equal'])==162 and all(v['per_chunk_bits_equal']) for v in r['results'].values()))
        check('stream dispatch '+str(batch),all(v['residual_async_calls']==(128 if batch==1 else 0) and v['norm_async_calls']==(256 if batch==1 else 0) for v in r['results'].values()))
        check('stream element counts '+str(batch),r['results']['encode']['graph_vs_corrected_eager']['elements']==5184*batch and r['results']['decode']['graph_vs_corrected_eager']['elements']==311040*batch)
        check('stream original offline discrepancy '+str(batch),all(v['corrected_eager_vs_offline']==old['results'][k]['corrected_eager_vs_offline'] for k,v in r['results'].items()))
    for frames in (1,3):
        r=read(f'full_residual_async_profile_f{frames}');old=read(f'full_norm_async_profile_f{frames}')
        for name,v in r['results'].items():
            s=v['kernel_summary'];prev=old['results'][name]['kernel_summary']
            check('profile replacement '+str((frames,name)),s['kernels_per_replay']==prev['kernels_per_replay'] and s['residual_gemv_async_per_replay']==(32 if frames==1 else 0) and s['norm_gemv_async_per_replay']==prev['norm_gemv_async_per_replay'] and s['ffn_gemv_per_replay']==prev['ffn_gemv_per_replay']-(32 if frames==1 else 0))
            check('profile disjoint grouping '+str((frames,name)),abs(sum(v['ms_per_replay'] for v in s['small_matrix_breakdown'].values())-s['groups']['small_matrix']['ms_per_replay'])<1e-9)
        r=read(f'full_codec_residual_async_f{frames}')
        check('fresh direct comparison '+str(frames),r['all_exact'] and r['extra_graph_warmup_replays']==200 and len(r['cases'])==2)
        for case in r['cases']:
            for name,v in case['medians_ms'].items():
                for mode in ('original_eager','original_graph'):
                    check('fresh ratio '+str((frames,case['batch'],name,mode)),abs(case['speedups'][name][mode]-v[mode]/v['optimized_graph'])<1e-12)
    r=read('residual_async_resources');old=read('residual_gemv_async_resources')
    check('resource source',r['source_sha256']==source and len(r['records'])==17 and r['enabled_in_runtime'])
    for row in r['records']:
        check('SASS '+str((row['shape'],row['config'])),row in old['records'] and row['matches_confirmation'] and bool(row['sass_async_copies'])==bool(row['config'][5]) and row['sass_local_load_store']==row['sass_matrix_instructions']==0)
    r=read('residual_async_package')
    check('33 package files and isolated import',r['all_match'] and r['isolated_import'] and len(r['records'])==33)
    check('package hashes',all(v['wheel_match'] and digest(v['path'])==v['sha256'] for v in r['records']))
    check('initial regression suite',bool(re.search('102 passed in',Path('results/residual_async_initial_tests.txt').read_text())))
    check('focused suite',bool(re.search('24 passed in',Path('results/residual_async_runtime_tests.txt').read_text())))
    check('full suite',bool(re.search('1079 passed in',Path('results/residual_async_full_tests.txt').read_text())))
    files=set(Path('results').glob('*residual_async*.json'))|set(Path('results').glob('*residual_async*.txt'))|set(Path('benchmarks').glob('residual_gemv_async*.py'))|{Path('tests/test_residual_async_runtime.py'),Path('fast_moss/residual_async.py'),Path('fast_moss/matrices.py')}
    report={'scope':'supported asynchronous residual GEMV integration cross-report audit','previous_production_commit':'b2a1086','integration_parent':'c4f560a','all_passed':True,'check_count':len(checks),'checks':checks,'sha256':{str(p):digest(p) for p in sorted(files) if p.name!='residual_async_audit.json'}}
    Path('results/residual_async_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(checks),'cross-report checks')

if __name__=='__main__':main()
