"""Cross-check residual GEMV research reports without executing GPU work."""
import hashlib,json,statistics
from pathlib import Path
from benchmarks.residual_gemv_async import SOURCE
from fast_moss.loading import REVISION


def main():
    checks=[]
    def check(label,value):
        assert value,label
        checks.append(label)
    def read(name):return json.loads(Path('results/'+name+'.json').read_text())
    probe=read('residual_gemv_async_probe');ring=read('residual_gemv_async_ring')
    confirm=read('residual_gemv_async_confirm');resources=read('residual_gemv_async_resources')
    model=read('full_residual_gemv_async');paired=read('full_residual_gemv_async_paired')
    source_hash=hashlib.sha256(SOURCE.encode()).hexdigest()
    for name,report in [('probe',probe),('ring',ring),('confirm',confirm),('model',model),('paired',paired)]:
        check(name+' exact',report['all_exact'])
        if 'revision' in report:check(name+' revision',report['revision']==REVISION)
    for name,report in [('probe',probe),('ring',ring),('confirm',confirm),('resources',resources)]:
        check(name+' source',report['source_sha256']==source_hash)
    check('188 exploratory schedules',sum(len(r['trials']) for r in probe['records'])==188)
    for row in probe['records']:
        check(str(row['shape'])+' exploratory bits',all(t['bits_equal'] for t in row['trials']))
    check('17 finalist schedules',sum(len(r['configs']) for r in confirm['records'])==17)
    check('532 eager/graph checks',sum(len(r['checks'])*2 for r in confirm['records'])==532)
    for row in confirm['records']:
        check(str(row['shape'])+' finalist bits',all(t['eager'] and t['graph'] for t in row['checks']))
        for r in row['rings']:
            check(str(row['shape'])+str(r['length'])+' sample count',len(r['rounds'])==5*(len(row['configs'])+1))
            for name,value in r['medians_ms'].items():
                check(str(row['shape'])+str(r['length'])+name+' median',value==statistics.median(t['per_call_ms'] for t in r['rounds'] if t['backend']==name))
            check(str(row['shape'])+str(r['length'])+' ratios',all(abs(v-r['medians_ms']['current']/r['medians_ms'][k])<1e-12 for k,v in r['speedups'].items()))
    check('17 audited binaries',len(resources['records'])==17)
    for row in resources['records']:
        key=str(row['shape'])+str(row['config'])
        check(key+' binary',row['matches_confirmation'])
        check(key+' async instructions',bool(row['sass_async_copies'])==bool(row['config'][5]))
        check(key+' no spills or matrix instructions',row['resources']['local_bytes']==row['sass_local_load_store']==row['sass_matrix_instructions']==0)
    check('48 full checkpoint cases',len(model['cases'])==48)
    check('selected FFN only',model['configs']=={'5120':[64,320,2,0,16,1,2]}==paired['configs'])
    for row in model['cases']:
        eligible=row['shape']==[1,1,1920]
        check(row['name']+' dispatch',(row['candidate_calls']>0)==eligible)
    for row in model['timings']:
        eligible=row['batch']==row['frames']==1
        for record in row['rounds']:
            for direction,result in record['results'].items():
                check(str((row['batch'],row['frames'],record['round'],record['backend'],direction))+' dispatch',(result['candidate_capture_calls']>0)==(eligible and record['backend']=='candidate'))
        check(str((row['batch'],row['frames']))+' model ratios',all(abs(v-row['medians_ms'][d]['current']/row['medians_ms'][d]['candidate'])<1e-12 for d,v in row['speedups'].items()))
    for row in paired['records']:
        check(str((row['batch'],row['frames']))+' stable lifetime',row['stable_storage'] and row['restored_bits_equal'] and len(row['samples'])==160)
        check(str((row['batch'],row['frames']))+' paired dispatch',(row['counts']['candidate']['candidate_calls']>0)==(row['batch']==row['frames']==1) and row['counts']['current']['candidate_calls']==0)
    cute=read('full_residual_gemv_async_cute')
    check('48 CuTe cases',len(cute['cases'])==48 and cute['all_exact'])
    for batch in (1,2):
        stream=read(f'full_residual_gemv_async_streaming_b{batch}')
        prior=read(f'full_norm_async_streaming_b{batch}')
        check(f'stream b{batch} exact',stream['graph_exact'] and stream['frames']==162)
        check(f'stream b{batch} existing offline difference',stream['offline_exact']==prior['offline_exact'])
        for direction,row in stream['results'].items():
            check(f'stream b{batch} {direction} chunks',len(row['per_chunk_bits_equal'])==162 and all(row['per_chunk_bits_equal']))
            check(f'stream b{batch} {direction} elements',row['graph_vs_corrected_eager']['elements']==prior['results'][direction]['graph_vs_corrected_eager']['elements'])
        check(f'stream b{batch} dispatch',all((r['candidate_calls']>0)==(batch==1) for r in stream['candidate_counts']))
    profile=read('full_residual_gemv_async_profile')
    for direction,total in [('encode',1184),('decode',676)]:
        summary=profile['results'][direction]['kernel_summary']
        check(direction+' profile launches',summary['kernels_per_replay']==total and summary['residual_gemv_async_per_replay']==32 and summary['ffn_gemv_per_replay']==32 and summary['norm_gemv_async_per_replay']==64)
        check(direction+' disjoint matrix accounting',abs(sum(v['ms_per_replay'] for v in summary['small_matrix_breakdown'].values())-summary['groups']['small_matrix']['ms_per_replay'])<1e-10)
    package=read('norm_async_package')
    check('32 production files',len(package['records'])==32)
    for row in package['records']:
        check(row['path']+' unchanged',hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'])
    check('15 focused tests','15 passed' in Path('results/residual_gemv_async_tests.txt').read_text())
    check('1055 full suite tests','1055 passed' in Path('results/residual_gemv_async_full_tests.txt').read_text())
    report={'scope':'residual GEMV research cross-report audit; production unchanged','previous_commit':'b2a1086','all_passed':True,'check_count':len(checks),'checks':checks,'source_sha256':source_hash}
    Path('results/residual_gemv_async_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(checks),'checks')

if __name__=='__main__':main()
