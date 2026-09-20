"""Run streaming/profile gates with the selected research shared-staging helper."""
from contextlib import contextmanager
import gzip,json,sys
from pathlib import Path
from benchmarks.residual_gemv_async_model import selected


def main():
    mode=sys.argv.pop(1)
    if mode=='profile':from benchmarks import profile_graph as harness
    elif mode=='stream':from benchmarks import streaming_fidelity as harness
    else:raise ValueError('Expected profile or stream')
    selection=json.loads(Path('results/residual_gemv_async_confirm.json').read_text());configs={}
    for row in selection['records']:
        ratios=row['rings'][-1]['speedups'];best=max(ratios,key=ratios.get)
        if row['shape'][2]==5120 and ratios[best]>1.005:configs[5120]=tuple(json.loads(best))
    counts=[];original=harness.optimized
    @contextmanager
    def optimized(model,**kwargs):
        with original(model,**kwargs),selected(model,configs) as current:
            yield
            counts.append(dict(current))
    harness.optimized=optimized
    if mode=='profile':
        original_summary=harness.kernel_summary
        def summary(path,replays=5):
            result=original_summary(path,replays)
            trace=json.loads(gzip.open(path,'rt').read())
            events=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('name')=='residual_gemv_async']
            ms=sum(e['dur'] for e in events)/replays/1000
            percent=100*ms/result['total_ms_per_replay'] if result['total_ms_per_replay'] else 0.
            result['residual_gemv_async_per_replay']=len(events)/replays
            result['small_matrix_breakdown']['residual_gemv_async']={'ms_per_replay':ms,'percent':percent}
            result['groups']['small_matrix']['ms_per_replay']+=ms
            result['groups']['small_matrix']['percent']+=percent
            result['small_matrix_breakdown_scope']+=' Research residual_gemv_async includes FFN contraction and scaled residual; counted once in small_matrix.'
            return result
        harness.kernel_summary=summary
    try:harness.main()
    finally:
        harness.optimized=original
        if mode=='profile':harness.kernel_summary=original_summary
    path=Path(sys.argv[sys.argv.index('--output')+1]);report=json.loads(path.read_text())
    report.update(research_residual_gemv_async=True,production_runtime_changed=False,configs=configs,candidate_counts=counts)
    path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
