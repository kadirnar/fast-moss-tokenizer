"""Run existing streaming/profile gates inside the research attention epilogue context."""
from contextlib import contextmanager
import gzip,json,sys
from pathlib import Path
from benchmarks.attention_residual import CONFIGS
from benchmarks.attention_residual_model import selected


def main():
    mode=sys.argv.pop(1)
    if mode=='profile':from benchmarks import profile_graph as harness
    elif mode=='stream':from benchmarks import streaming_fidelity as harness
    else:raise ValueError('Expected profile or stream')
    configs={json.dumps(shape):cfg for shape,cfg in CONFIGS.items()};counts=[]
    original=harness.optimized
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
            events=[e for e in json.loads(gzip.open(path,'rt').read())['traceEvents'] if e.get('cat')=='kernel' and e.get('name')=='attention_residual']
            ms=sum(e['dur'] for e in events)/replays/1000;pct=100*ms/result['total_ms_per_replay']
            result['attention_residual_cuda_per_replay']=len(events)/replays
            result['small_matrix_breakdown']['attention_residual']={'ms_per_replay':ms,'percent':pct}
            result['groups']['small_matrix']['ms_per_replay']+=ms;result['groups']['small_matrix']['percent']+=pct
            result['small_matrix_breakdown_scope']+=' Research attention residual shares the existing GEMV/fixed kernel names and adds attention_residual for CUDA.'
            return result
        harness.kernel_summary=summary
    try:harness.main()
    finally:
        harness.optimized=original
        if mode=='profile':harness.kernel_summary=original_summary
    path=Path(sys.argv[sys.argv.index('--output')+1]);report=json.loads(path.read_text())
    report.update(research_attention_residual=True,production_runtime_changed=False,attention_residual_configs=configs,candidate_counts=counts)
    path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
