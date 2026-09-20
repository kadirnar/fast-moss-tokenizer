"""Profile the steady-selected research fusion through the existing graph harness."""
from contextlib import contextmanager
import gzip,json,sys
from pathlib import Path
from benchmarks import profile_graph as profile
from benchmarks.norm_gemv_model import selected


def main():
    records=json.loads(Path('results/norm_gemv_steady.json').read_text())['records'];configs={}
    for row in records:
        speeds=row['rings'][-1]['speedups'];name=max(speeds,key=speeds.get)
        if speeds[name]>1.005:configs[row['mode']]=tuple(json.loads(name))
    original=profile.optimized;original_summary=profile.kernel_summary
    @contextmanager
    def optimized(model,**kwargs):
        with original(model,**kwargs),selected(model,configs):yield
    def summary(path,replays=5):
        result=original_summary(path,replays)
        trace=json.loads(gzip.open(path,'rt').read());events=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('name')=='norm_gemv']
        ms=sum(e['dur'] for e in events)/replays/1000;pct=100*ms/result['total_ms_per_replay']
        result['norm_gemv_per_replay']=len(events)/replays
        result['small_matrix_breakdown']['norm_gemv']={'ms_per_replay':ms,'percent':pct}
        result['groups']['small_matrix']['ms_per_replay']+=ms;result['groups']['small_matrix']['percent']+=pct
        result['small_matrix_breakdown_scope']+=' norm_gemv also includes fused LayerNorm and optional GELU.'
        return result
    profile.optimized=optimized;profile.kernel_summary=summary
    try:profile.main()
    finally:profile.optimized=original;profile.kernel_summary=original_summary
    path=Path(sys.argv[sys.argv.index('--output')+1]);report=json.loads(path.read_text())
    report.update(scope='research-only fused normalization/projection graph',norm_gemv_configs=configs,production_runtime_changed=False)
    path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
