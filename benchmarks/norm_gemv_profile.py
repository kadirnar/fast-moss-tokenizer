"""Profile the steady-selected research fusion through the existing graph harness."""
from contextlib import contextmanager
import json,sys
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
        result['norm_gemv_per_replay']=result['norm_projection_per_replay']
        return result
    profile.optimized=optimized;profile.kernel_summary=summary
    try:profile.main()
    finally:profile.optimized=original;profile.kernel_summary=original_summary
    path=Path(sys.argv[sys.argv.index('--output')+1]);report=json.loads(path.read_text())
    report.update(scope='research-only fused normalization/projection graph',norm_gemv_configs=configs,production_runtime_changed=False)
    path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
