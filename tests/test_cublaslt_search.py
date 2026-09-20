"""Exercise the expanded ABI and serialized capability-search configurations."""
import itertools
import torch
from benchmarks.cublaslt import LinearPlan,library
from benchmarks.cublaslt_search import candidates
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def test_capability_candidates_restore_and_replay_in_fp32():
    torch.manual_seed(1238)
    x=torch.randn(24,1280,device='cuda');w=torch.randn(5120,1280,device='cuda')
    gold=(x.double()@w.double().T).float()
    with LinearPlan(w,24,'packed') as plan:
        found=list(itertools.islice(candidates(plan,[1,5]),4))
        assert len(found)==4
        for result,configuration in found:
            assert configuration['numerical_flags']&0xffffff==0x080201
            plan.algorithms.append(result);index=len(plan.algorithms)-1
            restored=plan.restore(plan.metadata(index),library().cublasLtGetVersion())
            out=plan(x,restored)
            torch.testing.assert_close(out,gold,atol=2e-4,rtol=2e-5)
            graph=GraphedCallable(lambda z:(plan(z,restored),),x)
            assert torch.equal(out,graph(x)[0])
            del graph
