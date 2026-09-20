import pytest
import torch
from benchmarks.fixtures import structural_model
from fast_moss.streaming import StreamingSession


@torch.inference_mode()
@pytest.mark.parametrize('direction',['encode','decode'])
@pytest.mark.parametrize('batch',[1,3,17])
def test_reset_matches_upstream_and_preserves_unselected_lanes(direction,batch):
    model=structural_model()
    with StreamingSession(model,direction,batch,fast_reset=True,use_graph=False) as session:
        plan=session._reset_plan
        assert plan is not None
        for i,tensor in enumerate(plan.offsets):tensor.copy_(torch.arange(batch,device='cuda')+2**40+i)
        for state in plan.states:
            if hasattr(state,'offset_cpu'):state.offset_cpu=777
        session._active.copy_(torch.arange(batch,device='cuda')%2==0)
        session._closed.fill_(True)
        before=[x.clone() for x in plan.offsets]
        original_active=session._active.clone();original_closed=session._closed.clone()
        mask=(torch.arange(batch*2,device='cuda')%3==0)[::2]
        # Run actual upstream state methods as oracle, including CPU bookkeeping.
        for state in plan.states:state.reset(mask)
        session._closed.masked_fill_(mask,False)
        expected=[x.clone() for x in plan.offsets]
        expected_active=session._active.clone();expected_closed=session._closed.clone()
        for tensor,value in zip(plan.offsets,before):tensor.copy_(value)
        session._active.copy_(original_active);session._closed.copy_(original_closed)
        for state in plan.states:
            if hasattr(state,'offset_cpu'):state.offset_cpu=777
        session.reset(mask)
        assert all(torch.equal(a,b) for a,b in zip(plan.offsets,expected))
        assert torch.equal(session._active,expected_active)
        assert torch.equal(session._closed,expected_closed)
        assert all(getattr(s,'offset_cpu',0)==0 for s in plan.states)
    assert session._reset_plan is None
