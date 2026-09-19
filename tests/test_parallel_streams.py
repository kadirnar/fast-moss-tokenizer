import pytest
import torch
from benchmarks.fixtures import structural_model
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession


@torch.inference_mode()
@pytest.mark.parametrize("direction", ["encode", "decode"])
def test_lanes_follow_independent_timelines(direction):
    model = structural_model()
    shape = (2,1,1920) if direction == "encode" else (32,2,1)
    torch.manual_seed(21)
    chunks = [torch.randn(shape,device="cuda")*.05 if direction=="encode"
              else torch.randint(1024,shape,device="cuda") for _ in range(5)]
    lane_dim = 1 if direction == "encode" else 0
    with optimized(model, residual_backend="triton", kv_backend="triton",
                   rope_backend="triton",share_rope_tables=True):
        with StreamingSession(model,direction,batch_size=2,use_graph=False) as session:
            ref0=[session.push(chunk)[0].clone() for chunk in chunks]
        with StreamingSession(model,direction,batch_size=2,use_graph=False) as session:
            ref1=[session.push(chunks[i])[0].clone() for i in [0,2,4]]
        mask=torch.tensor([True,False],device="cuda")
        with StreamingSession(model,direction,batch_size=2) as session:
            actual=[]
            for i,chunk in enumerate(chunks):
                out,lengths=session.push(chunk,active_mask=mask if i%2 else None)
                actual.append(out)
                assert torch.equal(out.select(lane_dim,0),ref0[i].select(lane_dim,0))
                if i%2:
                    assert lengths[1].item()==0
                else:
                    assert torch.equal(out.select(lane_dim,1),ref1[i//2].select(lane_dim,1))
            # Reuse lane 1 for a fresh stream without resetting lane 0.
            session.reset(torch.tensor([False,True],device="cuda"))
            reset_out,_=session.push(chunks[0])
            assert torch.equal(reset_out.select(lane_dim,1),ref1[0].select(lane_dim,1))
            # Earlier outputs remain owned after multiple replays and a partial reset.
            assert torch.equal(actual[0],ref0[0])
