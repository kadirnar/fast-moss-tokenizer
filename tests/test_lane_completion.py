import pytest
import torch
from benchmarks.fixtures import structural_model
from benchmarks.lane_fixtures import schedule,independent_reference
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession


@torch.inference_mode()
@pytest.mark.parametrize('direction',['encode','decode'])
@pytest.mark.parametrize('use_graph',[False,True])
def test_independent_tail_completion_and_reuse(direction,use_graph):
    model=structural_model();entries=schedule(direction,warm_steps=3)
    reference=independent_reference(model,direction,entries)
    dim=1 if direction=='encode' else 0
    with optimized(model,residual_backend='triton',kv_backend='triton',rope_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton'):
        with StreamingSession(model,direction,3,chunk_frames=2,use_graph=use_graph) as session:
            first_graph=None;owned=[]
            for i,entry in enumerate(entries):
                if entry['reset']:
                    session.reset(torch.tensor([b in entry['reset'] for b in range(3)],device='cuda'))
                out,lengths=session.push(entry['chunk'],valid_lengths=entry['lengths'],final_lanes=entry['ends'])
                if use_graph:
                    if first_graph is None:first_graph=session.graph
                    assert session.graph is first_graph
                for lane,ref in reference[i].items():
                    n=0 if ref is None else ref.shape[-1]
                    assert lengths[lane].item()==n
                    if n:
                        assert torch.equal(out.select(dim,lane)[...,:n],ref)
                        owned.append((out.select(dim,lane)[...,:n],ref))
            assert all(torch.equal(a,b) for a,b in owned)


@torch.inference_mode()
def test_lane_metadata_validation_and_mode_switch():
    model=structural_model();x=torch.zeros(3,1,1920,device='cuda')
    with optimized(model,kv_backend='triton'):
        with StreamingSession(model,'encode',3) as session:
            session.push(x)  # Legacy fast path captures the same reusable graph.
            graph=session.graph
            for kwargs,error in [({'valid_lengths':[1,1920,1920]},ValueError),
                                 ({'valid_lengths':[1921,0,0]},ValueError),
                                 ({'valid_lengths':[-1,0,0]},ValueError),
                                 ({'valid_lengths':[True,0,0]},TypeError),
                                 ({'valid_lengths':torch.ones(3,device='cuda')},TypeError),
                                 ({'final_lanes':[1,0,0]},TypeError)]:
                with pytest.raises(error):session.push(x,**kwargs)
            _,lengths=session.push(x,valid_lengths=[1,1920,0],final_lanes=[True,False,True])
            assert lengths.tolist()==[1,1,0]
            assert session.graph is graph
            _,lengths=session.push(x)
            assert lengths.tolist()==[0,1,0]
            session.reset(torch.tensor([True,False,False],device='cuda'))
            _,lengths=session.push(x)
            assert lengths.tolist()==[1,1,0]
            # Empty completion does not advance a lane or require a dummy token.
            offsets=[m._streaming_state.offset.clone() for m in model.modules()
                     if getattr(getattr(m,'_streaming_state',None),'offset',None) is not None]
            out,lengths=session.push(x[...,:0],valid_lengths=[0]*3,final_lanes=[True]*3)
            assert out.shape[-1]==0 and lengths.tolist()==[0]*3
            current=[m._streaming_state.offset for m in model.modules()
                     if getattr(getattr(m,'_streaming_state',None),'offset',None) is not None]
            assert all(torch.equal(a,b) for a,b in zip(offsets,current))
            session.reset()
            assert session.push(x)[1].tolist()==[1,1,1]
            assert session.graph is graph


@torch.inference_mode()
def test_lane_controls_require_mask_aware_cache():
    model=structural_model()
    with StreamingSession(model,'encode',3,use_graph=False) as session:
        with pytest.raises(ValueError,match='kv_backend'):
            session.push(torch.zeros(3,1,1920,device='cuda'),valid_lengths=[1920]*3)


@torch.inference_mode()
def test_empty_end_before_capture_and_wrong_stream_are_safe():
    model=structural_model();x=torch.zeros(3,1,1920,device='cuda')
    with optimized(model,kv_backend='triton'):
        with StreamingSession(model,'encode',3) as session:
            out,lengths=session.push(x[...,:0],valid_lengths=[0]*3,final_lanes=[True,False,False])
            assert session.graph is None and out.shape[-1]==0
            assert lengths.tolist()==[0]*3
            assert session.push(x)[1].tolist()==[0,1,1]
            assert session.push(x)[1].tolist()==[0,1,1]
            active=session._active.clone();closed=session._closed.clone()
            other=torch.cuda.Stream()
            with torch.cuda.stream(other):
                with pytest.raises(RuntimeError,match='construction stream'):
                    session.push(x,valid_lengths=[1920]*3,final_lanes=[True]*3)
                with pytest.raises(RuntimeError,match='construction stream'):
                    session.reset()
            assert torch.equal(active,session._active) and torch.equal(closed,session._closed)
