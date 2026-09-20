"""Heterogeneous stream schedules and equal-shape independent references."""
import torch
from fast_moss.streaming import StreamingSession


def schedule(direction, warm_steps=1, batch=3):
    if batch != 3:raise ValueError('This diagnostic schedule uses three lanes')
    width=3840 if direction=='encode' else 2
    short=777 if direction=='encode' else 1
    later=1921 if direction=='encode' else 1
    entries=[]
    for i in range(warm_steps):
        entries.append(([width,width,0],[False]*3,[]))
    entries.extend([
        ([short,width,width],[True,False,False],[]),
        ([width,0,later],[False,False,True],[]),
        ([short,width,width],[True,False,False],[0]),
        ([0,0,width],[True,True,False],[0,2]),
        ([width,width,short],[False,False,True],[]),
        ([width]*3,[False]*3,[]),
    ])
    torch.manual_seed(932)
    result=[]
    for lengths,ends,resets in entries:
        shape=(batch,1,width) if direction=='encode' else (32,batch,width)
        x=torch.randn(shape,device='cuda')*.05 if direction=='encode' else torch.randint(1024,shape,device='cuda')
        for lane,n in enumerate(lengths):
            # Invalid tails must not leak NaNs or out-of-range code indices.
            view=x[lane] if direction=='encode' else x[:,lane]
            view[...,n:]=float('nan') if direction=='encode' else -999
        result.append({'chunk':x,'lengths':lengths,'ends':ends,'reset':resets})
    return result


@torch.inference_mode()
def independent_reference(model,direction,entries):
    """Keep batch shape identical so vendor matrix choices cannot confound equality.

    A reference session follows one lane's timeline; other batch lanes repeat its
    payload. This is a numerical oracle, not a throughput/performance baseline.
    """
    batch=3;dim=1 if direction=='encode' else 0
    expected=[{} for _ in entries]
    for lane in range(batch):
        closed=False
        with StreamingSession(model,direction,batch,chunk_frames=2,use_graph=False,fast_reset=False) as session:
            for index,entry in enumerate(entries):
                if lane in entry['reset']:
                    session.reset();closed=False
                n=entry['lengths'][lane]
                end=entry['ends'][lane]
                if n and not closed:
                    x=entry['chunk']
                    part=(x[lane:lane+1,:,:n].expand(batch,-1,-1).contiguous() if direction=='encode'
                          else x[:,lane:lane+1,:n].expand(-1,batch,-1).contiguous())
                    out,lengths=session.push(part,final=end)
                    expected[index][lane]=out.select(dim,lane).clone()
                else:
                    expected[index][lane]=None
                closed |= end
    return expected
