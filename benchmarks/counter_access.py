"""Read-only CUPTI range-profiler access probe; never changes driver policy."""
import ctypes as C
import json
import os
from pathlib import Path
import torch
import nvidia.cuda_cupti
from cuda.bindings import driver as cu


class Initialize(C.Structure):
    _fields_=[('structSize',C.c_size_t),('pPriv',C.c_void_p)]


class Enable(C.Structure):
    _fields_=[('structSize',C.c_size_t),('pPriv',C.c_void_p),('ctx',C.c_void_p),('obj',C.c_void_p)]


class Disable(C.Structure):
    _fields_=[('structSize',C.c_size_t),('pPriv',C.c_void_p),('obj',C.c_void_p)]


class Availability(C.Structure):
    _fields_=[('structSize',C.c_size_t),('pPriv',C.c_void_p),('ctx',C.c_void_p),
              ('imageSize',C.c_size_t),('image',C.c_void_p)]


def main():
    context_tensor=torch.empty(1,device='cuda')
    result,ctx=cu.cuCtxGetCurrent()
    if int(result):raise RuntimeError(result)
    library=Path(nvidia.cuda_cupti.__path__[0])/'lib/libcupti.so.12'
    lib=C.CDLL(str(library));lib.cuptiGetResultString.argtypes=[C.c_int,C.POINTER(C.c_char_p)]
    def name(code):
        text=C.c_char_p();lib.cuptiGetResultString(code,C.byref(text))
        return text.value.decode() if text.value else str(code)
    init=Initialize(C.sizeof(Initialize),None);rc=lib.cuptiProfilerInitialize(C.byref(init))
    report={'scope':'CUPTI permission probe only; no counters collected or driver settings changed',
        'previous_commit':'21cb936','gpu':torch.cuda.get_device_name(),'uid':os.getuid(),
        'initialize':{'code':rc,'name':name(rc)},
        'driver_policy':[line for line in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'Profil' in line],
        'capabilities':[line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('Cap')]}
    if rc==0:
        params=Enable(C.sizeof(Enable),None,int(ctx),None)
        rc=lib.cuptiRangeProfilerEnable(C.byref(params))
        report['enable']={'code':rc,'name':name(rc)}
        if rc==0:
            disable=Disable(C.sizeof(Disable),None,params.obj)
            report['disable_code']=lib.cuptiRangeProfilerDisable(C.byref(disable))
    if report.get('disable_code',0)==0 and report['initialize']['code']==0:
        availability=Availability(C.sizeof(Availability),None,int(ctx),0,None)
        rc=lib.cuptiProfilerGetCounterAvailability(C.byref(availability))
        report['counter_availability']={'code':rc,'name':name(rc),'required_bytes':availability.imageSize}
    report['counter_availability_query_succeeded']=rc==0
    report['counters_collected']=False
    Path('results/counter_access.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
