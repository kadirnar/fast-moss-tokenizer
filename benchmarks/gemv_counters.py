"""Collect actual CUPTI memory counters when permitted, recording failures exactly."""
import ctypes as C
import json
import subprocess
import tempfile
from pathlib import Path
import torch
import nvidia.cuda_cupti
import nvidia.cuda_runtime
from cuda.bindings import driver as cu
from fast_moss.loading import strict_precision
from fast_moss.small_matrices import linear


@torch.inference_mode()
def main():
    strict_precision();inputs=torch.load('results/matrix_inputs.pt',weights_only=True)
    _,ctx=cu.cuCtxGetCurrent()
    root=Path(nvidia.cuda_cupti.__path__[0]);runtime=Path(nvidia.cuda_runtime.__path__[0])
    report={'scope':'CUPTI user ranges over GEMV graphs; all profiling setup/collection separate from latency timing',
            'previous_commit':'21cb936','gpu':torch.cuda.get_device_name(),'records':[],'counters_collected':False}
    out=Path('results/gemv_counters.json')
    with tempfile.TemporaryDirectory(prefix='moss-counters-') as tmp:
        target=Path(tmp)/'bridge.so'
        subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','benchmarks/range_counter_bridge.cpp',
            '-I'+str(root/'include'),'-I'+str(runtime/'include'),'-L'+str(root/'lib'),
            '-Wl,-rpath,'+str(root/'lib'),'-l:libcupti.so.12','-o',str(target)],check=True)
        lib=C.CDLL(str(target));lib.counter_error.restype=C.c_char_p;lib.counter_warning.restype=C.c_char_p
        lib.counter_setup.argtypes=[C.c_void_p,C.POINTER(C.c_char_p),C.c_size_t]
        lib.counter_values.argtypes=[C.POINTER(C.c_double)]
        names=['dram__bytes_read.sum','dram__bytes_write.sum','lts__t_sector_hit_rate.pct',
               'sm__throughput.avg.pct_of_peak_sustained_elapsed']
        encoded=(C.c_char_p*len(names))(*(name.encode() for name in names))
        for shape in [(1,5120,1280),(1,1280,5120)]:
            x,w=inputs[shape]['x'],inputs[shape]['weight'];weights=[w.clone() for _ in range(32)]
            for length in (1,32):
                for t in weights[:length]:linear(x,t)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs=[linear(x,t) for t in weights[:length]]
                graph.replay();torch.cuda.synchronize()
                rc=lib.counter_setup(int(ctx),encoded,len(names))
                record={'shape':shape,'length':length,'logical_weight_bytes':length*w.numel()*4,'metrics':names,
                        'setup_warning':lib.counter_warning().decode()}
                try:
                    if rc<0:raise RuntimeError(lib.counter_error().decode())
                    for passes in range(1,21):
                        if lib.counter_start()<0:raise RuntimeError(lib.counter_error().decode())
                        graph.replay();torch.cuda.synchronize()
                        status=lib.counter_stop()
                        if status<0:raise RuntimeError(lib.counter_error().decode())
                        if status==1:break
                    else:raise RuntimeError('CUPTI exceeded twenty passes')
                    values=(C.c_double*len(names))()
                    if lib.counter_values(values)<0:raise RuntimeError(lib.counter_error().decode())
                    record.update(passes=passes,values=dict(zip(names,values)));report['counters_collected']=True
                except RuntimeError as e:
                    record['error']=str(e);report['records'].append(record)
                    out.write_text(json.dumps(report,indent=2)+'\n');print(record,flush=True);return
                finally:lib.counter_close()
                report['records'].append(record);out.write_text(json.dumps(report,indent=2)+'\n')
                print(record,flush=True);del graph,outputs

if __name__=='__main__':main()
