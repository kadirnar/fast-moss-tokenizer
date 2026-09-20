"""Inspect generated SASS for the research GEMV load widths (own kernels only)."""
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import torch
import triton
from cuda.bindings import nvrtc

from benchmarks.gemv_vector import compile_kernel
from benchmarks.small_vector_cuda import check


def main():
    # Materialize PyTorch's primary context before calling the CUDA driver API.
    context_tensor=torch.empty(1,device='cuda')
    search=json.loads(Path('results/gemv_vector_confirm.json').read_text())
    selected={(tuple(r['shape']),tuple(c)) for r in search['records'] for c in r['configs']}
    # Explicit wide-load examples supplement the finalists, whose winners may be scalar.
    selected.update(((1,5120,1280),(8,v,32,2,4)) for v in (4,8))
    executable=Path(triton.__file__).parent/'backends/nvidia/bin/cuobjdump'
    records=[]
    with tempfile.TemporaryDirectory(prefix='moss-vector-sass-') as tmp:
        for shape,config in sorted(selected):
            _,_,resources,blob=compile_kernel(shape[1],shape[2],config)
            path=Path(tmp)/'kernel.cubin';path.write_bytes(blob)
            sass=subprocess.check_output([str(executable),'--dump-sass',str(path)],text=True)
            instructions=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',sass))
            records.append({'shape':shape,'config':config,'resources':resources,
                'cubin_sha256':hashlib.sha256(blob).hexdigest(),
                'instruction_counts':dict(sorted(instructions.items())),
                'wide_global_loads':sum(count for op,count in instructions.items()
                    if op.startswith('LDG.') and ('.64' in op or '.128' in op)),
                'matrix_instructions':sum(count for op,count in instructions.items()
                    if any(token in op for token in ('MMA','TCGEN'))),
                'local_load_store_instructions':sum(count for op,count in instructions.items()
                    if op.startswith(('LDL','STL')))})
    report={'scope':'generated own-kernel SASS, finalists and two explicit wide-load examples',
        'architecture':'sm_120','nvrtc_version':list(check(nvrtc.nvrtcVersion())),
        'source_sha256':hashlib.sha256(Path('benchmarks/gemv_vector.py').read_bytes()).hexdigest(),
        'enabled_in_runtime':False,'records':records}
    Path('results/gemv_vector_instructions.json').write_text(json.dumps(report,indent=2)+'\n')
    print('inspected',len(records),'kernels;',sum(r['wide_global_loads']>0 for r in records),'with wide loads')


if __name__=='__main__':main()
