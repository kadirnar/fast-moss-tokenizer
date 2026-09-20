"""Inspect the own-kernel SASS and compiler provenance of staging finalists."""
import argparse,hashlib,json,re,subprocess,tempfile
from collections import Counter
from pathlib import Path
import torch,triton
from benchmarks.norm_gemv_multi import compile_kernel,SOURCE


def main():
    helper=compile_kernel
    context_tensor=torch.empty(1,device='cuda')
    prior=json.loads(Path('results/norm_gemv_multi_confirm.json').read_text())
    executable=Path(triton.__file__).parent/'backends/nvidia/bin/cuobjdump';records=[]
    with tempfile.TemporaryDirectory(prefix='moss-async-sass-') as tmp:
        for row in prior['records']:
            for config in row['configs']:
                _,_,resources,blob=helper(row['shape'][1],row['mode'],tuple(config))
                assert resources==row['resources'][json.dumps(config)]
                path=Path(tmp)/'kernel.cubin';path.write_bytes(blob)
                sass=subprocess.check_output([str(executable),'--dump-sass',str(path)],text=True)
                counts=Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',sass))
                async_copies=sum(v for op,v in counts.items() if op.startswith('LDGSTS'))
                assert bool(async_copies)==bool(config[6])
                records.append({'shape':row['shape'],'mode':row['mode'],'config':config,'resources':resources,'matches_confirmation':True,
                    'sass_instructions':dict(sorted(counts.items())),'sass_async_copies':async_copies,
                    'sass_local_load_store':sum(v for op,v in counts.items() if op.startswith(('LDL','STL'))),
                    'sass_matrix_instructions':sum(v for op,v in counts.items() if any(s in op for s in ('MMA','TCGEN')))})
    report={'scope':'own-kernel SASS and byte-identical confirmation binaries','previous_commit':'beb5c32','architecture':'sm_120',
        'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'enabled_in_runtime':False,'records':records}
    Path('results/norm_gemv_multi_resources.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(records),'compiler/SASS variants;',sum(r['sass_async_copies']>0 for r in records),'with LDGSTS')

if __name__=='__main__':main()
