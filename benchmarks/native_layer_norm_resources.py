"""Audit the integrated LayerNorm compiler output with its production flags."""
import hashlib
import json
from pathlib import Path
import torch
from fast_moss.normalization import CONFIGS, SOURCE, _compile, _check, compiler
from fast_moss.loading import strict_precision, REVISION


def main():
    strict_precision();context_tensor = torch.empty(1, device='cuda')
    bindings = compiler();cu, nvrtc = bindings
    device = torch.device('cuda', torch.cuda.current_device())
    report = {'scope': 'integrated LayerNorm compiler resource audit', 'revision': REVISION,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest(), 'records': []}
    for n, config in sorted({(shape[1], config) for shape, config in CONFIGS.items()}):
        _, fn = _compile(device, n, config, bindings)
        register, threads, unroll = config
        definitions = dict(N=n, REGISTER=int(register), THREADS=threads, UNROLL=unroll)
        source = ('\n'.join(f'#define {k} {v}' for k, v in definitions.items())+'\n'+SOURCE).encode()
        program = _check(nvrtc.nvrtcCreateProgram(source, b'layer_norm.cu', 0, [], []))
        options = [b'--gpu-architecture=sm_120', b'--std=c++17', b'--ftz=false', b'--fmad=true']
        try:
            _check(nvrtc.nvrtcCompileProgram(program, len(options), options))
            ptx = b' '*_check(nvrtc.nvrtcGetPTXSize(program))
            _check(nvrtc.nvrtcGetPTX(program, ptx));ptx = ptx.decode()
            blob = b' '*_check(nvrtc.nvrtcGetCUBINSize(program))
            _check(nvrtc.nvrtcGetCUBIN(program, blob))
        finally:_check(nvrtc.nvrtcDestroyProgram(program))
        attr = cu.CUfunction_attribute
        record = {'width': n, 'config': config, 'flags': [o.decode() for o in options],
                  'cubin_sha256': hashlib.sha256(blob).hexdigest(),
                  'fp32_fma_instructions': ptx.count('fma.rn.f32'),
                  'matrix_instructions': sum(ptx.count(s) for s in ('mma.sync','wgmma.','tcgen05.mma'))}
        for name, value in [('registers', attr.CU_FUNC_ATTRIBUTE_NUM_REGS),
                            ('shared_bytes', attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
                            ('local_bytes', attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)]:
            record[name] = _check(cu.cuFuncGetAttribute(value, fn))
        report['records'].append(record)
    Path('results/native_layer_norm_resources.json').write_text(json.dumps(report, indent=2)+'\n')
    assert all(r['local_bytes'] == r['matrix_instructions'] == 0 for r in report['records'])
    print(report)


if __name__ == '__main__':main()
