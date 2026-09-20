"""Version-gated resident FP32 cuBLASLt/Triton matrices for pinned MOSS.

Packing changes parameter storage/strides, not values or Parameter identity.
Do not retain external weight aliases or raw CUDA graphs across this context.
Managed GraphedCallable objects on the device are invalidated on entry/exit.
"""
import json
from pathlib import Path
from threading import get_ident
from types import MethodType

import torch
import torch.nn.functional as F

from .cublaslt import LinearPlan, library
from .loading import REVISION
from . import storage_epoch


def folds_to_mm(x):
    # Frozen 2-D right operand: mirror PyTorch 2.8's should_fold stride guard.
    # is_contiguous alone is insufficient for gapped singleton dimensions.
    return all(x.stride(i) == x.stride(i + 1) * x.shape[i + 1] for i in range(x.ndim - 2))


def load_profile(model):
    profile = json.loads(Path(__file__).with_name('matrix_profile.json').read_text())
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('cuBLASLt matrices require a CUDA model')
    properties = torch.cuda.get_device_properties(device)
    if (profile['torch'] != torch.__version__ or profile['gpu'] != properties.name
            or profile['capability'] != [properties.major, properties.minor]
            or profile['sm_count'] != properties.multi_processor_count
            or profile['cublaslt_version'] != library().cublasLtGetVersion()):
        raise ValueError('Matrix profile requires its recorded GPU, SM count, PyTorch, and cuBLASLt versions')
    if (type(model).__name__ != 'MossAudioTokenizerModel'
            or getattr(model.config, '_commit_hash', None) != profile['revision']
            or profile['revision'] != REVISION):
        raise ValueError('Matrix profile requires the pinned MOSS checkpoint revision')
    return profile


class MatrixRuntime:
    """Internal owner of packed parameters, per-stream workspaces, and plans.

    Enter before capturing graphs and exit after finishing model/session work.
    Calls use one host thread; workspace and plans are separate per CUDA stream.
    Other layouts, shapes, bias, gradients, and autocast use the original FP32
    weight layout through F.linear. The profile is empirical exactness evidence,
    not a claim that all pedantic algorithms reproduce PyTorch's reduction.
    backend='triton' substitutes ordered SIMT kernels for profiled attention/FFN shapes;
    it retains the same packing, fallbacks, per-stream plans and lifetime guards.
    backend='cuda' additionally uses pinned CUDA kernels for five native small-row
    shapes and six long-K shared-partial kernels. Validated small-row shapes use
    native storage without persistent workspace.
    """

    def __init__(self, model, *, backend='cublaslt', _profile=None):
        if backend not in {'cublaslt', 'triton', 'cuda'}:
            raise ValueError('Unknown matrix backend')
        if backend in {'triton', 'cuda'}:
            import triton
            if triton.__version__ != '3.4.0':
                raise ValueError('Ordered matrices require the validated Triton 3.4.0 compiler')
        self.model = model
        self.backend = backend
        self.triton_calls = 0
        self.small_calls = 0
        self.small_warmed = set()
        self.ffn_calls = 0
        self.ffn_gemv_calls = 0
        self.ffn_gemv_warmed = set()
        self.residual_async_enabled = True
        self.residual_async_calls = 0
        self.ffn_short_enabled = True
        self.ffn_short_calls = 0
        self.ffn_strided_enabled = True
        self.ffn_strided_calls = 0
        self.ffn_gemv_enabled = True
        self.ffn_staged_calls = 0
        self.ffn_enabled = True
        self.profile = load_profile(model) if _profile is None else _profile
        self.cuda_enabled = True
        self.cuda_calls = 0
        self.wide_enabled = True
        self.wide_calls = 0
        if backend == 'cuda':
            from .cuda_matrices import compiler
            self.cuda_bindings = compiler()
        self.selected = {tuple(r['shape']): r for r in self.profile['records']}
        self.device = next(model.parameters()).device
        self.thread = get_ident()
        self.packed, self.plans, self.workspaces = {}, {}, {}
        self.saved = []
        self.forwards = {}
        self.norm_gemv_enabled = True
        self.norm_gemv_calls = self.norm_qkv_calls = self.norm_ffn_calls = 0
        self.norm_gemv_warmed = set()
        self.norm_async_enabled = True
        self.norm_async_calls = self.norm_async_qkv_calls = self.norm_async_ffn_calls = 0
        self.attention_residual_enabled = True
        self.attention_residual_calls = 0
        self.attention_residual_strided_calls = 0
        self.attention_residual_warmed = set()
        self.attention_residual_shapes = {}
        self.active = self.used = False

    @torch.inference_mode()
    def __enter__(self):
        if self.used or getattr(self.model, '_fast_matrix_runtime', None) is not None:
            raise RuntimeError('Matrix runtime is already active or has been used')
        if getattr(self.model, '_fast_streaming_owner', None) is not None:
            raise RuntimeError('Enter matrix optimization before opening a streaming session')
        if getattr(self.model, '_fast_cublaslt_active', False):
            raise RuntimeError('Cannot combine supported and experimental matrix contexts')
        if (self.model.training or any(p.requires_grad or p.dtype != torch.float32
                                      or p.device != self.device for p in self.model.parameters())
                or self.device.type != 'cuda'):
            raise ValueError('Matrix optimization requires a frozen FP32 model on one CUDA device')
        if get_ident() != self.thread:
            raise RuntimeError('Use matrix runtime on its construction host thread')
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Matrix profile requires TF32 disabled')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Enter matrix optimization before CUDA graph capture')
        shapes = {shape[1:] for shape in self.selected}
        if self.backend in {'triton', 'cuda'}:
            from .ordered_matrices import SHAPES
            from .small_matrices import SHAPES as SMALL_SHAPES
            shapes.update(shape[1:] for shape in SHAPES)
            shapes.update(shape[1:] for shape in SMALL_SHAPES)
        if self.backend == 'cuda':
            from .cuda_matrices import CONFIGS as CUDA_CONFIGS
            shapes.update(shape[1:] for shape in CUDA_CONFIGS)
            from .wide_matrices import CONFIGS as WIDE_CONFIGS
            shapes.update(shape[1:] for shape in WIDE_CONFIGS)
        modules = [m for m in self.model.modules() if type(m) is torch.nn.Linear
                   and tuple(m.weight.shape) in shapes and m.bias is None and 'forward' not in m.__dict__]
        # Include registered buffers and repeated/tied parameter names. External
        # aliases cannot be enumerated and must not cross a storage transition.
        storage_owners = {}
        tensors = list(self.model.named_parameters(remove_duplicate=False))
        tensors += list(self.model.named_buffers(remove_duplicate=False))
        for name, value in tensors:
            storage_owners.setdefault(value.untyped_storage().data_ptr(), []).append(name)
        for module in modules:
            weight = module.weight
            if (not weight.is_contiguous() or weight.storage_offset() or weight.data_ptr() % 256
                    or len(storage_owners[weight.untyped_storage().data_ptr()]) != 1):
                raise ValueError('Resident matrices require aligned, contiguous, unaliased parameter storage')
        self.used = self.active = True
        self.model._fast_matrix_runtime = self
        # Synchronization makes the transition safe for already submitted work;
        # managed graphs become stale before storage is changed.
        torch.cuda.synchronize(self.device)
        storage_epoch.advance(self.device, context=True)
        try:
            with torch.cuda.device(self.device):
                for module in modules:
                    self.saved.append((module, module.forward))
                    module.forward = MethodType(self._wrap(module.forward), module)
                    self.forwards[module] = module.forward
            return self
        except BaseException:
            self.close()
            raise

    def _wrap(self, original):
        def forward(module, x, *, _fast_epilogue=None, _fast_stages=None):
            def finish(y):
                if _fast_epilogue is None:
                    return y
                mode,residual,scale,extra = _fast_epilogue
                if mode == 'attention_residual':return extra(residual,y,scale)
                return F.gelu(y) if mode == 'gelu' else residual + y*scale
            if get_ident() != self.thread or not self.active:
                raise RuntimeError('Use the active matrix runtime on its construction host thread')
            if _fast_epilogue is not None and _fast_epilogue[0] not in {'gelu','residual','attention_residual'}:
                raise ValueError('Unknown FFN epilogue')
            if x.ndim < 1 or x.shape[-1] != module.in_features:
                return finish(original(x))
            shape = (x.numel() // x.shape[-1], *module.weight.shape)
            ordered_shape = False
            small_shape = False
            cuda_shape = False
            wide_shape = False
            if self.backend in {'triton', 'cuda'}:
                from .ordered_matrices import SHAPES
                from .small_matrices import SHAPES as SMALL_SHAPES
                ordered_shape = shape in SHAPES
                small_shape = shape in SMALL_SHAPES and module.weight.is_contiguous()
            if self.backend == 'cuda' and self.cuda_enabled:
                from .cuda_matrices import CONFIGS as CUDA_CONFIGS
                cuda_shape = shape in CUDA_CONFIGS and module.weight.is_contiguous()
            if self.backend == 'cuda' and self.wide_enabled:
                from .wide_matrices import CONFIGS as WIDE_CONFIGS
                wide_shape = shape in WIDE_CONFIGS and module.weight.is_contiguous()
            if ((shape not in self.selected and not ordered_shape and not small_shape and not cuda_shape and not wide_shape) or module.bias is not None or x.device != self.device
                    or x.dtype != torch.float32 or not x.is_contiguous() or not folds_to_mm(x)
                    or x.data_ptr() % 256 or x.requires_grad or torch.is_autocast_enabled('cuda')):
                return finish(F.linear(x, module.weight.contiguous(), module.bias)
                              if id(module.weight) in self.packed else original(x))
            stream = torch.cuda.current_stream(self.device).cuda_stream
            key = (id(module), shape, stream)
            if small_shape or cuda_shape or wide_shape:
                # Native-layout kernels do not pack weights or retain workspace.
                # If a larger call later packs this parameter, the normal native
                # fallback above restores contiguous arithmetic for small calls.
                if key not in self.small_warmed:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError('Warm matrix shapes on the capture stream before capturing')
                    self.small_warmed.add(key)
                if (_fast_epilogue is not None and _fast_epilogue[0]=='attention_residual'
                        and self.backend=='cuda' and self.attention_residual_enabled):
                    from .attention_residual import CONFIGS as ATTENTION_CONFIGS,linear as attention_linear
                    if shape in ATTENTION_CONFIGS:
                        _,residual,scale,_=_fast_epilogue
                        warm=(id(module),shape,stream,tuple(residual.shape),tuple(residual.stride()))
                        if torch.cuda.is_current_stream_capturing() and warm not in self.attention_residual_warmed:
                            raise RuntimeError('Warm attention residual layout on the capture stream')
                        out=attention_linear(x.reshape(shape[0],shape[-1]),module.weight,residual,scale)
                        self.attention_residual_warmed.add(warm)
                        self.attention_residual_calls+=1
                        self.attention_residual_shapes[shape]=self.attention_residual_shapes.get(shape,0)+1
                        self.attention_residual_strided_calls+=int(not residual.is_contiguous())
                        return out
                if _fast_epilogue is not None and _fast_epilogue[0]!='attention_residual' and self.backend == 'cuda' and self.ffn_short_enabled:
                    from .short_ffn import CONFIGS as SHORT_FFN, linear as short_ffn_linear
                    if shape in SHORT_FFN and (_fast_epilogue[0] != "residual"
                            or _fast_epilogue[1].is_contiguous() or self.ffn_strided_enabled):
                        mode,residual,scale,library = _fast_epilogue
                        if mode == 'residual' and not residual.is_contiguous():
                            from .strided_ffn import linear as strided_ffn_linear
                            self.ffn_calls += 1
                            self.ffn_strided_calls += 1
                            if SHORT_FFN[shape][0] == "fixed":
                                self.small_calls += 1
                                self.triton_calls += 1
                            return strided_ffn_linear(x.reshape(shape[0],shape[-1]),module.weight,
                                residual,scale,library,self.cuda_bindings)
                        self.ffn_calls += 1
                        self.ffn_short_calls += 1
                        if SHORT_FFN[shape][0] == 'fixed':
                            self.small_calls += 1
                            self.triton_calls += 1
                        return short_ffn_linear(x.reshape(shape[0],shape[-1]),module.weight,mode,
                            residual.reshape(shape[0],module.out_features) if residual is not None else None,
                            scale,library,self.cuda_bindings).reshape(*x.shape[:-1],module.out_features)
                if wide_shape:
                    from .wide_matrices import linear as wide_linear
                    self.wide_calls += 1
                    return finish(wide_linear(x.reshape(shape[0], shape[-1]), module.weight,
                        self.cuda_bindings).reshape(*x.shape[:-1], module.out_features))
                if cuda_shape:
                    from .cuda_matrices import linear as cuda_linear
                    self.cuda_calls += 1
                    return finish(cuda_linear(x.reshape(shape[0], shape[-1]), module.weight,
                        self.cuda_bindings).reshape(*x.shape[:-1], module.out_features))
                from .small_matrices import linear as small_linear
                with torch.cuda.device(self.device):
                    if _fast_epilogue is not None and _fast_epilogue[0]!='attention_residual' and shape in ((1,5120,1280),(1,1280,5120)):
                        from .ffn import gemv_linear
                        mode,residual,scale,library = _fast_epilogue
                        asynchronous = (self.backend == 'cuda' and self.residual_async_enabled
                                        and shape == (1,1280,5120) and mode == 'residual')
                        warm = (id(module),shape,stream,mode,asynchronous)
                        if torch.cuda.is_current_stream_capturing() and warm not in self.ffn_gemv_warmed:
                            raise RuntimeError('Warm FFN GEMV backend on the capture stream')
                        values = (x.reshape(1,shape[-1]),module.weight,
                                  residual.reshape(1,module.out_features) if residual is not None else None,scale)
                        if asynchronous:
                            from .residual_async import linear as residual_linear
                            out = residual_linear(*values)
                            self.residual_async_calls += 1
                        else:
                            out = gemv_linear(values[0],values[1],mode,values[2],values[3],library)
                            self.triton_calls += 1
                        self.ffn_gemv_warmed.add(warm)
                        self.small_calls += 1
                        self.ffn_calls += 1
                        self.ffn_gemv_calls += 1
                        return out.reshape(*x.shape[:-1],module.out_features)
                    self.triton_calls += 1
                    self.small_calls += 1
                    return finish(small_linear(x.reshape(shape[0],shape[-1]),module.weight).reshape(
                        *x.shape[:-1],module.out_features))
            if key not in self.plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Warm matrix shapes on the capture stream before capturing')
                with torch.cuda.device(self.device):
                    weight = module.weight
                    if id(weight) not in self.packed:
                        # Existing managed graphs become stale before the
                        # parameter changes. Warm every needed shape before
                        # constructing a collection of reusable graphs.
                        torch.cuda.synchronize(self.device)
                        storage_epoch.advance(self.device)
                        packed = weight.T.contiguous()
                        torch.cuda.synchronize(self.device)
                        self.packed[id(weight)] = (weight, packed)
                        weight.data = packed.T
                    if stream not in self.workspaces:
                        self.workspaces[stream] = torch.empty(32 * 1024 * 1024, device=self.device, dtype=torch.uint8)
                    # Some exact ordered shapes have no useful vendor profile.
                    # Keep a stream warmup entry without manufacturing a plan;
                    # unsupported later calls still use contiguous native weights.
                    record = self.selected.get(shape)
                    plan, index = None, None
                    if record is not None:
                        plan = LinearPlan(module.weight, shape[0], 'packed', workspace=self.workspaces[stream],
                                          packed_weight=self.packed[id(module.weight)][1])
                        try:
                            index = plan.restore(record['algorithm'], self.profile['cublaslt_version'])
                        except BaseException:
                            plan.close()
                            raise
                    self.plans[key] = (plan, index)
            plan, index = self.plans[key]
            with torch.cuda.device(self.device):
                if self.backend in {'triton', 'cuda'}:
                    from .ordered_matrices import linear
                    if ordered_shape:
                        self.triton_calls += 1
                        if _fast_epilogue is not None and _fast_epilogue[0]!='attention_residual':
                            from .ffn import linear as fused_linear
                            mode,residual,scale,library = _fast_epilogue
                            self.ffn_calls += 1
                            return fused_linear(x.reshape(shape[0],shape[-1]),self.packed[id(module.weight)][1],
                                mode,residual.reshape(24,module.out_features) if residual is not None else None,
                                scale,library).reshape(*x.shape[:-1],module.out_features)
                        if _fast_stages is not None:
                            self.ffn_staged_calls += 1
                        kwargs = {} if _fast_stages is None else {'stages':_fast_stages}
                        return finish(linear(x.reshape(shape[0], shape[-1]), self.packed[id(module.weight)][1],**kwargs).reshape(
                            *x.shape[:-1], module.out_features))
                return finish(plan(x.reshape(shape[0], shape[-1]), index).reshape(*x.shape[:-1], module.out_features))
        return forward

    @property
    def packed_bytes(self):
        return sum(value.numel() * value.element_size() for _, value in self.packed.values())

    @torch.inference_mode()
    def close(self):
        if not self.active:
            return
        if get_ident() != self.thread:
            raise RuntimeError('Close matrix runtime on its construction host thread')
        torch.cuda.synchronize(self.device)
        storage_epoch.advance(self.device, context=True)
        for module, _ in reversed(self.saved):
            del module.forward
        self.saved.clear()
        self.forwards.clear()
        self.norm_gemv_warmed.clear()
        self.ffn_gemv_warmed.clear()
        self.attention_residual_warmed.clear()
        with torch.cuda.device(self.device):
            for plan, _ in self.plans.values():
                if plan is not None:
                    plan.close()
        self.plans.clear()
        self.small_warmed.clear()
        # Plans hold packed tensors: drop the final loop reference before
        # restoring one parameter at a time, retaining a single model copy.
        if 'plan' in locals():
            del plan
        self.workspaces.clear()
        while self.packed:
            key = next(iter(self.packed))
            weight, packed = self.packed[key]
            weight.data = packed.T.contiguous()
            del self.packed[key]
        del self.model._fast_matrix_runtime
        self.active = False

    def __exit__(self, *args):
        self.close()
