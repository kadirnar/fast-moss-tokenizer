"""Fixed-shape CUDA graphs. Returned tensors are owned by the caller."""
import torch
from . import storage_epoch


class GraphedCallable:
    """Capture a pure tensor function after warming up on a separate stream.

    Stateful streaming functions must reset their caches after construction.
    Calls are serialized on the CUDA stream used during construction.
    Resident matrix storage transitions invalidate managed graphs on the device.
    Warm every required matrix shape before capturing reusable graph collections.
    """

    @torch.inference_mode()
    def __init__(self, fn, *examples, warmup=3):
        if not examples or not all(t.is_cuda for t in examples):
            raise ValueError("CUDA tensor examples required")
        if any(t.device != examples[0].device for t in examples):
            raise ValueError("All graph inputs must be on one device")
        self.device = examples[0].device
        self.storage_epoch = storage_epoch.current(self.device)
        lifecycle = storage_epoch.lifecycle(self.device)
        self.stream = torch.cuda.current_stream(self.device)
        self.inputs = tuple(t.clone() for t in examples)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(self.stream)
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn(*self.inputs)
        self.stream.wait_stream(side)
        if lifecycle != storage_epoch.lifecycle(self.device):
            raise RuntimeError('Matrix storage contexts must enclose graph construction')
        # Warmup may lazily pack newly reached matrix shapes. Older graphs are
        # invalidated, while this graph captures only the resulting storage.
        self.storage_epoch = storage_epoch.current(self.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            self.outputs = fn(*self.inputs)
        self.stream.wait_stream(side)
        if self.storage_epoch != storage_epoch.current(self.device):
            raise RuntimeError('Matrix storage changed during graph capture')
        if not isinstance(self.outputs, tuple) or not all(isinstance(t, torch.Tensor) for t in self.outputs):
            raise TypeError("Captured function must return a tuple of tensors")

    @torch.inference_mode()
    def __call__(self, *args):
        if self.storage_epoch != storage_epoch.current(self.device):
            raise RuntimeError('Matrix storage changed; construct a new graph in the active context')
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("Use this graph on its construction stream")
        if len(args) != len(self.inputs):
            raise ValueError("Input count differs from capture")
        for src, dst in zip(args, self.inputs):
            if (src.shape, src.dtype, src.device) != (dst.shape, dst.dtype, dst.device):
                raise ValueError("Input shape, dtype, and device must match capture")
            dst.copy_(src)
        self.graph.replay()
        return tuple(t.clone() for t in self.outputs)
