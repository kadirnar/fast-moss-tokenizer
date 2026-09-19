"""Fixed-shape CUDA graphs. Returned tensors are owned by the caller."""
import torch


class GraphedCallable:
    """Capture a pure tensor function after warming up on a separate stream.

    Stateful streaming functions must reset their caches after construction.
    Calls are serialized on the CUDA stream used during construction.
    """

    @torch.inference_mode()
    def __init__(self, fn, *examples, warmup=3):
        if not examples or not all(t.is_cuda for t in examples):
            raise ValueError("CUDA tensor examples required")
        if any(t.device != examples[0].device for t in examples):
            raise ValueError("All graph inputs must be on one device")
        self.device = examples[0].device
        self.stream = torch.cuda.current_stream(self.device)
        self.inputs = tuple(t.clone() for t in examples)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(self.stream)
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn(*self.inputs)
        self.stream.wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            self.outputs = fn(*self.inputs)
        self.stream.wait_stream(side)
        if not isinstance(self.outputs, tuple) or not all(isinstance(t, torch.Tensor) for t in self.outputs):
            raise TypeError("Captured function must return a tuple of tensors")

    @torch.inference_mode()
    def __call__(self, *args):
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
