"""Optional fixed-original-batch ATen execution; no model/source modifications.

Every execution object belongs to one complete padded validation batch. All six
feature tensors remain dynamic. Qmask, umask, lengths, dtype and device are fixed.
Numerical equivalence must be checked by the caller before production use.
"""
import torch


class _Forward(torch.nn.Module):
    def __init__(self, model, lengths):
        super().__init__()
        self.model = model
        self.lengths = tuple(lengths)

    def forward(self, t0, t1, t2, t3, visual, audio, qmask, umask):
        return self.model([t0, t1, t2, t3], qmask, umask,
                          self.lengths, audio, visual, 0)[0]


class FixedBatchExecution:
    """Run a traced full original batch; optionally replay a captured CUDA graph."""
    def __init__(self, model, tensors, lengths, cuda_graph=True):
        if model.training or any(m.training for m in model.modules()):
            raise ValueError('All original model modules must be in eval mode')
        if torch.is_grad_enabled():
            raise ValueError('Construct inside torch.no_grad()')
        self.lengths = tuple(int(x) for x in lengths)
        self.shapes = tuple(tuple(x.shape) for x in tensors[:8])
        self.dtypes = tuple(x.dtype for x in tensors[:8])
        self.devices = tuple(x.device for x in tensors[:8])
        self.qmask = tensors[6].clone()
        self.umask = tensors[7].clone()
        self.graph = None
        self.capture_error = None
        self.mode = 'jit_trace_frozen'
        self.static = tuple(x.clone() for x in tensors[:8])
        module = _Forward(model, self.lengths).eval()
        self.traced = torch.jit.trace(module, self.static, check_trace=False,
                                      strict=False)
        self.traced = torch.jit.freeze(self.traced, optimize_numerics=False)
        for _ in range(3):
            self.traced(*self.static)
        if cuda_graph and self.static[0].is_cuda:
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.traced(*self.static)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self.traced(*self.static)
                self.graph, self.output = graph, output
                self.mode = 'jit_trace_frozen_cuda_graph'
            except RuntimeError as error:
                self.capture_error = repr(error)
                # The trace remains usable; caller must validate it separately.
                self.graph = None

    def __call__(self, tensors, lengths=None):
        if torch.is_grad_enabled():
            raise ValueError('Execute inside torch.no_grad()')
        if lengths is not None and tuple(lengths) != self.lengths:
            raise ValueError('Different lengths require a new execution object')
        if tuple(tuple(x.shape) for x in tensors[:8]) != self.shapes:
            raise ValueError('Different batch shapes require a new execution object')
        if tuple(x.dtype for x in tensors[:8]) != self.dtypes:
            raise ValueError('Dtype changed')
        if tuple(x.device for x in tensors[:8]) != self.devices:
            raise ValueError('Device changed')
        if not torch.equal(tensors[6], self.qmask) or not torch.equal(tensors[7], self.umask):
            raise ValueError('Speaker or validity masks changed')
        if self.graph is None:
            return self.traced(*tuple(tensors[:8]))
        for dst, src in zip(self.static[:6], tensors[:6]):
            dst.copy_(src)
        self.graph.replay()
        # Output storage is reused on the next replay; caller consumes it now.
        return self.output

    @property
    def stats(self):
        return {'mode': self.mode, 'capture_error': self.capture_error,
                'fixed_lengths': list(self.lengths)}

    def close(self):
        self.graph = None


def build_executor(model, tensors, lengths):
    return FixedBatchExecution(model, tensors, lengths)
