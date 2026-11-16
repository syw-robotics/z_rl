"""Warm up, capture and replay updates with a stable input and state contract."""

import torch

from .state import _config_snapshot


class GraphedUpdate:
    """Capture after three real warm-up updates, guarding inputs and declared state.

    Inputs keep the same keys, shapes, dtypes and device until invalidate().
    state() supplies referenced tensors; config() supplies Python values used by
    the update. Configuration changes restart warm-up and capture. Callers must
    declare non-tensor dependencies through config or invalidate explicitly.
    """

    def __init__(self, function, state=None, config=None):
        self.function = function
        self.state = state
        self.config = config
        self.calls = 0
        self.graph = None
        self.stream = None
        self.identity = None
        self.configuration = None
        self.failed = False

    def _identity(self):
        if self.state is None:
            return None
        return tuple((tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype, tensor.device)
                     for tensor in self.state())

    def invalidate(self):
        if self.failed:
            raise RuntimeError("CUDA capture failed; restart this training process")
        if self.stream is not None:
            self.stream.synchronize()
        self.graph = None
        self.calls = 0
        self.identity = None
        self.configuration = None

    def _check_inputs(self, inputs):
        if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs.values()):
            raise TypeError("Captured updates require a nonempty dictionary of tensors")
        device = next(iter(inputs.values())).device
        if device.type != "cuda" or any(value.device != device for value in inputs.values()):
            raise ValueError("Captured inputs must all be on the same CUDA device")
        if self.calls:
            if inputs.keys() != self.inputs.keys():
                raise ValueError("Captured input keys changed; call invalidate() before changing the input contract")
            for key, value in inputs.items():
                expected = self.inputs[key]
                if (value.shape, value.dtype, value.device) != (expected.shape, expected.dtype, expected.device):
                    raise ValueError(f"Captured input {key!r} changed shape, dtype or device; call invalidate() first")
        return device

    def __call__(self, inputs):
        if self.failed:
            raise RuntimeError("CUDA capture failed; restart this training process")
        device = self._check_inputs(inputs)  # Validate all fields before copying or updating anything.
        configuration = _config_snapshot(self.config()) if self.config is not None else None
        if self.calls and configuration != self.configuration:
            self.invalidate()
        if self.graph is not None and self._identity() != self.identity:
            self.invalidate()
        with torch.cuda.device(device):
            if self.calls == 0:
                self.inputs = {key: value.clone() for key, value in inputs.items()}
                self.stream = torch.cuda.Stream(device=device)
                self.configuration = configuration
            else:
                for key, value in inputs.items():
                    self.inputs[key].copy_(value)
            current = torch.cuda.current_stream()
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                if self.calls < 3:
                    output = self.function(self.inputs)
                else:
                    if self.graph is None:
                        graph = torch.cuda.CUDAGraph()
                        try:
                            with torch.cuda.graph(graph, stream=self.stream):
                                self.output = self.function(self.inputs)
                        except Exception:
                            self.failed = True
                            raise
                        self.graph = graph
                        self.identity = self._identity()
                    self.graph.replay()
                    output = self.output
                result = {key: value.clone() for key, value in output.items()}
            current.wait_stream(self.stream)
        self.calls += 1
        return result
