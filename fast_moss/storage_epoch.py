"""Invalidate managed CUDA graphs across explicit parameter-storage transitions."""

_epochs = {}
_contexts = {}


def current(device):
    return _epochs.get(device, 0)


def advance(device, *, context=False):
    _epochs[device] = current(device) + 1
    if context:
        _contexts[device] = lifecycle(device) + 1


def lifecycle(device):
    return _contexts.get(device, 0)
