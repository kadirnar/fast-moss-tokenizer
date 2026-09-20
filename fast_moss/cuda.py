"""Version-checked CUDA driver and NVRTC bindings for quantizer kernels."""

from importlib.metadata import version


def _check(result):
    if int(result[0]):
        raise RuntimeError(str(result[0]))
    return result[1] if len(result) == 2 else result[1:]


def compiler():
    if (
        version("cuda-bindings") != "13.4.2"
        or version("nvidia-cuda-nvrtc-cu12") != "12.8.93"
    ):
        raise ValueError(
            "Quantizer kernels require cuda-bindings 13.4.2 and NVRTC 12.8.93"
        )
    from cuda.bindings import driver, nvrtc

    if tuple(_check(nvrtc.nvrtcVersion())) != (12, 8):
        raise ValueError("Quantizer kernels require NVRTC 12.8")
    return driver, nvrtc
