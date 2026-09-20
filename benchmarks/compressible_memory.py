"""Research-only CUDA VMM allocations with verified generic compression flags.

Keep external tensors alive through all graph replays. The owning CUDA-array
interface object synchronizes its device before unmapping on final release.
No allocation-size reduction is claimed: this requests transparent data
compression for memory traffic, not a smaller virtual allocation.
"""
import ctypes as C
import torch


class Location(C.Structure):
    _fields_ = [('type', C.c_int), ('id', C.c_int)]


class AllocationFlags(C.Structure):
    _fields_ = [('compressionType', C.c_ubyte), ('gpuDirectRDMACapable', C.c_ubyte),
                ('usage', C.c_ushort), ('reserved', C.c_ubyte * 4)]


class AllocationProperties(C.Structure):
    _fields_ = [('type', C.c_int), ('requestedHandleTypes', C.c_int), ('location', Location),
                ('win32HandleMetaData', C.c_void_p), ('allocFlags', AllocationFlags)]


class Access(C.Structure):
    _fields_ = [('location', Location), ('flags', C.c_int)]


def driver():
    lib = C.CDLL('libcuda.so.1')
    signatures = {
        'cuInit': [C.c_uint], 'cuDeviceGet': [C.POINTER(C.c_int), C.c_int],
        'cuDeviceGetAttribute': [C.POINTER(C.c_int), C.c_int, C.c_int],
        'cuMemGetAllocationGranularity': [C.POINTER(C.c_size_t), C.POINTER(AllocationProperties), C.c_int],
        'cuMemCreate': [C.POINTER(C.c_uint64), C.c_size_t, C.POINTER(AllocationProperties), C.c_uint64],
        'cuMemGetAllocationPropertiesFromHandle': [C.POINTER(AllocationProperties), C.c_uint64],
        'cuMemAddressReserve': [C.POINTER(C.c_uint64), C.c_size_t, C.c_size_t, C.c_uint64, C.c_uint64],
        'cuMemMap': [C.c_uint64, C.c_size_t, C.c_size_t, C.c_uint64, C.c_uint64],
        'cuMemSetAccess': [C.c_uint64, C.c_size_t, C.POINTER(Access), C.c_size_t],
        'cuMemUnmap': [C.c_uint64, C.c_size_t], 'cuMemAddressFree': [C.c_uint64, C.c_size_t],
        'cuMemRelease': [C.c_uint64], 'cuGetErrorName': [C.c_int, C.POINTER(C.c_char_p)],
    }
    for name, signature in signatures.items():
        fn = getattr(lib, name); fn.argtypes = signature; fn.restype = C.c_int
    return lib


def check(lib, status):
    if status:
        name = C.c_char_p(); lib.cuGetErrorName(status, C.byref(name))
        raise RuntimeError(f'CUDA driver error {status}: {name.value.decode() if name.value else "unknown"}')


class Allocation:
    def __init__(self, shape, compressed=True, device=0):
        self.lib = driver(); self.device = device
        self.handle = C.c_uint64(); self.pointer = C.c_uint64(); self.mapped = False
        self.shape = tuple(shape)
        count = 1
        for n in self.shape:
            if not isinstance(n, int) or n < 1: raise ValueError('Positive dimensions required')
            count *= n
        self.logical_bytes = count * 4
        with torch.cuda.device(device):
            torch.empty(0, device=f'cuda:{device}')  # Establish the primary context.
            check(self.lib, self.lib.cuInit(0))
            dev = C.c_int(); check(self.lib, self.lib.cuDeviceGet(C.byref(dev), device))
            support = C.c_int(); check(self.lib, self.lib.cuDeviceGetAttribute(C.byref(support), 107, dev))
            self.supported = bool(support.value)
            if compressed and not self.supported: raise RuntimeError('Generic compression is unavailable')
            prop = AllocationProperties(type=1, location=Location(1, dev.value))
            prop.allocFlags.compressionType = int(compressed)
            granularity = C.c_size_t()
            check(self.lib, self.lib.cuMemGetAllocationGranularity(C.byref(granularity), C.byref(prop), 0))
            self.granularity = granularity.value
            self.size = ((self.logical_bytes + self.granularity - 1) // self.granularity) * self.granularity
            check(self.lib, self.lib.cuMemCreate(C.byref(self.handle), self.size, C.byref(prop), 0))
            actual = AllocationProperties()
            check(self.lib, self.lib.cuMemGetAllocationPropertiesFromHandle(C.byref(actual), self.handle))
            self.compression_type = actual.allocFlags.compressionType
            if self.compression_type != int(compressed):
                raise RuntimeError('Driver did not grant the requested compression mode')
            check(self.lib, self.lib.cuMemAddressReserve(C.byref(self.pointer), self.size, self.granularity, 0, 0))
            check(self.lib, self.lib.cuMemMap(self.pointer, self.size, 0, self.handle, 0))
            self.mapped = True
            access = Access(Location(1, dev.value), 3)
            check(self.lib, self.lib.cuMemSetAccess(self.pointer, self.size, C.byref(access), 1))

    @property
    def __cuda_array_interface__(self):
        return {'shape': self.shape, 'typestr': '<f4', 'data': (self.pointer.value, False),
                'strides': None, 'version': 3}

    def tensor(self):
        # PyTorch retains this object in the tensor storage's deleter.
        return torch.as_tensor(self, device=f'cuda:{self.device}')

    def metadata(self):
        return {'logical_bytes': self.logical_bytes, 'allocated_bytes': self.size,
                'granularity': self.granularity, 'compression_type': self.compression_type,
                'device_support': self.supported}

    def __del__(self):
        # Partially constructed allocations also release acquired resources.
        try:
            if self.handle.value or self.pointer.value:
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize(self.device)
                    if self.mapped: check(self.lib, self.lib.cuMemUnmap(self.pointer, self.size))
                    if self.pointer.value: check(self.lib, self.lib.cuMemAddressFree(self.pointer, self.size))
                    if self.handle.value: check(self.lib, self.lib.cuMemRelease(self.handle))
        except Exception:
            # CUDA may already be shut down during interpreter teardown.
            pass


def capabilities():
    lib = driver(); check(lib, lib.cuInit(0))
    dev = C.c_int(); check(lib, lib.cuDeviceGet(C.byref(dev), 0))
    report = {'scope': 'read-only CUDA driver capability query', 'gpu': torch.cuda.get_device_name(0),
              'device_ordinal': 0, 'attributes': {}}
    for name, code in [('generic_compression_supported', 107), ('l2_cache_bytes', 38),
                       ('memory_clock_khz', 36), ('memory_bus_bits', 37),
                       ('compute_major', 75), ('compute_minor', 76)]:
        value = C.c_int(); check(lib, lib.cuDeviceGetAttribute(C.byref(value), code, dev))
        report['attributes'][name] = {'id': code, 'value': value.value}
    return report


if __name__ == '__main__':
    import json
    from pathlib import Path
    report = capabilities()
    Path('results/compression_capability.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)
