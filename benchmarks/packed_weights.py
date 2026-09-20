"""Research-only reversible FP32 storage packing for frozen linear weights.

Resident mode preserves values and Parameter identity, but changes storage and
strides until close(). Existing graphs and external aliases must not be used
across this transition. No host offload or precision conversion is involved.
"""
import torch


class PackedWeights:
    def __init__(self,model,resident=False):
        self.resident=resident
        self.entries={}
        self.aliased=set()
        owners={}
        for p in model.parameters():
            key=p.untyped_storage().data_ptr()
            if key in owners:
                self.aliased.update([id(p),id(owners[key])])
            owners[key]=p

    def get(self,weight):
        key=id(weight)
        if key not in self.entries:
            if weight.requires_grad or weight.dtype!=torch.float32 or not weight.is_cuda or not weight.is_contiguous():
                raise ValueError('Packing requires frozen contiguous CUDA FP32 weights')
            if self.resident and key in self.aliased:
                raise ValueError('Resident packing cannot replace externally aliased parameter storage')
            packed=weight.T.contiguous()
            self.entries[key]=(weight,packed)
            if self.resident:weight.data=packed.T
        return self.entries[key][1]

    @property
    def bytes(self):
        return sum(packed.numel()*packed.element_size() for _,packed in self.entries.values())

    def close(self):
        while self.entries:
            _,(weight,packed)=self.entries.popitem()
            if self.resident:weight.data=packed.T.contiguous()
