# -*- coding: utf-8 -*-
"""Read-only mapped safetensors.

safetensors hands out PAGE_WRITECOPY views, and Windows charges a copy-on-write
view against the process commit limit by file size: the MOVA base costs 72GB of
commit the moment it is mapped (measured per shard: 9.28GB file -> 9.24GB of
commit). A read-only view of the same file costs nothing (measured: 0.02GB), and
the tensor values are identical.

Read-only views cannot be written, so they may only back weights that inference
never overwrites.
"""
import json
import mmap
import os
import struct

import torch

_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
}


def _file_tensors(path):
    handle = open(path, "rb")
    mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    header_size = struct.unpack("<Q", mm[:8])[0]
    header = json.loads(mm[8:8 + header_size])
    data_start = 8 + header_size
    tensors = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = torch.frombuffer(mm, dtype=torch.uint8, count=end - start, offset=data_start + start)
        tensors[key] = raw.view(_DTYPE_MAP[meta["dtype"]]).reshape(meta["shape"])
    return tensors, (handle, mm)


def read_only_state_dict(path, logger=None):
    """Read-only views of a safetensors file, or of every such file in a directory.

    Returns (state_dict, owners). Keep ``owners`` alive for as long as the views
    are used; they hold the file handles and the mappings.
    """
    if os.path.isdir(path):
        files = [os.path.join(path, name) for name in sorted(os.listdir(path))
                 if name.endswith(".safetensors")]
    else:
        files = [path]
    state = {}
    owners = []
    for file in files:
        tensors, owner = _file_tensors(file)
        state.update(tensors)
        owners.append(owner)
    if logger is not None:
        logger(f"[readonly] {os.path.basename(os.path.normpath(path))}: "
               f"{len(state)} tensors mapped read-only")
    return state, owners


def unset_tensors(module):
    """Name every tensor of ``module`` that still lives on the meta device.

    Covers parameters, buffers, and plain tensor attributes such as the RoPE
    frequency tables, which no checkpoint carries and which therefore have to be
    rebuilt after a meta-device build.
    """
    found = [name for name, tensor in
             list(module.named_parameters(remove_duplicate=False)) + list(module.named_buffers())
             if tensor.is_meta]
    for module_name, sub in module.named_modules():
        for attr, value in vars(sub).items():
            values = value if isinstance(value, (tuple, list)) else (value,)
            if any(isinstance(t, torch.Tensor) and t.is_meta for t in values):
                found.append(f"{module_name}.{attr}" if module_name else attr)
    return found


def alias_readonly(module, path, logger=None):
    """Point matching parameters and buffers at read-only views of ``path``.

    Uses ``load_state_dict(assign=True)``, the sanctioned way to fill a module
    built on the meta device: it assigns the mapped tensors instead of copying
    into existing storage, so no weight memory is ever allocated, and the
    safetensors WRITECOPY views are dropped for good.

    Returns (replaced, unset): ``unset`` names the tensors the checkpoint does not
    cover, i.e. tensors still living on the meta device.
    """
    state, owners = read_only_state_dict(path, logger)
    targets = dict(module.named_parameters(remove_duplicate=False))
    targets.update(module.named_buffers())
    assign = {}
    for key, tensor in state.items():
        target = targets.get(key)
        if target is None or target.shape != tensor.shape or target.dtype != tensor.dtype:
            continue
        assign[key] = tensor
    module.load_state_dict(assign, strict=False, assign=True)
    module._readonly_views = owners
    unset = unset_tensors(module)
    if logger is not None:
        logger(f"[readonly] {len(assign)}/{len(state)} weights aliased read-only, "
               f"{len(unset)} unset")
    return len(assign), unset
