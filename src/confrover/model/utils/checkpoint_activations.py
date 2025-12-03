# Copyright 2025 Bytedance Ltd. and/or its affiliates.
#
# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”)
#
# Original file was released under MIT, with the full license text available below.
# This modified file is released under the same license.
#
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Activation checkpointing utilities modified from fairseq

See https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/checkpoint_activations.py
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.utils.checkpoint as checkpoint

# from fairseq import utils
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def checkpoint_wrapper(m, offload_to_cpu=False):
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:
    - wraps an nn.Module, so that all subsequent calls will use checkpointing
    - handles keyword arguments in the forward
    - handles non-Tensor outputs from the forward

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
    """
    # should I check whether original_forward has already been set?
    assert not hasattr(m, "precheckpoint_forward"), (
        "checkpoint function has already been applied?"
    )
    m.precheckpoint_forward = m.forward
    m.forward = functools.partial(
        _checkpointed_forward,
        m.precheckpoint_forward,  # original_forward
        offload_to_cpu,
    )
    return m


def unwrap_checkpoint(m: torch.nn.Module):
    """
    unwrap a module and its children from checkpoint_wrapper
    """
    for module in m.modules():
        if hasattr(module, "precheckpoint_forward"):
            module.forward = module.precheckpoint_forward
            del module.precheckpoint_forward
        if hasattr(module, "old_deepcopy_method"):
            module.__deepcopy__ = module.old_deepcopy_method
            del module.old_deepcopy_method
    return m


def _checkpointed_forward(original_forward, offload_to_cpu, *args, **kwargs):
    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict = {"offload": offload_to_cpu}
    output = CheckpointFunction.apply(
        original_forward, parent_ctx_dict, kwarg_keys, *flat_args
    )
    if isinstance(output, torch.Tensor):
        return output
    else:
        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)
        return output


def pack_kwargs(*args, **kwargs) -> Tuple[List[str], List[Any]]:
    """
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == [1, 2]
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys = []
    flat_args = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return kwarg_keys, flat_args


def unpack_kwargs(
    kwarg_keys: List[str], flat_args: List[Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs


def split_non_tensors(
    mixed: Union[torch.Tensor, Tuple[Any]],
) -> Tuple[Tuple[torch.Tensor], Dict[str, List[Any]]]:
    """
    Usage::

        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors = []
    packed_non_tensors = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors


def unpack_non_tensors(
    tensors: Tuple[torch.Tensor],
    packed_non_tensors: Dict[str, List[Any]],
) -> Tuple[Any]:
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict)
    mixed = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list)
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)


class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.

    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling ``unpack_non_tensors``.
    """

    @staticmethod
    def forward(ctx, run_function, parent_ctx_dict, kwarg_keys, *args):
        if torch.is_grad_enabled():  # grad may be disabled, e.g., during validation
            checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = get_rng_state()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(x.requires_grad for x in tensor_inputs)
            tensor_inputs = tuple(
                x.to(torch.device("cpu"), non_blocking=True) for x in tensor_inputs
            )

        else:
            ctx.fwd_device, ctx.grad_requirements = None, None

        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs

        with torch.no_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)

        if isinstance(outputs, torch.Tensor):
            return outputs
        else:
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs
            return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        tensor_inputs: Tuple = ctx.saved_tensors
        tensor_inputs = checkpoint.detach_variable(tensor_inputs)
        if ctx.fwd_device is not None:
            tensor_inputs = [
                t.to(ctx.fwd_device[i], non_blocking=True)
                for i, t in enumerate(tensor_inputs)
            ]
            for i, need_grad in enumerate(ctx.grad_requirements):
                tensor_inputs[i].requires_grad = need_grad
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)

        # Store the current states.
        bwd_rng_state = get_rng_state()

        # Set the states to what it used to be before the forward pass.
        set_rng_state(ctx.fwd_rng_state)

        with torch.enable_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(ctx.kwarg_keys, inputs)
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)
        # Set the states back to what it was at the start of this function.
        set_rng_state(bwd_rng_state)

        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of the outputs have requires_grad=True, "
                "this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs
        )
        return (None, None, None) + grads


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if xm is not None:
        state["xla_rng_state"] = xm.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if xm is not None:
        xm.set_rng_state(state["xla_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])
