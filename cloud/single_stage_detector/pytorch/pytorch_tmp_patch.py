"""
This is a temporary patch since many fixes in PyTorch are still in PR and not checked in yet.
"""

import torch

"""
PyTorch Custom Ops support
"""
_custom_ops_symbolic_fn = {}

def register_custom_op_symbolic(symbolic_name, symbolic_fn):
    # TODO: do checks
    ns, op_name = symbolic_name.split('::')
    if not ns in _custom_ops_symbolic_fn:
        _custom_ops_symbolic_fn[ns] = {}
    _custom_ops_symbolic_fn[ns][op_name] = symbolic_fn

from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes
import warnings
def _run_symbolic_function(g, n, inputs, env, operator_export_type=OperatorExportTypes.ONNX):
    # NB: Returning None means the node gets cloned as is into
    # the new graph
    try:
        import torch.onnx.symbolic
        # See Note [Export inplace]
        # TODO: I think this is not necessary anymore
        if n.kind().endswith('_'):
            ns_op_name = n.kind()[:-1]
        else:
            ns_op_name = n.kind()
        ns, op_name = ns_op_name.split("::")

        if ns == "onnx":
            # Use the original node directly
            return None

        elif ns == "aten":
            is_exportable_aten_op = hasattr(torch.onnx.symbolic, op_name)
            is_onnx_aten_export = operator_export_type == OperatorExportTypes.ONNX_ATEN
            is_aten_fallback_export = operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK
            if is_onnx_aten_export or (not is_exportable_aten_op and is_aten_fallback_export):
                # Direct ATen export requested
                attrs = {k + "_" + n.kindOf(k)[0]: n[k] for k in n.attributeNames()}
                outputs = n.outputsSize()
                attrs["outputs"] = outputs
                return _graph_at(g, op_name, *inputs, aten=True, **attrs)

            else:
                # Export it regularly
                attrs = {k: n[k] for k in n.attributeNames()}
                if not is_exportable_aten_op:
                    warnings.warn("ONNX export failed on ATen operator {} because torch.onnx.symbolic.{} does not exist"
                                  .format(op_name, op_name))
                    return None
                fn = getattr(torch.onnx.symbolic, op_name)
                return fn(g, *inputs, **attrs)

        elif ns == "prim":
            if op_name == "Constant" and not n.mustBeNone():
                if n.kindOf("value") == "t":
                    return g.op("Constant", value_t=n["value"])
                elif n.kindOf("value") == "is":
                    value = torch.stack([torch.tensor(v) for v in n["value"]]) if n["value"] else []
                    return g.op("Constant", value_t=value)
                elif n.output().type().kind() == "DeviceObjType":
                    return None
                else:
                    raise RuntimeError("Unsupported prim::Constant kind: `{}`. Send a bug report.".format(
                        n.kindOf("value")))
            elif n.mustBeNone() or op_name == "ListConstruct" or op_name == "ListUnpack":
                # None is not an ONNX operator; keep it as None
                # let the exporter handle finally eliminating these

                # For ListConstruct/ListUnpack, it will be erased in the ONNX peephole pass
                return None
            elif op_name == 'Loop' or op_name == 'If':
                print('loop node(orig):', n)
                new_op_outputs = g.op(op_name, *inputs, outputs=n.outputsSize())
                print('loop node:', new_op_outputs)
                new_node = new_op_outputs[0].node() if n.outputsSize() > 1 else new_op_outputs.node()
                for b in n.blocks():
                    new_block = new_node.addBlock()
                    torch._C._jit_pass_onnx_block(b, new_block, operator_export_type, env)
                return new_op_outputs
            else:
                symbolic_name = 'prim_' + op_name
                symbolic_fn = getattr(torch.onnx.symbolic, symbolic_name, None)
                if symbolic_fn is None:
                    warnings.warn("ONNX export failed on primitive operator {}; please report a bug".format(op_name))
                    return None
                attrs = {k: n[k] for k in n.attributeNames()}
                return symbolic_fn(g, *inputs, **attrs)

        elif ns in _custom_ops_symbolic_fn:
            # TODO: do checks
            symbolic_fn = _custom_ops_symbolic_fn[ns][op_name]
            attrs = {k: n[k] for k in n.attributeNames()}
            return symbolic_fn(g, *inputs, **attrs)

        else:
            warnings.warn("ONNX export failed on an operator with unrecognized namespace {}::{}; "
                          "please report a bug".format(ns, op_name))
            return None

    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise

import torch.onnx
torch.onnx.register_custom_op_symbolic = register_custom_op_symbolic

import torch.onnx.utils
torch.onnx.utils._run_symbolic_function = _run_symbolic_function


"""
PyTorch Exporter Ops Support
"""
from torch.onnx.symbolic import parse_args, log, _maybe_get_const, _is_value, scalar_type_to_pytorch_type, t, _unpack_list, index_select, _parse_arg

@parse_args('v')
def prim_shape(g, self):
    return g.op('Shape', self)

@parse_args('v')
def log2(g, self):
    _ln2 = 0.693147180559945309
    return g.op('Div', log(g, self), g.op('Constant', value_t=torch.Tensor([_ln2])))

@parse_args('v')
def floor(g, self):
    return g.op('Floor', self)
@parse_args('v')
def ceil(g, self):
    return g.op('Ceil', self)

@parse_args('v', 'i', 'v', 'v')
def scatter(g, self, dim, index, source):
    return g.op('Scatter', self, index, source, axis_i=dim)

cast_pytorch_to_onnx = {
    'Byte': torch.onnx.TensorProtoDataType.UINT8,
    'Char': torch.onnx.TensorProtoDataType.INT8,
    'Double': torch.onnx.TensorProtoDataType.DOUBLE,
    'Float': torch.onnx.TensorProtoDataType.FLOAT,
    'Half': torch.onnx.TensorProtoDataType.FLOAT16,
    'Int': torch.onnx.TensorProtoDataType.INT32,
    'Long': torch.onnx.TensorProtoDataType.INT64,
    'Short': torch.onnx.TensorProtoDataType.INT16,
    'Bool': torch.onnx.TensorProtoDataType.BOOL,
}

def _cast_Bool(g, input, non_blocking):
    return g.op("Cast", input, to_i=cast_pytorch_to_onnx['Bool'])

def _cast_Float(g, input, non_blocking):
    return g.op("Cast", input, to_i=cast_pytorch_to_onnx['Float'])

def _cast_Long(g, input, non_blocking):
    return g.op("Cast", input, to_i=cast_pytorch_to_onnx['Long'])

def _cast_Int(g, input, non_blocking):
    return g.op("Cast", input, to_i=cast_pytorch_to_onnx['Int'])

def wrap_logical_op_with_cast_to_and_from_bool(func):
    def wrap_with_cast(g, input, other):
        out = func(g, _cast_Bool(g, input, False), _cast_Bool(g, other, False))
        return g.op("Cast", out, to_i=cast_pytorch_to_onnx[input.type().scalarType()])
    return wrap_with_cast

@wrap_logical_op_with_cast_to_and_from_bool
def __and_(g, input, other):
    return g.op('And', input, other)

@parse_args('v', 'i', 'v', 'v')
def zeros(g, sizes, dtype, layout, device):
    # NOTE: no way to set device and layout in ONNX, so we ignore it
    return g.op("ConstantOfShape", sizes,
                value_t=torch.tensor([0], dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v')
def zeros_like(g, input, dtype, layout, device):
    shape = g.op("Shape", input)
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor([0], dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v')
def ones(g, sizes, dtype, layout, device):
    return g.op("ConstantOfShape", sizes,
                value_t=torch.tensor([1], dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'i', 'v', 'v')
def ones_like(g, input, dtype, layout, device):
    shape = g.op("Shape", input)
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor([1], dtype=scalar_type_to_pytorch_type[dtype]))


def full(g, sizes, value, dtype, layout, device):
    const_value = _maybe_get_const(value, 't')
    if _is_value(const_value):
        tmp = zeros(sizes, dtype, layout, device)
        return add(tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = _get_const(dtype, 'i', 'dtype')
        return g.op("ConstantOfShape", sizes,
                    value_t=torch.tensor([const_value], dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v', 'f', 'i', 'v', 'v')
def full_like(g, input, fill_value, dtype, layout, device):
    shape = g.op("Shape", input)
    return g.op("ConstantOfShape", shape,
                value_t=torch.tensor([fill_value], dtype=scalar_type_to_pytorch_type[dtype]))


@parse_args('v')
def nonzero(g, input):
    return t(g, g.op('NonZero', _cast_Float(g, input, False)))


@parse_args('v', 'v', 'i', 'i', 'i')
def topk(g, self, k, dim, largest, sorted, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    if not largest:
        _unimplemented("TopK", "Ascending TopK is not supported")

    k_value = _maybe_get_const(k, 'i')
    if not _is_value(k_value):
        k_value = g.op("Constant", value_t=torch.tensor(k_value, dtype=torch.long))

    k_value = g.op("Unsqueeze", k_value, axes_i=[0])
    return g.op("TopK", self, k_value, axis_i=dim, outputs=2)

def _is_packed_list(list_value):
    list_node = list_value.node()
    return list_node.kind() == "prim::ListConstruct"
def index(g, self, index):
    if _is_packed_list(index):
        indices = _unpack_list(index)
    else:
        indices = [index]

    if len(indices) == 1:
        if indices[0].type().scalarType() == "Byte":
            indices[0] = squeeze(g, nonzero(g, indices[0]), dim=1)
        return index_select(g, self, 0, indices[0])
    else:
        raise NotImplementedError("Unsupported aten::index operator with more than 1 indices tensor. ")


def symbolic_min(g, self, dim_or_y=None, keepdim=None):
    if dim_or_y is None and keepdim is None:
        # tmp workaround for ort not supporting int64 reducemin
        # return g.op("ReduceMin", self, keepdims_i=0)
        return _cast_Long(g, g.op("ReduceMin", _cast_Int(g, self, False), keepdims_i=0), False)
    if keepdim is None:
        return g.op("Min", self, dim_or_y)
    else:
        dim = _get_const(dim_or_y, 'i', 'dim')
        keepdim = _get_const(keepdim, 'i', 'keepdim')
        # TODO: export it as ReduceMax
        return g.op("ATen",
                    self,
                    operator_s="min",
                    dim_i=dim,
                    keepdim_i=keepdim,
                    outputs=2)


# export slice for opset 10.
def slice(g, self, dim, start, end, step):
    if start.node().kind() != 'onnx::Constant' or \
            end.node().kind() != 'onnx::Constant' or dim.node().kind() != 'onnx::Constant' or \
            step.node().kind() != 'onnx::Constant':
        start_unsqueezed = g.op("Unsqueeze", start, axes_i=[0])
        end_unsqueezed = g.op("Unsqueeze", end, axes_i=[0])
        dim_unsqueezed = g.op("Unsqueeze", dim, axes_i=[0])
        step_unsqueezed = g.op("Unsqueeze", step, axes_i=[0])
        return g.op("Slice", self, start_unsqueezed, end_unsqueezed, dim_unsqueezed, step_unsqueezed)
    else:
        start = _parse_arg(start, 'i')
        end = _parse_arg(end, 'i')
        dim = _parse_arg(dim, 'i')
        step = _parse_arg(step, 'i')
        start_tensor = g.op('Constant', value_t=torch.tensor([start], dtype=torch.long))
        end_tensor = g.op('Constant', value_t=torch.tensor([end], dtype=torch.long))
        dim_tensor = g.op('Constant', value_t=torch.tensor([dim], dtype=torch.long))
        step_tensor = g.op('Constant', value_t=torch.tensor([step], dtype=torch.long))
        return g.op("Slice", self, start_tensor, end_tensor, dim_tensor, step_tensor)


torch.onnx.symbolic._default_onnx_opset_version = 10

torch.onnx.symbolic.prim_shape = prim_shape
torch.onnx.symbolic.log2 = log2
torch.onnx.symbolic.floor = floor
torch.onnx.symbolic.ceil = ceil
torch.onnx.symbolic.scatter = scatter
torch.onnx.symbolic.__and_ = __and_
torch.onnx.symbolic.nonzero = nonzero
torch.onnx.symbolic.topk = topk
torch.onnx.symbolic.min = symbolic_min
torch.onnx.symbolic.index = index
torch.onnx.symbolic.slice = slice