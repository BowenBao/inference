import onnx
import torch
import os
import numpy as np
from onnx import shape_inference, numpy_helper

data_dir = 'test_data_set_0'

def SaveTensorProto(file_path, name, data):
    tp = numpy_helper.from_array(data)
    tp.name = name

    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())

def SaveData(test_data_dir, prefix, data_list):
    if isinstance(data_list, torch.autograd.Variable) or isinstance(data_list, torch.Tensor):
        data_list = [data_list]
    for i, d in enumerate(data_list):
        d = d.data.cpu().numpy()
        SaveTensorProto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), prefix + str(i+1), d)

def Save(dir, name, model, inputs, outputs, input_names = ['input1'], output_names = ['output1']):
    if hasattr(model, 'train'):
        model.train(False)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, 'test_' + name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    def f(t):
        return [f(i) for i in t] if isinstance(t, (list, tuple)) else t
    def g(t, res):
        for i in t:
            res.append(i) if not isinstance(i, (list, tuple)) else g(i, res)
        return res

    torch.onnx.export(model, tuple(inputs), os.path.join(dir, 'model.onnx'), verbose=True, input_names=input_names, output_names=output_names)

    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    inputs = f(inputs)
    inputs = g(inputs, [])
    outputs = f(outputs)
    outputs = g(outputs, [])

    SaveData(test_data_dir, 'input', inputs)
    SaveData(test_data_dir, 'output', outputs)

def update_inputs_outputs_dims(model, input_dims, output_dims):
    """
        This function updates the sizes of dimensions of the model's inputs and outputs to the values
        provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
        will be set for that dimension.
    """
    def update_dim(tensor, dim, i, j, dim_param_prefix):
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                dim_proto.dim_value = dim
            else:
                dim_proto.dim_param = dim_param_prefix + str(i) + '_' + str(j)
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            raise ValueError('Only int or str is accepted as dimension value, incorrect type: {}'.format(type(dim)))

    for i, input_dim_arr in enumerate(input_dims):
        for j, dim in enumerate(input_dim_arr):
            update_dim(model.graph.input[i], dim, i, j, 'in_')

    for i, output_dim_arr in enumerate(output_dims):
        for j, dim in enumerate(output_dim_arr):
            update_dim(model.graph.output[i], dim, i, j, 'out_')

    #onnx.checker.check_model(model)
    return model


def update_with_default_names(model):
    """
        This function updates the names of the model with the default of 'OpType_id'. This is useful for models
        exported from PyTorch. Because in PyTorch ops don't have names and exported names are all unique number ids.
    """
    old_name_to_new_name = {}
    old_name_to_input_of_node = {}
    new_name_to_node = {}

    def add_to_map(data_map, key, value):
        if not key in data_map:
            data_map[key] = []
        data_map[key].append(value)

    for node in model.graph.node:
        node.name = node.op_type + '_' + node.output[0]
        for i, input_name in enumerate(node.input):
            if not input_name in new_name_to_node.keys():
                if not input_name in old_name_to_new_name.keys():
                    add_to_map(old_name_to_input_of_node, input_name, node)
                else:
                    node.input[i] = old_name_to_new_name[input_name]
        for i, output_name in enumerate(node.output):
            if output_name[:6] == 'output':
                continue
            if not output_name in new_name_to_node.keys():
                new_output_name = node.op_type + '_' + output_name
                add_to_map(new_name_to_node, new_output_name, node)
                old_name_to_new_name[output_name] = new_output_name
                node.output[i] = new_output_name

    for old_name, input_of_node in old_name_to_input_of_node.items():
        if old_name in old_name_to_new_name.keys():
            for i, name in enumerate(input_of_node.input):
                if old_name == name:
                    input_of_node.input[i] = old_name_to_new_name[old_name]

    # onnx.checker.check_model(model)
    return model