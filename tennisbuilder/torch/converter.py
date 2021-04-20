import torch
# from torchsummary import summary

from .module import convert_module

import numpy

import tennis as ts

from .onnx import convert as convert_onnx
from ..onnx import converter as onnx_converter
import tempfile


def convert_by_onnx(input_module, output_file, input, temp_onnx=None, opset_version=None):
    """
    convert troch model to tsm
    :param input_module: torch.nn.Module or param can be parsed to torch.load(param)
    :param output_file: str of path to file
    :param input: list of tuple or ts.Node
    :param temp_onnx: temp file
    :param opset_version: onnx opset version
    :return: ts.Module
    """
    torch_model = None
    if isinstance(input_module, str):
        torch_model = torch.load(input_module)
    elif isinstance(input_module, torch.nn.Module):
        torch_model = input_module

    if not isinstance(torch_model, torch.nn.Module):
        raise NotImplementedError("Not supported model: {}".format(type(input_module)))
    for param in torch_model.parameters():
        param.requires_grad = False

    for i in range(len(input)):
        node = input[i]
        if isinstance(node, (tuple, list)):
            for i in node:
                if not isinstance(i, (int, )):
                    raise RuntimeError("input must be a list of tuple[int]")
        else:
            raise RuntimeError("input must be a list of tuple[int]")

    assert isinstance(torch_model, torch.nn.Module)

    torch_model.eval()

    temp_onnx_file = temp_onnx
    if temp_onnx_file is None:
        temp_onnx_file = tempfile.mktemp()

    convert_onnx(torch_model, temp_onnx_file, input=input, opset_version=opset_version)
    # convert_onnx(torch_model, temp_onnx_file, input=input, opset_version=11)
    return onnx_converter.convert(temp_onnx_file, output_file, check_graph=False)


def convert(input_module, output_file, input):
    """
    convert troch model to tsm
    :param input_module: torch.nn.Module or param can be parsed to troch.load(param)
    :param output_file: str of path to file
    :param input: list of tuple or ts.Node
    :return: ts.Module
    """
    torch_model = None
    if isinstance(input_module, str):
        torch_model = torch.load(input_module)
    elif isinstance(input_module, torch.nn.Module):
        torch_model = input_module

    if not isinstance(torch_model, torch.nn.Module):
        raise NotImplementedError("Not supported model: {}".format(type(input_module)))
    for param in torch_model.parameters():
        param.requires_grad = False

    if not isinstance(input, (tuple, list)):
        raise RuntimeError("input must be a list of tuple of ts.Node")

    input_nodes = []
    for i in range(len(input)):
        node = input[i]
        if isinstance(node, ts.Node):
            input_nodes.append(node)
        elif isinstance(node, (tuple, list)):
            for i in node:
                if not isinstance(i, int):
                    raise RuntimeError("input must be a list of tuple of ts.Node")
            input_nodes.append(ts.menu.param("_input_%d" % (i, ), shape=node))
        else:
            raise RuntimeError("input must be a list of tuple of ts.Node")

    assert isinstance(torch_model, torch.nn.Module)

    module = None
    torch_model.eval()
    with torch.no_grad():
        ts_graph_outputs = convert_module(torch_model, input_nodes)

        ts_module = ts.module.Module()
        ts_module.load(ts_graph_outputs)
        ts_module.sort_inputs(input_nodes)

        module = ts_module

    with open(output_file, "wb") as fo:
        ts.Module.Save(stream=fo, module=module)

    print("============ Summary ============")
    print("Input file: {}".format(input_nodes))
    print("Output file: {}".format(output_file))
    index = 0
    print("Input node: ")
    for node in module.inputs:
        assert isinstance(node, ts.Node)
        print("{}: {}, shape={}".format(index, node.name, node.shape))
        index += 1
    index = 0
    print("Output node: ")
    for node in module.outputs:
        assert isinstance(node, ts.Node)
        print("{}: {}".format(index, node.name))
        index += 1

    return module

