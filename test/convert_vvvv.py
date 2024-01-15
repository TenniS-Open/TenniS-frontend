#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from typing import Union

import numpy

from tennisbuilder.vvvv.converter import convert
from tennisbuilder.vvvv import param
import tennis as ts

from tennisfence.fence import Fence
from tennisfence.metanode import MetaGraph


def fuse_padding_to_conv2d(node):
    # type: (ts.Node) -> Union[None, ts.Node]
    conv2d = ts.graph.clone_bubble(node)
    pad: ts.Node = node.inputs[0]
    x: ts.Node = pad.inputs[0]
    w: ts.Node = node.inputs[1]

    padding_value = ts.zoo.to_const(pad.inputs[1])
    conv2d.set("padding", padding_value, numpy.int32)

    ts.Node.Link(conv2d, [x, w])

    print("--# -=[ Freeze layer {}: {} -> {}".format(node.name, node.op, conv2d.op))

    return conv2d


def fence_module(input_name: str, output_name: str = None):
    if not output_name:
        output_name = input_name

    input_module = input_name
    output_module = output_name

    with open(input_module, "rb") as f:
        m = ts.Module.Load(f)

    fence = Fence()
    fence.register(MetaGraph([
        {"#op": "pad"},
        ({"#op": "conv2d"}, -1),
    ]), fuse_padding_to_conv2d)

    inputs = m.inputs
    outputs = m.outputs

    cache = {}
    outputs = fence.convert(outputs, cache)
    if inputs is not None:
        inputs = [cache[n] for n in inputs]

    m = ts.Module()
    m.load(outputs)
    m.sort_inputs(inputs)

    with open(output_module, "wb") as f:
        ts.Module.Save(f, m)


def test():
    with open("net.dat", "rb") as stream:
        header = param.read_param(stream, param.Int, param.Int, param.Int, param.Int, param.Float, param.Float, param.Float)
        input_channels = header[0]
        input_height = header[1]
        input_width = header[2]
        input = ts.menu.param("_input", shape=(1, input_channels, input_height, input_width))
        convert(stream, "test.vvvv.tsm", inputs=[input, ])


if __name__ == '__main__':
    test()
