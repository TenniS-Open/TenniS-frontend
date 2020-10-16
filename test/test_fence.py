#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from tennisfence.metanode import MetaGraph
from tennisfence.metanode import EQ, NE, GT, LT, GE, LE
from tennisfence.fence import Fence
import tennis as ts


def narrow_prelu(node: ts.Node):
    assert len(node.inputs) == 2
    x = node.inputs[0]
    unsqueeze = node.inputs[1]
    slope = unsqueeze.inputs[0]

    value = slope.get("value")
    if len(value.shape) != 1:
        return None

    return ts.zoo.prelu(node.name, x=x, dim=1, slope=slope)


def test():
    file = r"raw.onnx.tsm"
    out = r"convert.onnx.tsm"

    with open(file, "rb") as fi:
        module = ts.Module.Load(stream=fi)

    inputs = module.inputs
    outputs = module.outputs

    f = Fence()
    f.register(MetaGraph([
        ts.Node.Const,
        ({"#op": "unsqueeze", "axes": EQ((1, 2))}, [-1]),
        ({"#op": "onnx::prelu"}, [None, -1])
    ]), narrow_prelu)

    outputs, inputs = f.convert(outputs, after=inputs)

    new = ts.Module()
    new.load(outputs)
    new.sort_inputs(inputs)

    with open(out, "wb") as fo:
        ts.Module.Save(stream=fo, module=new)


if __name__ == '__main__':
    test()
