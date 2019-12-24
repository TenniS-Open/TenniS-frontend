import tensorstack as ts

from typing import List, Tuple

from stackfence.fence import Fence
from stackfence.metanode import *


def _convert_conv2d_v2(node):
    # type: (ts.Node) -> Union[None, ts.Node]
    conv2d = ts.graph.clone_bubble(node)
    conv2d.op = "conv2d"
    conv2d.set("padding", conv2d.get("#padding"))
    conv2d.clear("#padding")
    ts.Node.Link(conv2d, [node.inputs[0], node.inputs[2]])
    return conv2d


def _convert_depthwise_conv2d_v2(node):
    # type: (ts.Node) -> Union[None, ts.Node]
    conv2d = ts.graph.clone_bubble(node)
    conv2d.op = "depthwise_conv2d"
    conv2d.set("padding", conv2d.get("#padding"))
    conv2d.clear("#padding")
    ts.Node.Link(conv2d, [node.inputs[0], node.inputs[2]])
    return conv2d


def _convert_pooling2d_v2(node):
    # type: (ts.Node) -> Union[None, ts.Node]
    pooling2d = ts.graph.clone_bubble(node)
    pooling2d.op = "pooling2d"
    pooling2d.set("padding", pooling2d.get("#padding"))
    pooling2d.clear("#padding")
    pooling2d.set("ksize", node.inputs[2].get("value"))
    pooling2d.set("stride", node.inputs[3].get("value"))
    ts.Node.Link(pooling2d, [node.inputs[0]])
    return pooling2d


def freeze(outputs, inputs=None, input_shape=None):
    # type: (List[ts.Node], List[ts.Node], Union[List[Tuple[int]], Dict[str, Tuple[int]]]) -> Tuple
    # -> Tuple[List[ts.Node], List[ts.Node]]
    """
    get frozen graph, convert each v2 node to v1
    :param outputs:
    :param inputs:
    :return:
    """
    cache = {}
    outputs = ts.graph.clone_graph(outputs, cache)
    if inputs is not None:
        inputs = [cache[n] for n in inputs]

    if input_shape is not None:
        if inputs is None:
            raise ValueError("parameter inputs must be set, if input_shape has been set.")
        if isinstance(input_shape, (list, tuple)):
            for i in range(min(len(inputs), len(input_shape))):
                inputs[i].shape = input_shape[i]
        elif isinstance(input_shape, dict):
            for k, v in input_shape.items():
                for i in inputs:
                    if i.name == k:
                        i.shape = v
                        break
        else:
            raise ValueError("param input_shape must be list or dict, got {}".format(type(input_shape)))

    ts.inferer.infer(outputs)

    fence = Fence()
    fence.register(MetaGraph([
        ts.Node.Const,     # weights
        ({"#op": "conv2d_v2",
          "#padding": HasSet,
          "#shape": GT([None, 0, 0, 0])}, {2: -1}),
    ]), _convert_conv2d_v2)
    fence.register(MetaGraph([
        ts.Node.Const,     # ksize
        ts.Node.Const,     # stride
        ({"#op": "pooling2d_v2",
          "#padding": HasSet,
          "#shape": GT([None, 0, 0, 0])}, {2: -2, 3: -1}),
    ]), _convert_pooling2d_v2)
    fence.register(MetaGraph([
        ts.Node.Const,     # weights
        ({"#op": "depthwise_conv2d_v2",
          "#padding": HasSet,
          "#shape": GT([None, 0, 0, 0])}, {2: -1}),
    ]), _convert_depthwise_conv2d_v2)

    cache = {}
    outputs = fence.convert(outputs, cache)
    if inputs is not None:
        inputs = [cache[n] for n in inputs]

    return outputs, inputs