import tennis as ts

from typing import List, Tuple

from tennisfence.fence import Fence
from tennisfence.metanode import *


def _convert_conv2d_v2(node):
    # type: (ts.Node) -> Union[None, ts.Node]
    conv2d = ts.graph.clone_bubble(node)
    conv2d.op = "conv2d"
    conv2d.set("padding", conv2d.get("#padding"))
    conv2d.clear("#padding")
    ts.Node.Link(conv2d, [node.inputs[0], node.inputs[2]])

    print("--# -=[ Freeze layer {}: {} -> {}".format(node.name, node.op, conv2d.op))
    return conv2d


def _convert_depthwise_conv2d_v2(node):
    # type: (ts.Node) -> Union[None, ts.Node]
    conv2d = ts.graph.clone_bubble(node)
    conv2d.op = "depthwise_conv2d"
    conv2d.set("padding", conv2d.get("#padding"))
    conv2d.clear("#padding")
    ts.Node.Link(conv2d, [node.inputs[0], node.inputs[2]])

    print("--# -=[ Freeze layer {}: {} -> {}".format(node.name, node.op, conv2d.op))
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

    print("--# -=[ Freeze layer {}: {} -> {}".format(node.name, node.op, pooling2d.op))
    return pooling2d


def _convert_resize2d(node):
    # type: (ts.Node) -> Union[None, ts.Node]
    x = node.inputs[0]
    size = node.inputs[1]

    if size.op == ts.Node.Const:
        return None

    size = ts.inferer._infer_value(size)

    if size is None:
        return None

    node_size = ts.menu.data(name=node.name + "_size", value=size, device=ts.device.CPU)

    resize2d = ts.graph.clone_bubble(node)
    ts.Node.Link(resize2d, [node.inputs[0], node_size])

    print("--# -=[ Freeze layer {}: {} -> {}".format(node.name, node.op, resize2d.op))
    return resize2d


def convert_reshape_v2_to_v1(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name
    reshape_v2 = node
    x = reshape_v2.inputs[0]
    shape = reshape_v2.inputs[1]
    assert isinstance(shape, ts.Node)

    if shape.op == ts.Node.Const:
        shape = shape.get("value")
    elif shape.has("#value"):
        shape = shape.get("#value")
    else:
        return None

    neg_count = 0
    for i in shape:
        if i < 0:
            neg_count += 1

    if neg_count > 1:
        return None

    reshape = ts.zoo.reshape(name, x, shape)
    if node.has("#shape"):
        reshape.shape = node.shape
    if node.has("#dtype"):
        reshape.dtype = node.dtype

    print("--# -=[ Freeze layer {}: {} -> {}".format(node.name, node.op, reshape.op))
    return reshape

def _convert_copy(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name
    x = node.inputs[0]

    copy = ts.graph.clone_bubble(x)
    ts.Node.Link(copy, x.inputs)
    copy.name = name

    # print("--# -=[ Freeze layer {}: {} -> {}".format(node.name, node.op, x.op))
    return copy


def _remove_node(node):
    # type: (ts.Node) -> Optional[ts.Node]
    return node.inputs[0]


def _check_input_shape_dict_str_int_list(shape):
    # type: (Dict[str, Tuple[int]]) -> bool
    if not isinstance(shape, dict):
        return False
    for k, v in shape.items():
        if not isinstance(k, str):
            return False
        if not _check_input_shape_int_list(v):
            return False
    return True


def _check_input_shape_int_list(shape):
    # type: (Union[List[int], Tuple[int]]) -> bool
    if not isinstance(shape, (list, tuple)):
        return False
    for i in shape:
        if not isinstance(i, int):
            return False
    return True


def _check_input_shape_list_of_int_list(shape):
    # type: ( List[Tuple[int]]) -> bool
    if not isinstance(shape, (list, tuple)):
        for i in shape:
            if not _check_input_shape_int_list(i):
                return False
    return True


def _check_input_shape(shape):
    # type: (Union[List[int], List[Tuple[int]], Dict[str, Tuple[int]]]) -> Union[List[Iterable[int]], Dict]
    def _error():
        raise Exception("Input shape must be List[int], List[Tuple[int]] or Dict[str, Tuple[int]]")

    if isinstance(shape, dict):
        if not _check_input_shape_dict_str_int_list(shape):
            _error()
        return shape

    if _check_input_shape_int_list(shape):
        return [shape]

    if not _check_input_shape_list_of_int_list(shape):
        _error()

    return shape


def is_infered_const(node):
    # type: (ts.Node) -> bool
    assert isinstance(node, ts.Node)

    if node.op == ts.Node.Const:
        return False

    return ts.inferer._infer_value(node) is not None


def to_const(node):
    # type: (ts.Node) -> Optional[ts.Node]
    assert isinstance(node, ts.Node)

    if node.op == ts.Node.Const:
        return None

    name = node.name
    value = ts.inferer._infer_value(node)

    if value is None:
        return None

    print("--# -=[ Freeze layer {}: {} -> {}".format(node.name, node.op, ts.Node.Const))
    return ts.menu.data(name=name, value=value)


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
        input_shape = _check_input_shape(input_shape)
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
    fence.register(is_infered_const, to_const)
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

    fence.register(MetaGraph([
        {"#op": "_resize2d"},
    ]), _convert_resize2d)
    fence.register(MetaNode({
            "#op": "_reshape_v2",
            "#shape": HasSet,
            "#dtype": NE(0),
        }), convert_reshape_v2_to_v1)
    fence.register(MetaGraph([
        {"#op": "_copy"},
    ]), _convert_copy)

    cache = {}
    outputs = fence.convert(outputs, cache)
    if inputs is not None:
        inputs = [cache[n] for n in inputs]

    return outputs, inputs