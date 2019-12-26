from stackfence.fence import Fence
from stackfence.metanode import *

from typing import Optional


def fuse_flatten_ip_bias_reshape(node):
    # type: (ts.Node) -> Optional[ts.Node]
    reshape = node
    add_bias = reshape.inputs[0]
    inner_prod = add_bias.inputs[0]
    flatten = inner_prod.inputs[0]
    x = flatten.inputs[0]
    W = inner_prod.inputs[1]
    B = add_bias.inputs[1]

    if flatten.has("dim") and int(flatten.get("dim")) != 1:
        return None

    shape = list(reshape.get("shape"))
    if len(shape) != 4 or shape[2] != 1 or shape[3] != 1:
        return None

    caffe_inner_prod = ts.graph.clone_bubble(inner_prod)
    caffe_inner_prod.name = node.name
    caffe_inner_prod.op = "caffe:inner_prod"

    ts.Node.Link(caffe_inner_prod, [x, W, B])

    return caffe_inner_prod


def fuse_flatten_ip_reshape(node):
    # type: (ts.Node) -> Optional[ts.Node]
    reshape = node
    inner_prod = reshape.inputs[0]
    flatten = inner_prod.inputs[0]
    x = flatten.inputs[0]
    W = inner_prod.inputs[1]

    if flatten.has("dim") and int(flatten.get("dim")) != 1:
        return None

    shape = list(reshape.get("shape"))
    if len(shape) != 4 or shape[2] != 1 or shape[3] != 1:
        return None

    caffe_inner_prod = ts.graph.clone_bubble(inner_prod)
    caffe_inner_prod.name = node.name
    caffe_inner_prod.op = "caffe:inner_prod"

    ts.Node.Link(caffe_inner_prod, [x, W])

    return caffe_inner_prod


def get_fence():
    # type: () -> Fence
    fence = Fence()

    fence.register(MetaGraph([
        ts.Node.Const,
        ts.Node.Const,
        {"#op": "flatten"},
        ({"#op": "inner_prod"}, [-1, ABS(0)]),
        ({"#op": "add_bias"}, [-1, ABS(1)]),
        ({"#op": "_reshape"}, -1),
    ]), fuse_flatten_ip_bias_reshape)

    fence.register(MetaGraph([
        ts.Node.Const,
        {"#op": "flatten"},
        ({"#op": "inner_prod"}, [-1, ABS(0)]),
        ({"#op": "_reshape"}, -1,),
    ]), fuse_flatten_ip_reshape)

    return fence


def throw_non_back(node):
    # type: (ts.Node) -> Optional[ts.Node]
    raise Exception("No registered converter for {}".format(node.op))


def back_fence():
    # type: () -> Fence
    fence = Fence()

    fence.register(
        lambda x: x.op[:6] == "caffe:",
        throw_non_back,
        -1
    )

    return fence

