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


def fuse_softmax(node):
    # type: (ts.Node) -> Optional[ts.Node]
    name = node.name
    exp = node.inputs[0]
    reduce_sum = node.inputs[1]
    x = exp.inputs[0]
    assert isinstance(reduce_sum, ts.Node)

    dims = reduce_sum.get("dims")
    keep_dims = reduce_sum.try_get("keep_dims", True)

    if not keep_dims:
        return None

    dims = numpy.asarray(dims).reshape([-1])
    if len(dims) > 1:
        return None

    dim = dims[0]

    softmax = ts.zoo.softmax(name=name, x=x, dim=dim, smooth=False)
    if node.has("#shape"):
        softmax.shape = node.shape
    if node.has("#dtype"):
        softmax.dtype = node.dtype

    return softmax


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

    return reshape


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

    fence.register(MetaGraph([
        MetaNode(),
        ({"#op": "exp"}, -1),
        ({"#op": "reduce_sum"}, -1),
        ({"#op": "div"}, (-2, -1)),
    ]), fuse_softmax)

    fence.register(MetaNode({
            "#op": "_reshape_v2",
            "#shape": HasSet,
            "#dtype": NE(0),
        }), convert_reshape_v2_to_v1)

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

