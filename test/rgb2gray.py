import tennis as ts
from tennisfence.fence import *

import numpy


def is_conv_bias(node):
    return node.op in {"add_bias"} and node.inputs[0].op in {"conv2d", "conv2d_v2"}


def is_conv(node):
    return node.op in {"conv2d", "conv2d_v2"}


def is_bias(node):
    return node.op in {"add_bias"}


def check_if_none_or_same(tuple_nodes):
    # type: (List[Tuple[ts.Node]]) -> Optional[Tuple[ts.Node]]
    found = None
    for nodes in tuple_nodes:
        if nodes is None:
            continue
        if found is None:
            found = nodes
        else:
            if found[0] is not nodes[0] or found[1] is not nodes[1]:
                raise Exception("Got different input conv: {}".format(
                    [[n.name for n in nodes] for nodes in tuple_nodes]))
    return found


def first_conv_bias(node, cache=None):
    # type: (ts.Node, Dict) -> (ts.Node, ts.Node)
    if cache is None:
        cache = {}

    if isinstance(node, (tuple, list)):
        return check_if_none_or_same(
            [cache[n] if n in cache else first_conv_bias(n, cache) for n in node])

    if node in cache:
        return node

    input_conv = [cache[n] if n in cache else first_conv_bias(n, cache) for n in node.inputs]
    front_conv = check_if_none_or_same(input_conv)

    result = None
    if front_conv is None:
        if is_conv(node):
            result = (node, None)
    else:
        if is_bias(node) and front_conv[0] is node.inputs[0] and \
                front_conv[1] is None:
            result = (front_conv[0], node)
        else:
            result = front_conv
    cache[node] = result
    return result


def get_conv_w(node):
    if node.op == "conv2d":
        return node.inputs[1]
    if node.op == "conv2d_v2":
        return node.inputs[2]
    return None


def get_bias_b(node):
    return node.inputs[1]


def rgb2gray(input_tsm, output_tsm,
             mean=None,
             variance=None,
             input_shape=None,
             wipeout=None):
    # use sub mean and div variance
    with open(input_tsm, "rb") as f:
        m = ts.Module.Load(f)
    conv, bias = first_conv_bias(m.outputs)
    print("[=<> Found first conv: {}".format(conv.name))

    if bias is None:
        raise Exception("conv must followed bias is this version")

    w_node = get_conv_w(conv)
    w_value = w_node.get("value")
    b_node = get_bias_b(bias)
    b_value = b_node.get("value")

    assert w_value.shape[1] == 3

    if mean is None:
        mean = [0, 0, 0]
    if variance is None:
        variance = [1, 1, 1]

    mean = numpy.asarray(mean)
    variance = numpy.asarray(variance)

    assert numpy.prod(mean.shape) == 3
    assert numpy.prod(variance.shape) == 1 or numpy.prod(variance.shape) == 3
    if numpy.prod(variance.shape) == 1:
        var = numpy.reshape(variance, [-1])[0]
        variance = numpy.asarray([var, var, var])

    w_value_update = \
        w_value[:, 0:1, :, :] / variance[0] + \
        w_value[:, 1:2, :, :] / variance[1] + \
        w_value[:, 2:3, :, :] / variance[2]

    out_c = b_value.shape[0]
    b_value_update = b_value
    if not numpy.all(mean == [0, 0, 0]):
        b_value_update = \
            b_value - \
            numpy.sum((mean[0] * w_value[:, 0:1, :, :] / variance[0]).reshape([out_c, -1]), axis=1) - \
            numpy.sum((mean[1] * w_value[:, 1:2, :, :] / variance[1]).reshape([out_c, -1]), axis=1) - \
            numpy.sum((mean[2] * w_value[:, 2:3, :, :] / variance[2]).reshape([out_c, -1]), axis=1)

    w_node.set("value", w_value_update, numpy.float32)
    b_node.set("value", b_value_update, numpy.float32)

    if wipeout is not None:
        if isinstance(wipeout, str):
            wipeout = [wipeout]

        wipeout = set(wipeout)

        f = Fence()
        f.register(
            lambda x: x.op in wipeout,
                lambda x: x.inputs[0])

        first_inputs = f.convert(conv.inputs)
        ts.Node.Link(conv, first_inputs)

    with open(output_tsm, "wb") as f:
        ts.Module.Save(f, m)


if __name__ == '__main__':
    rgb2gray(r"F:\rt\NIR\_acc96_res18_suzhounir_2time_no_equ_16k_1224_1.tsm",
             r"F:\rt\NIR\nir_fas_224x224x1_1224_1.tsm",
             mean=[104, 117, 123],
             wipeout=["sub"])
    pass