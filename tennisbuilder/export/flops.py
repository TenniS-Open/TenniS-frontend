import tennis as ts

import os
import sys

from typing import Tuple, List, Union
import numpy

from . import fridge

_flops_counter_map = {}

def _register_flops_counter_map(name, counter):
    _flops_counter_map[name] = counter


def infer_shape_only(outputs, inputs, input_shape):
    cache = {}
    outputs = ts.graph.clone_graph(outputs, cache)
    if inputs is not None:
        inputs = [cache[n] for n in inputs]

    if input_shape is not None:
        input_shape = fridge._check_input_shape(input_shape)
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

    return outputs

def _get_shape(node):
    assert isinstance(node, ts.Node)
    if node.op == ts.Node.Const:
        return node.get("value").shape
    if node.has("#shape"):
        return node.get("#shape")
    return None

def check_int_list(shape):
    try:
        for i in shape:
            i = int(i)
            if i <= 0:
                return False
        return True
    except:
        return False


def check_node_io(node):
    for i in node.inputs:
        if not check_int_list(_get_shape(i)):
            return False
    if not check_int_list(_get_shape(node)):
        return False
    return True


def ignore_flops_counter(node):
    # type: (ts.Node) -> int
    return 0


def analysis(outputs, inputs, input_shape, freeze = True):
    # type: (Union[Tuple, List, ts.Node]) -> int
    if isinstance(outputs, ts.Node):
        outputs = [outputs]
    try:
        outputs = [node for node in outputs]
    except:
        raise Exception("parameter 0 should be ts.Node or iterable of ts.Node")

    if freeze:
        outputs, _ = fridge.freeze(outputs=outputs, inputs=inputs, input_shape=input_shape)
        outputs = ts.optimizer.optimize(outputs)
    else:
        outputs = infer_shape_only(outputs=outputs, inputs=inputs, input_shape=input_shape)

    nodes, _ = ts.graph.walk_graph(outputs)

    flops = 0.
    for node in nodes:
        assert isinstance(node, ts.Node)

        if node.op not in _flops_counter_map:
            sys.stderr.write("--=[ GFLOPS  ignore layer: {}: {}\n".format(node.op, node.name))
            continue

        if not check_node_io(node):
            sys.stderr.write("--=[ GFLOPS uninfer layer: {}: {}\n".format(node.op, node.name))
            continue
        node_flops = _flops_counter_map[node.op](node)
        flops += float(node_flops)
        # sys.stderr.write("--=[ {:.3f} / {:.3f} of {}: {}\n".format(format_gflops(node_flops), format_gflops(flops), node.op, node.name))


    return flops


def format_gflops(flops):
    return float(flops) / pow(1000, 3)


_register_flops_counter_map("<const>", ignore_flops_counter)
_register_flops_counter_map("<param>", ignore_flops_counter)
_register_flops_counter_map("_cast", ignore_flops_counter)
_register_flops_counter_map("to_float", ignore_flops_counter)


def flops_counter_of_conv2d(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    input_shape = _get_shape(node.inputs[0])
    kernel_shape = _get_shape(node.inputs[1])
    output_shape = _get_shape(node)

    batch_size = input_shape[0]
    output_height, output_width = output_shape[2:]
    output_dims = list(output_shape[2:])

    kernel_dims = list(kernel_shape[2:])
    in_channels = input_shape[1]
    out_channels = output_shape[1]
    groups = 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = numpy.prod(kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * numpy.prod(output_dims)

    overall_conv_flops = conv_per_position_flops * active_elements_count

    return overall_conv_flops


_register_flops_counter_map("conv2d", flops_counter_of_conv2d)
_register_flops_counter_map("conv2d_v2", flops_counter_of_conv2d)


def flops_counter_of_add_bias(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    input_shape = _get_shape(node.inputs[0])
    output_shape = _get_shape(node)

    batch_size = input_shape[0]
    output_dims = list(output_shape[2:])

    out_channels = output_shape[1]

    active_elements_count = batch_size * numpy.prod(output_dims)

    bias_flops = out_channels * active_elements_count

    return bias_flops


_register_flops_counter_map("add_bias", flops_counter_of_add_bias)


def flops_counter_of_relu(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    input_shape = _get_shape(node.inputs[0])
    output_shape = _get_shape(node)

    active_elements_count = numpy.prod(input_shape)

    return active_elements_count


_register_flops_counter_map("relu", flops_counter_of_relu)
_register_flops_counter_map("prelu", flops_counter_of_relu)
_register_flops_counter_map("relu_max", flops_counter_of_relu)
_register_flops_counter_map("leaky_relu", flops_counter_of_relu)

_register_flops_counter_map("_reshape", ignore_flops_counter)


def flops_counter_of_element(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    return numpy.prod(_get_shape(node))


_register_flops_counter_map("add", flops_counter_of_element)
_register_flops_counter_map("sub", flops_counter_of_element)
_register_flops_counter_map("mul", flops_counter_of_element)
_register_flops_counter_map("div", flops_counter_of_element)


def flops_counter_of_gemm(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    A = _get_shape(node.inputs[0])
    B = _get_shape(node.inputs[1])

    A = [A[0], numpy.prod(A[1:])]

    transA = int(node.get("transA"))
    transB = int(node.get("transB"))

    alpha = float(node.try_get("alpha", 1))
    beta = float(node.try_get("beta", 1))

    M = A[0] if not transA else A[1]
    K = A[1] if not transA else A[0]
    N = B[1] if not transB else B[0]

    alpha = 0 if alpha == 0 else 1
    beta = 0 if beta == 0 else 1

    return M * K * N * alpha + N * beta

def flops_counter_of_inner_prod(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    A = _get_shape(node.inputs[0])
    B = _get_shape(node.inputs[1])

    A = [A[0], numpy.prod(A[1:])]

    transA = 0
    transB = int(node.try_get("transpose", 0))

    alpha = 1
    beta = 0

    M = A[0] if not transA else A[1]
    K = A[1] if not transA else A[0]
    N = B[1] if not transB else B[0]

    alpha = 0 if alpha == 0 else 1
    beta = 0 if beta == 0 else 1

    return M * K * N * alpha + N * beta


_register_flops_counter_map("gemm", flops_counter_of_gemm)


def flops_counter_of_pool(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    input_shape = _get_shape(node.inputs[0])

    return numpy.prod(input_shape)


_register_flops_counter_map("pooling2d", flops_counter_of_pool)
_register_flops_counter_map("pooling2d_v2", flops_counter_of_pool)
_register_flops_counter_map("global_pooling2d", flops_counter_of_pool)


def flops_counter_of_batch_norm(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    x = _get_shape(node.inputs[0])

    return numpy.prod(x) * 2


_register_flops_counter_map("batch_norm", flops_counter_of_batch_norm)


def flops_counter_of_batch_scale(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    x = _get_shape(node.inputs[0])

    return numpy.prod(x) * 2


_register_flops_counter_map("batch_scale", flops_counter_of_batch_scale)


def flops_counter_of_fused_batch_norm(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    x = _get_shape(node.inputs[0])

    return numpy.prod(x) * 4


_register_flops_counter_map("fused_batch_norm", flops_counter_of_fused_batch_norm)

_register_flops_counter_map("_dragon_conv2d_padding", ignore_flops_counter)
_register_flops_counter_map("_dragon_pooling2d_padding", ignore_flops_counter)
_register_flops_counter_map("_mx_pooling2d_padding", ignore_flops_counter)
_register_flops_counter_map("_onnx_pooling2d_padding", ignore_flops_counter)
_register_flops_counter_map("_tf_conv2d_padding", ignore_flops_counter)
_register_flops_counter_map("_tf_pooling2d_padding", ignore_flops_counter)

_register_flops_counter_map("_copy", ignore_flops_counter)
_register_flops_counter_map("_dims", ignore_flops_counter)
_register_flops_counter_map("_dimshuffle", ignore_flops_counter)
_register_flops_counter_map("_expand", ignore_flops_counter)


def flops_counter_of_limit(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    x = _get_shape(node.inputs[0])
    y = _get_shape(node)

    return numpy.prod(y) if not numpy.all(x == y) else 0



_register_flops_counter_map("_limit", ignore_flops_counter)
_register_flops_counter_map("_nhwc_center_crop2d", ignore_flops_counter)


def flops_counter_of_sample(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    x = _get_shape(node.inputs[0])
    y = _get_shape(node)

    return numpy.prod(y)

_register_flops_counter_map("_nhwc_letterbox", flops_counter_of_sample)
_register_flops_counter_map("_nhwc_scale_resize2d", flops_counter_of_sample)
_register_flops_counter_map("_resize2d", flops_counter_of_sample)

_register_flops_counter_map("_reshape", ignore_flops_counter)
_register_flops_counter_map("_reshape_v2", ignore_flops_counter)
_register_flops_counter_map("_shape", ignore_flops_counter)

_register_flops_counter_map("_transpose", ignore_flops_counter)
_register_flops_counter_map("abs", flops_counter_of_relu)

def flops_counter_of_affine_sample2d(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    x = _get_shape(node.inputs[0])
    y = _get_shape(node.inputs[1])

    return numpy.prod(y) * 3

_register_flops_counter_map("affine_sample2d", flops_counter_of_affine_sample2d)
_register_flops_counter_map("argmax", flops_counter_of_relu)

_register_flops_counter_map("batch_to_space4d", ignore_flops_counter)
_register_flops_counter_map("broadcast", ignore_flops_counter)

_register_flops_counter_map("ceil", flops_counter_of_element)

_register_flops_counter_map("chunk", ignore_flops_counter)
_register_flops_counter_map("concat", ignore_flops_counter)
_register_flops_counter_map("constant_of_shape", ignore_flops_counter)

_register_flops_counter_map("crop_nd", ignore_flops_counter)
_register_flops_counter_map("crop_to", ignore_flops_counter)
# _register_flops_counter_map("dcn_v2_forward", ignore_flops_counter)


def flops_counter_of_depthwise_conv2d(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    input_shape = _get_shape(node.inputs[0])
    kernel_shape = _get_shape(node.inputs[1])
    output_shape = _get_shape(node)

    batch_size = input_shape[0]
    output_height, output_width = output_shape[2:]
    output_dims = list(output_shape[2:])

    kernel_dims = list(kernel_shape[2:])
    in_channels = input_shape[1]
    out_channels = output_shape[1]
    groups = in_channels

    filters_per_channel = out_channels // groups
    conv_per_position_flops = numpy.prod(kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * numpy.prod(output_dims)

    overall_conv_flops = conv_per_position_flops * active_elements_count

    return overall_conv_flops

_register_flops_counter_map("depthwise_conv2d", flops_counter_of_depthwise_conv2d)
_register_flops_counter_map("depthwise_conv2d_v2", flops_counter_of_depthwise_conv2d)

# _register_flops_counter_map("detection_output", ignore_flops_counter)

_register_flops_counter_map("divided", ignore_flops_counter)

_register_flops_counter_map("equal", flops_counter_of_element)
_register_flops_counter_map("exp", flops_counter_of_element)
_register_flops_counter_map("flatten", ignore_flops_counter)
_register_flops_counter_map("flatten2d", ignore_flops_counter)
_register_flops_counter_map("floor", flops_counter_of_element)

def flops_counter_of_force_gray(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    x = _get_shape(node.inputs[0])
    y = _get_shape(node)

    return numpy.prod(y) * 3 if not numpy.all(x == y) else 0

_register_flops_counter_map("force_color", ignore_flops_counter)
_register_flops_counter_map("force_gray", flops_counter_of_force_gray)

_register_flops_counter_map("gather", ignore_flops_counter)
_register_flops_counter_map("gatherv2", ignore_flops_counter)
_register_flops_counter_map("inner_prod", flops_counter_of_inner_prod)

def flops_counter_of_norm(node):
    # type: (ts.Node) -> int
    assert isinstance(node, ts.Node)

    x = _get_shape(node.inputs[0])

    return numpy.prod(x) * 2    # pow(x, 2) and div

_register_flops_counter_map("l2_norm", flops_counter_of_norm)
_register_flops_counter_map("norm_image", flops_counter_of_norm)

_register_flops_counter_map("max", flops_counter_of_relu)
_register_flops_counter_map("maximum", flops_counter_of_element)
# _register_flops_counter_map("non_max_suppression_v3", ignore_flops_counter)
# _register_flops_counter_map("prior_box", ignore_flops_counter)
# _register_flops_counter_map("proposal", ignore_flops_counter)
# _register_flops_counter_map("roi_align", ignore_flops_counter)

_register_flops_counter_map("pad", ignore_flops_counter)
_register_flops_counter_map("prewhiten", flops_counter_of_norm)
_register_flops_counter_map("range", ignore_flops_counter)
_register_flops_counter_map("reduce_mean", flops_counter_of_relu)
_register_flops_counter_map("reduce_sum", flops_counter_of_relu)
_register_flops_counter_map("resize_nearest_neighbor", flops_counter_of_sample)
_register_flops_counter_map("rsqrt", flops_counter_of_relu)
_register_flops_counter_map("sample2d", flops_counter_of_sample)
_register_flops_counter_map("sample2d_v2", flops_counter_of_sample)
_register_flops_counter_map("shape_index_patch", ignore_flops_counter)
_register_flops_counter_map("sigmoid", flops_counter_of_relu)
_register_flops_counter_map("slice", ignore_flops_counter)
_register_flops_counter_map("slice_v2", ignore_flops_counter)
_register_flops_counter_map("slice_v3", ignore_flops_counter)
_register_flops_counter_map("softmax", flops_counter_of_norm)
_register_flops_counter_map("softplus", flops_counter_of_norm)
_register_flops_counter_map("space_to_batch4d", ignore_flops_counter)
_register_flops_counter_map("sqrt", flops_counter_of_relu)
_register_flops_counter_map("square", flops_counter_of_relu)
_register_flops_counter_map("squeeze", ignore_flops_counter)
_register_flops_counter_map("stack", ignore_flops_counter)
_register_flops_counter_map("strided_slice", ignore_flops_counter)
_register_flops_counter_map("strided_slice", ignore_flops_counter)

_register_flops_counter_map("tanh", flops_counter_of_relu)
_register_flops_counter_map("tile", ignore_flops_counter)
_register_flops_counter_map("topkv2", flops_counter_of_relu)
_register_flops_counter_map("transpose_conv2d", flops_counter_of_conv2d)
_register_flops_counter_map("unsqueeze", ignore_flops_counter)
_register_flops_counter_map("where", ignore_flops_counter)
_register_flops_counter_map("yolo", flops_counter_of_relu)
# _register_flops_counter_map("yolo_poster", ignore_flops_counter)


