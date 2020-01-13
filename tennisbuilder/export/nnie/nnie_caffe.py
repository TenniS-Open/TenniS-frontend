# coding: UTF-8

import tennis as ts

from .proto import nnie_pb2 as caffe

from typing import List
import os
from collections import OrderedDict
import numpy

from tennisfence.fence import Fence
from tennisfence.metanode import *

from typing import Optional


def fuse_conv2d_bias(node):
    # type: (ts.Node) -> ts.Node
    conv2d = ts.graph.clone_bubble(node.inputs[0])
    conv2d.name = node.name
    ts.Node.Link(conv2d, (node.inputs[0].inputs[0], node.inputs[0].inputs[1], node.inputs[1]))
    return conv2d


def fuse_ip_bias(node):
    # type: (ts.Node) -> ts.Node
    ip = ts.graph.clone_bubble(node.inputs[0])
    ip.name = node.name
    ts.Node.Link(ip, (node.inputs[0].inputs[0], node.inputs[0].inputs[1], node.inputs[1]))
    return ip


def fuse_flatten_ip(node):
    # type: (ts.Node) -> Optional[ts.Node]
    flatten = node.inputs[0]
    ip = node
    w = node.inputs[1]

    if flatten.has("dim") and int(flatten.get("dim")) != 1:
        return None

    new_ip = ts.graph.clone_bubble(ip)
    ts.Node.Link(new_ip, (flatten.inputs[0], w))

    return new_ip


def fuse_bias_reshape(node):
    # type: (ts.Node) -> Optional[ts.Node]
    bias = node.inputs[0]
    reshape = node

    shape = list(reshape.get("shape"))
    if len(shape) != 4 or shape[2] != 1 or shape[3] != 1:
        return None

    return bias


def change_sub_neg(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[1]

    return ts.menu.op(node.name, "neg", [x])


def change_relu6_x4(node):
    # type: (ts.Node) -> Optional[ts.Node]
    x = node.inputs[0]
    name = node.name

    channels = node.shape[1]
    max = float(node.get("max"))

    # change relu6 to x - (x - 6) * thresh(x, 6)
    relu = ts.zoo.relu(name + "_relu_", x)
    bias = ts.zoo.add_bias(name + "_bias_", relu, b=[-max, ] * channels, dim=1)
    thresh = ts.menu.op(name + "_thresh_", "threshold", [relu])
    thresh.set("threshold", max, numpy.float32)
    bias_x_thresh = ts.zoo.mul(name + "_bias_x_thresh", bias, thresh)
    relu6_x4 = ts.zoo.sub(name, relu, bias_x_thresh)

    if node.has("#dtype"):
        dtype = node.dtype
        relu.dtype = dtype
        bias.dtype = dtype
        thresh.dtype = dtype
        bias_x_thresh.dtype = dtype
        relu6_x4.dtype = dtype

    relu.shape = node.shape
    bias.shape = node.shape
    thresh.shape = node.shape
    bias_x_thresh.shape = node.shape
    relu6_x4.shape = node.shape

    return relu6_x4


def _get_caffe_fence():
    fence = Fence()
    fence.register(MetaGraph([
        "conv2d",
        ("add_bias", -1)
    ]), fuse_conv2d_bias)
    fence.register(MetaGraph([
        "depthwise_conv2d",
        ("add_bias", -1)
    ]), fuse_conv2d_bias)
    fence.register(MetaGraph([
        "inner_prod",
        ("add_bias", -1)
    ]), fuse_ip_bias)
    fence.register(MetaGraph([
        "flatten",
        ("inner_prod", -1)
    ]), fuse_flatten_ip)
    fence.register(MetaGraph([
        "add_bias",
        ("_reshape", -1)
    ]), fuse_bias_reshape)
    fence.register(MetaGraph([
        {"#op": ts.Node.Const, "value": EQ(0)},
        ({"#op": "sub", "#shape": HasShape(4)}, {0: -1})
    ]), change_sub_neg)
    fence.register(MetaNode(
        "relu_max"
    ), change_relu6_x4)
    return fence


class CaffeNode(object):
    class Top(object):
        def __init__(self, node, i, name):
            # type: (CaffeNode, int, str) -> None
            self.__index = i
            self.__node = node
            self.__name = name

        @property
        def node(self):
            # type: () -> CaffeNode
            return self.__node

        @property
        def index(self):
            # type: () -> int
            return self.__index

        @property
        def name(self):
            # type: () -> str
            return self.__name

    def __init__(self, type, name, bottoms=None, top_count=None):
        # type: (str, str, List[CaffeNode, Top], int) -> None
        if bottoms is None:
            bottoms = []
        if isinstance(bottoms, (CaffeNode, self.Top)):
            bottoms = [bottoms, ]
        if top_count is None:
            top_count = 1
        assert isinstance(type, basestring)
        assert isinstance(name, basestring)

        self.__type = str(type)
        self.__name = str(name)

        self.__proto = caffe.LayerParameter()
        self.__proto.type = self.__type
        self.__proto.name = self.__name
        self.__bottoms = bottoms
        self.__top_count = top_count
        self.__tops = [None] * top_count

    @property
    def type(self):
        # type: () -> str
        return self.__type

    @property
    def name(self):
        # type: () -> str
        return self.__name

    @name.setter
    def name(self, value):
        # type: (str) -> None
        self.__name = str(value)
        self.__proto.name = self.__name

    @property
    def proto(self):
        # type: () -> caffe.LayerParameter
        return self.__proto

    @proto.setter
    def proto(self, value):
        # type: (caffe.LayerParameter) -> None
        assert isinstance(value, caffe.LayerParameter)
        self.__proto = value

    @property
    def bottoms(self):
        # type: () -> List[Union[CaffeNode, Top]]
        return self.__bottoms

    @property
    def top_count(self):
        # type: () -> int
        return self.__top_count

    @property
    def tops(self):
        # type: () -> List[Top]
        return self.__tops

    def top(self, i, name):
        # type: (int, str) -> Top
        if not self.__tops[i]:
            self.__tops[i] = self.Top(self, i, name)
        return self.__tops[i]


node2converter = {
}


def register_node_converter(node, converter):
    # type: (str, CallableMeta) -> None
    """
    :param node:
    :param converter: assume as (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    :return:
    """
    node2converter[node] = converter


def convert2caffenode(node, cache=None):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    if cache is None:
        cache = {}
    if node in cache:
        return cache[node]
    op = node.op
    if op not in node2converter:
        raise NotImplementedError("Not support layer {}:{} {}".format(node.op, node.name, node))
    caffenode = node2converter[op](node, cache=cache)
    cache[node] = caffenode
    return caffenode


def _make_sure_all_layer_name_diff(outputs, inputs=None):
    # type: (List[Union[CaffeNode, CaffeNode.Top]], List[CaffeNode]) -> None
    if inputs is None:
        inputs = []

    named = set()
    set_layer_names = set()

    def new_name(name):
        if name in set_layer_names:
            i = 0
            while True:
                new_name = "{}_{}".format(name, i)
                if new_name in set_layer_names:
                    i += 1
                    continue
                name = new_name
                break
        set_layer_names.add(name)
        return name

    def do_rename(node):
        if node in named:
            return
        named.add(node)

        if isinstance(node, CaffeNode.Top):
            do_rename(node.node)
        elif isinstance(node, CaffeNode):
            node.name = new_name(node.name)
            for b in node.bottoms:
                do_rename(b)

    for node in list(inputs) + list(outputs):
        do_rename(node)


def _build_layers_setup_bottom_top(outputs, inputs):
    # type: (List[Union[CaffeNode, CaffeNode.Top]], List[CaffeNode]) -> List[CaffeNode]
    """

    :param nodes: List as inputs + outputs, the return list are input first
    :return:
    """
    _make_sure_all_layer_name_diff(outputs, inputs)

    nodes = inputs + outputs

    set_blob_names = set()
    map_node_top_name = {}
    layers = []

    def new_name(name):
        if name in set_blob_names:
            i = 0
            while True:
                new_name = "{}_{}".format(name, i)
                if new_name in set_blob_names:
                    i += 1
                    continue
                name = new_name
                break
        set_blob_names.add(name)
        return name

    def do_map_node_top_name(node):
        # type: (Union[CaffeNode, CaffeNode.Top]) -> None
        if node in map_node_top_name:
            return
        name = new_name(node.name)
        map_node_top_name[node] = name

        if isinstance(node, CaffeNode.Top):
            do_map_node_top_name(node.node)
        elif isinstance(node, CaffeNode):
            for b in node.bottoms:
                do_map_node_top_name(b)
            layers.append(node)

    for n in nodes:
        do_map_node_top_name(n)

    # rename report and cpu tailed blob name
    for k in map_node_top_name.keys():
        v = map_node_top_name[k]
        if v[-4:] == "_cpu":
            map_node_top_name[k] = new_name(v + "_hide")
        if v[-7:] == "_report" and k not in outputs:
            map_node_top_name[k] = new_name(v + "_hide")

    bottom_used_count = {}
    for n in layers:
        for bottom in n.bottoms:
            if bottom in bottom_used_count:
                bottom_used_count[bottom] += 1
            else:
                bottom_used_count[bottom] = 1

    for n in layers[::-1]:
        if isinstance(n, CaffeNode):
            if n in outputs:    # do not in-place output layer
                continue
            # batchnorm，scale，bias，relu，sigmoid，tanh、prelu、absval
            if str(n.proto.type) in {
                "BatchNorm", "Scale", "Bias", "ReLU", "Sigmoid", "TanH", "PReLU", "AbsVal",
                "ELU", "Exp", "Log", "Power", "RReLU"}:
                if n.bottoms[0] not in inputs and bottom_used_count[n.bottoms[0]] == 1:
                    if n.bottoms[0] not in outputs:     # do not change output node name
                        map_node_top_name[n.bottoms[0]] = map_node_top_name[n]

    # set output name by add report
    map_marked_names = {}
    for k in outputs:
        v = map_node_top_name[k]
        if v[-7:] != "_report":
            map_marked_names[v] = new_name(v + "_report")

    for n in layers:
        bottom_names = []
        top_names = []

        name = map_node_top_name[n]

        for i, top in enumerate(n.tops):
            assert top is None or isinstance(top, CaffeNode.Top)
            if top is None:
                if i == 0:
                    top_names.append(name)
                else:
                    top_names.append(new_name("{}_top_{}".format(name, i)))
            else:
                top_names.append(map_node_top_name[top])

        for i, bottom in enumerate(n.bottoms):
            bottom_names.append(map_node_top_name[bottom])

        for b in bottom_names:
            n.proto.bottom.append(b)
        for t in top_names:
            if t in map_marked_names:
                t = map_marked_names[t]
            n.proto.top.append(t)

    return layers


def convert(outputs, inputs, prototxt, caffemodel):
    # type: (List[ts.Node], List[ts.Node], str, str) -> None
    """
    outputs must be sorted, bottom output must be list first, no check for this
    :param outputs:
    :param inputs:
    :param prototxt:
    :param caffemodel:
    :return:
    """
    _, net_name = os.path.split(prototxt)
    print("[INFO]: --[== Translate network...")
    # 1. zip graph, convert each nodes
    cache = {}
    outputs = _get_caffe_fence().convert(outputs, cache)
    inputs = [cache[i] for i in inputs]
    # 2. write each proto node
    # 2.1 special for inputs

    print("[INFO]: --[== Convert network...")
    # 2.2 convert each nodes
    cache = OrderedDict()
    caffe_inputs = [convert2caffenode(i, cache=cache) for i in inputs]
    caffe_outputs = [convert2caffenode(o, cache=cache) for o in outputs]

    layers = _build_layers_setup_bottom_top(caffe_outputs, caffe_inputs)

    print("[INFO]: --[== Convert about {} layer(s). Start write files...".format(len(layers)))
    # 3. output
    # 3.1 build full net
    caffe_net = caffe.NetParameter()
    caffe_net.name = net_name
    for layer in layers:
        assert isinstance(layer, CaffeNode)
        caffe_net.layer.extend([layer.proto])

    # 3.2 split attrs and parameters
    with open(caffemodel, "wb") as f:
        f.write(caffe_net.SerializeToString())
    for layer in caffe_net.layer:
        while len(layer.blobs):
            layer.blobs.pop()
    with open(prototxt, "w") as f:
        f.write(str(caffe_net))

    print("[INFO]: --[== Write files done.".format(len(cache)))

    pass


def update_blob_shape(blob_shape, shape):
    # type: (caffe.BlobShape, List[int]) -> None
    while len(blob_shape.dim):
        blob_shape.dim.pop()
    for i in shape:
        blob_shape.dim.append(i)


def update_blob(blob, data):
    # type: (caffe.BlobProto, numpy.ndarray) -> None
    data = numpy.asarray(data, dtype=numpy.float32)
    update_blob_shape(blob.shape, data.shape)
    while len(blob.data):
        blob.data.pop()
    for datum in data.reshape([-1]):
        blob.data.append(datum)


def convert_field(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode.Top
    x = convert2caffenode(node.inputs[0], cache)
    i = int(node.get("offset"))
    return x.top(i, node.name)


register_node_converter("_field", convert_field)


def convert_param(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    cn = CaffeNode("Input", name=node.name)
    param = cn.proto.input_param

    shape = list(node.get("#shape"))

    if len(shape) < 1:
        raise ValueError("Input shape must be 4D, got {}".format(shape))

    for dim in shape[1:]:
        if dim <= 0:
            raise ValueError("Input shape must be definite, got {}".format(shape))

    if shape[0] <= 0:
        shape[0] = 1

    update_blob_shape(param.shape.add(), shape)

    return cn


register_node_converter("<param>", convert_param)

import math


def conv2d_forward(x, padding, dilation, kernel, stride):
    return int(math.floor((x + padding - (dilation * (kernel - 1) + 1)) / stride + 1))


def conv2d_backward(y, padding, dilation, kernel, stride):
    return (y - 1) * stride + (dilation * (kernel - 1) + 1) - padding


def pooling2d_forward(x, padding, kernel, stride):
    return int(math.ceil((x + padding - kernel) / float(stride) + 1))


def pooling2d_backward(y, padding, kernel, stride):
    return (y - 1) * stride + kernel - padding


def conv2d_same_padding(x, padding, dilation, kernel, stride):
    # type: (int, Tuple[int], int, int, int) -> Tuple[int]
    if padding[0] == padding[1]:
        return padding
    y = conv2d_forward(x, padding[0] + padding[1], dilation, kernel, stride)
    if y == conv2d_forward(x, padding[0] + padding[0], dilation, kernel, stride):
        return [padding[0], padding[0]]

    padding_min = conv2d_backward(y, x, dilation, kernel, stride)
    padding_max = padding_min + (stride - 1)

    padding_left_diff = stride * 2
    padding_left = None
    padding_right = None
    for i in range(padding_min, padding_max + 1):
        if i % 2 != 0:
            continue
        may_padding_left = i // 2
        may_padding_left_diff = abs(may_padding_left - padding[0])
        if may_padding_left_diff < padding_left_diff:
            padding_left_diff = may_padding_left_diff
            padding_left = may_padding_left
            padding_right = padding_left

    if padding_left is None or padding_right is None:
        raise ValueError("Conv2D can not same padding with: x={}, padding={}, dilation={}, kernel={}, stride={}".format(
            x, padding, dilation, kernel, stride
        ))

    return [padding_left, padding_right]


def pooling2d_same_padding(x, padding, kernel, stride):
    # type: (int, Tuple[int], int, int) -> Tuple[int]
    if padding[0] == padding[1]:
        return padding
    y = pooling2d_forward(x, padding[0] + padding[1], kernel, stride)
    if y == pooling2d_forward(x, padding[0] + padding[0], kernel, stride):
        return [padding[0], padding[0]]

    padding_max = pooling2d_backward(y, x, kernel, stride)
    padding_min = padding_max - (stride - 1)

    padding_left_diff = stride * 2
    padding_left = None
    padding_right = None
    for i in range(padding_min, padding_max + 1):
        if i % 2 != 0:
            continue
        may_padding_left = i // 2
        may_padding_left_diff = abs(may_padding_left - padding[0])
        if may_padding_left_diff < padding_left_diff:
            padding_left_diff = may_padding_left_diff
            padding_left = may_padding_left
            padding_right = padding_left

    if padding_left is None or padding_right is None:
        raise ValueError("Polling2d can not same padding with: x={}, padding={}, kernel={}, stride={}".format(
            x, padding, kernel, stride
        ))

    return [padding_left, padding_right]


def convert_conv2d(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Convolution", node.name, [x])
    param = cn.proto.convolution_param
    blobs = cn.proto.blobs

    format = str(node.get("format"))
    assert format == "NCHW"

    W = node.inputs[1].get("value")
    update_blob(blobs.add(), W)

    param.num_output = W.shape[0]
    param.group = 1

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    dilation = numpy.asarray(node.get("dilation")).reshape([-1])[-2:]
    kernel_size = W.shape[-2:]
    input_size = list(node.inputs[0].shape)[-2:]

    pad_h = conv2d_same_padding(input_size[0], padding[0], dilation[0], kernel_size[0], stride[0])
    pad_w = conv2d_same_padding(input_size[1], padding[1], dilation[1], kernel_size[1], stride[1])

    if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
        print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
            node.op, node.name, padding[0], padding[1], pad_h, pad_w
        ))

    padding[0, :] = pad_h
    padding[1, :] = pad_w

    if kernel_size[0] == kernel_size[1]:
        param.kernel_size.extend(kernel_size[-1:])
    else:
        param.kernel_size.extend(kernel_size[-2:])

    if dilation[0] == dilation[1]:
        param.dilation.extend(dilation[-1:])
    else:
        param.dilation.extend(dilation[-2:])

    if stride[0] == stride[1]:
        param.stride.extend(stride[-1:])
    else:
        param.stride.extend(stride[-2:])

    assert padding[0, 0] == padding[0, 1]
    assert padding[1, 0] == padding[1, 1]

    if padding[0, 0] == padding[1, 0]:
        param.pad.extend([padding[0, 0]])
    else:
        param.pad.extend([padding[0, 0], padding[1, 0]])

    if len(node.inputs) > 2:
        B = node.inputs[2].get("value")
        update_blob(blobs.add(), B)

        param.bias_term = True
    else:
        param.bias_term = False

    return cn


register_node_converter("conv2d", convert_conv2d)


def convert_add_bias(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Bias", node.name, [x])
    param = cn.proto.bias_param
    blobs = cn.proto.blobs

    format = None
    dim = None
    if node.has("dim"):
        dim = int(node.get("dim"))
    if node.has("format"):
        format = str(node.get("format"))

    if dim is None:
        if format is None:
            raise ValueError("add_bias must set format and dim")
        if format == "HCHW":
            dim = 1
        elif format == "NHWC":
            dim = 3
        else:
            raise ValueError("add_bias not support format {}".format(format))

    param.axis = dim
    param.num_axes = 1
    B = node.inputs[1].get("value")

    update_blob(blobs.add(), B)

    return cn


register_node_converter("add_bias", convert_add_bias)


def convert_pooling2d(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Pooling", node.name, [x])
    param = cn.proto.pooling_param
    blobs = cn.proto.blobs

    format = str(node.get("format"))
    assert format == "NCHW"

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    ksize = numpy.asarray(node.get("ksize")).reshape([-1])[-2:]
    type = int(node.get("type"))
    input_size = list(node.inputs[0].shape)[-2:]

    pad_h = pooling2d_same_padding(input_size[0], padding[0], ksize[0], stride[0])
    pad_w = pooling2d_same_padding(input_size[1], padding[1], ksize[1], stride[1])

    if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
        print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
            node.op, node.name, padding[0], padding[1], pad_h, pad_w
        ))

    padding[0, :] = pad_h
    padding[1, :] = pad_w

    if ksize[0] == ksize[1]:
        param.kernel_size = ksize[0]
    else:
        param.kernel_h = ksize[0]
        param.kernel_w = ksize[1]

    if stride[0] == stride[1]:
        param.stride = stride[0]
    else:
        param.stride_h = stride[0]
        param.stride_w = stride[1]

    assert padding[0, 0] == padding[0, 1]
    assert padding[1, 0] == padding[1, 1]

    if padding[0, 0] == padding[1, 0]:
        param.pad = padding[0, 0]
    else:
        param.pad_h = padding[0, 0]
        param.pad_w = padding[1, 0]

    if type == 0:
        param.pool = 0  # MAX
    elif type == 1:
        param.pool = 1  # AVG
    else:
        raise ValueError("pooling2d not supported pooling type: {}".format(type))

    return cn


register_node_converter("pooling2d", convert_pooling2d)


def convert_batch_scale(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Scale", node.name, [x])
    param = cn.proto.scale_param
    blobs = cn.proto.blobs

    scale = node.inputs[1].get("value")
    bias = node.inputs[2].get("value")
    dim = int(node.get("dim"))

    param.axis = dim
    param.num_axes = 1
    param.bias_term = True

    update_blob(blobs.add(), scale)
    update_blob(blobs.add(), bias)

    return cn


register_node_converter("batch_scale", convert_batch_scale)


def convert_batch_norm(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("BatchNorm", node.name, [x])
    param = cn.proto.batch_norm_param
    blobs = cn.proto.blobs

    mean = node.inputs[1].get("value")
    var = node.inputs[2].get("value")
    dim = int(node.get("dim"))
    epsilon = 1e-5
    if node.has("epsilon"):
        epsilon = float(node.get("epsilon"))

    if abs(epsilon - 1e-5) > 1e-7:
        var = var + epsilon - 1e-5

    assert dim == 1

    # param.eps = epsilon
    param.use_global_stats = True

    update_blob(blobs.add(), mean)
    update_blob(blobs.add(), var)
    update_blob(blobs.add(), [1])

    return cn


register_node_converter("batch_norm", convert_batch_norm)


def convert_add(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    y = convert2caffenode(node.inputs[1], cache)
    cn = CaffeNode("Eltwise", node.name, [x, y])
    param = cn.proto.eltwise_param
    blobs = cn.proto.blobs

    # 0-PROD, 1-SUM, 2-MAX
    param.operation = 1

    return cn


register_node_converter("add", convert_add)


def convert_relu(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("ReLU", node.name, [x, ])

    return cn


register_node_converter("relu", convert_relu)
# register_node_converter("relu_max", convert_relu)


def convert_inner_prod(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("InnerProduct", node.name, [x])
    param = cn.proto.inner_product_param
    blobs = cn.proto.blobs

    W = node.inputs[1].get("value")

    transpose = False
    if node.has("transpose"):
        transpose = bool(node.get("transpose"))

    if transpose:
        param.num_output = W.shape[0]
    else:
        param.num_output = W.shape[1]

    if not transpose:
        W = numpy.transpose(W)
        transpose = not transpose

    param.transpose = not transpose

    update_blob(blobs.add(), W)
    if len(node.inputs) > 2:
        B = node.inputs[2].get("value")
        update_blob(blobs.add(), B)

        param.bias_term = True
    else:
        param.bias_term = False

    return cn


register_node_converter("inner_prod", convert_inner_prod)
register_node_converter("caffe:inner_prod", convert_inner_prod)


def convert_concat(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = [convert2caffenode(i, cache) for i in node.inputs]
    cn = CaffeNode("Concat", node.name, x)
    param = cn.proto.concat_param
    blobs = cn.proto.blobs

    dim = int(node.get("dim"))
    param.axis = dim

    return cn


register_node_converter("concat", convert_concat)


def convert_neg(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Power", node.name, [x])
    param = cn.proto.power_param
    blobs = cn.proto.blobs
    # Use Power layer: power = 1, scale = -1.0, shift = 0
    param.power = 1
    param.scale = -1
    param.shift = 0

    return cn


register_node_converter("neg", convert_neg)


def convert_transpose(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Permute", node.name, [x])
    param = cn.proto.permute_param
    blobs = cn.proto.blobs

    permute = list(node.get("permute"))

    param.order.extend(permute)

    return cn


register_node_converter("_transpose", convert_transpose)


def convert_depthwise_conv2d_group(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Convolution", node.name, [x])
    param = cn.proto.convolution_param
    blobs = cn.proto.blobs

    format = str(node.get("format"))
    assert format == "NCHW"

    W = node.inputs[1].get("value")

    number_filters = W.shape[0]
    input_channels = W.shape[1]
    output_channels = number_filters * input_channels

    W = numpy.transpose(W, (1, 0, 2, 3))    # change number filter to dim 1
    W = numpy.reshape(W, [output_channels, 1, W.shape[2], W.shape[3]])

    update_blob(blobs.add(), W)

    param.num_output = output_channels
    param.group = input_channels

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    dilation = numpy.asarray(node.get("dilation")).reshape([-1])[-2:]
    kernel_size = W.shape[-2:]
    input_size = list(node.inputs[0].shape)[-2:]

    pad_h = conv2d_same_padding(input_size[0], padding[0], dilation[0], kernel_size[0], stride[0])
    pad_w = conv2d_same_padding(input_size[1], padding[1], dilation[1], kernel_size[1], stride[1])

    if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
        print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
            node.op, node.name, padding[0], padding[1], pad_h, pad_w
        ))

    padding[0, :] = pad_h
    padding[1, :] = pad_w

    if kernel_size[0] == kernel_size[1]:
        param.kernel_size.extend(kernel_size[-1:])
    else:
        param.kernel_size.extend(kernel_size[-2:])

    if dilation[0] == dilation[1]:
        param.dilation.extend(dilation[-1:])
    else:
        param.dilation.extend(dilation[-2:])

    if stride[0] == stride[1]:
        param.stride.extend(stride[-1:])
    else:
        param.stride.extend(stride[-2:])

    assert padding[0, 0] == padding[0, 1]
    assert padding[1, 0] == padding[1, 1]

    if padding[0, 0] == padding[1, 0]:
        param.pad.extend([padding[0, 0]])
    else:
        param.pad.extend([padding[0, 0], padding[1, 0]])

    if len(node.inputs) > 2:
        B = node.inputs[2].get("value")
        update_blob(blobs.add(), B)

        param.bias_term = True
    else:
        param.bias_term = False

    return cn


def convert_depthwise_conv2d_nnie(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("DepthwiseConv", node.name, [x])
    param = cn.proto.convolution_param
    blobs = cn.proto.blobs

    format = str(node.get("format"))
    assert format == "NCHW"

    W = node.inputs[1].get("value")

    number_filters = W.shape[0]
    input_channels = W.shape[1]
    output_channels = number_filters * input_channels

    W = numpy.transpose(W, (1, 0, 2, 3))    # change number filter to dim 1
    W = numpy.reshape(W, [output_channels, 1, W.shape[2], W.shape[3]])

    update_blob(blobs.add(), W)

    param.num_output = output_channels
    # param.group = input_channels  # no group parameter for depthwise

    padding = numpy.asarray(node.get("padding")).reshape([-1, 2])[-2:]
    stride = numpy.asarray(node.get("stride")).reshape([-1])[-2:]
    dilation = numpy.asarray(node.get("dilation")).reshape([-1])[-2:]
    kernel_size = W.shape[-2:]
    input_size = list(node.inputs[0].shape)[-2:]

    pad_h = conv2d_same_padding(input_size[0], padding[0], dilation[0], kernel_size[0], stride[0])
    pad_w = conv2d_same_padding(input_size[1], padding[1], dilation[1], kernel_size[1], stride[1])

    if pad_h[0] != padding[0, 0] or pad_w[0] != padding[1, 0]:
        print("[WARNING]: Layer {}:{} change padding [{}, {}] => [{}, {}]".format(
            node.op, node.name, padding[0], padding[1], pad_h, pad_w
        ))

    padding[0, :] = pad_h
    padding[1, :] = pad_w

    if kernel_size[0] == kernel_size[1]:
        param.kernel_size.extend(kernel_size[-1:])
    else:
        param.kernel_size.extend(kernel_size[-2:])

    if dilation[0] == dilation[1]:
        param.dilation.extend(dilation[-1:])
    else:
        param.dilation.extend(dilation[-2:])

    if stride[0] == stride[1]:
        param.stride.extend(stride[-1:])
    else:
        param.stride.extend(stride[-2:])

    assert padding[0, 0] == padding[0, 1]
    assert padding[1, 0] == padding[1, 1]

    if padding[0, 0] == padding[1, 0]:
        param.pad.extend([padding[0, 0]])
    else:
        param.pad.extend([padding[0, 0], padding[1, 0]])

    if len(node.inputs) > 2:
        B = node.inputs[2].get("value")
        update_blob(blobs.add(), B)

        param.bias_term = True
    else:
        param.bias_term = False

    return cn


def convert_depthwise_conv2d(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    W = node.inputs[1].get("value")
    number_filters = W.shape[0]
    if number_filters > 1:
        return convert_depthwise_conv2d_group(node, cache)
    return convert_depthwise_conv2d_nnie(node, cache)


register_node_converter("depthwise_conv2d", convert_depthwise_conv2d)


def convert_softmax(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Softmax", node.name, [x])
    param = cn.proto.softmax_param
    blobs = cn.proto.blobs

    dim = int(node.get("dim"))
    if dim < 0:
        dim += 4

    param.axis = dim

    return cn


register_node_converter("softmax", convert_softmax)


def convert_reshape(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Reshape", node.name, [x])
    param = cn.proto.reshape_param
    blobs = cn.proto.blobs

    shape = list(node.get("shape"))
    assert len(shape) == 4

    assert shape[0] == 0 or shape[0] == 1

    shape[0] = 0    # do not reshape N

    update_blob_shape(param.shape, shape)

    return cn


register_node_converter("_reshape", convert_reshape)


def convert_sub(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    y = convert2caffenode(node.inputs[1], cache)
    cn = CaffeNode("Eltwise", node.name, [x, y])
    param = cn.proto.eltwise_param
    blobs = cn.proto.blobs

    # 0-PROD, 1-SUM, 2-MAX
    param.operation = 1
    param.coeff.extend([1, -1])

    return cn


register_node_converter("sub", convert_sub)


def convert_mul(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    y = convert2caffenode(node.inputs[1], cache)
    cn = CaffeNode("Eltwise", node.name, [x, y])
    param = cn.proto.eltwise_param
    blobs = cn.proto.blobs

    # 0-PROD, 1-SUM, 2-MAX
    param.operation = 0

    return cn


register_node_converter("mul", convert_mul)


def convert_threshold(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)
    cn = CaffeNode("Threshold", node.name, [x])
    param = cn.proto.threshold_param
    blobs = cn.proto.blobs

    param.threshold = float(node.get("threshold"))

    return cn


register_node_converter("threshold", convert_threshold)


def convert_copy(node, cache):
    # type: (ts.Node, Dict[ts.Node, CaffeNode]) -> CaffeNode
    x = convert2caffenode(node.inputs[0], cache)

    if len(node.inputs[0].outputs) > 1:
        return x
    elif x.type == "Input":   # do not cover input layer
        # use input layer directly, assume that no only split graph
        return x
    else:
        x.name = node.name
        return x


register_node_converter("_copy", convert_copy)
