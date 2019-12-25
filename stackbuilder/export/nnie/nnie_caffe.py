import tensorstack as ts

from .proto import caffe_pb2 as caffe

from typing import List
import os
from collections import OrderedDict
import numpy

from stackfence.fence import Fence
from stackfence.metanode import *


def fuse_conv2d_bias(node):
    # type: (ts.Node) -> ts.Node
    conv2d = ts.graph.clone_bubble(node.inputs[0])
    conv2d.name = node.name
    ts.Node.Link(conv2d, (node.inputs[0].inputs[0], node.inputs[0].inputs[1], node.inputs[1]))
    return conv2d


def _get_caffe_fence():
    fence = Fence()
    fence.register(MetaGraph([
        "conv2d",
        ("add_bias", -1)
    ]), fuse_conv2d_bias)
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

    @property
    def proto(self):
        # type: () -> caffe.LayerParameter
        return self.__proto

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


def _build_layers_setup_bottom_top(outputs, inputs):
    # type: (List[Union[CaffeNode, CaffeNode.Top]]) -> List[CaffeNode]
    """

    :param nodes: List as inputs + outputs, the return list are input first
    :return:
    """
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

    for n in layers[::-1]:
        if isinstance(n, CaffeNode):
            if str(n.proto.type) in {"ELU", "Exp", "Log", "Power", "PReLU", "ReLU", "Sigmoid", "TanH", "RReLU"}:
                if n.bottoms[0] not in inputs:
                    map_node_top_name[n.bottoms[0]] = map_node_top_name[n]

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
            n.proto.top.append(t)

    return layers


def convert(outputs, inputs, prototxt, caffemodel):
    # type: (List[ts.Node], List[ts.Node], str, str) -> None
    _, net_name = os.path.split(prototxt)
    # 1. zip graph, convert each nodes
    cache = {}
    outputs = _get_caffe_fence().convert(outputs, cache)
    inputs = [cache[i] for i in inputs]
    # 2. write each proto node
    # 2.1 special for inputs

    # 2.2 convert each nodes
    # TODO: deal with copy with split
    cache = OrderedDict()
    caffe_inputs = [convert2caffenode(i, cache=cache) for i in inputs]
    caffe_outputs = [convert2caffenode(o, cache=cache) for o in outputs]

    layers = _build_layers_setup_bottom_top(caffe_outputs, caffe_inputs)

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

    shape = node.get("#shape")
    update_blob_shape(param.shape.add(), list(shape))

    return cn


register_node_converter("<param>", convert_param)


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
        param.pool = 0
    elif type == 1:
        param.pool = 1
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
