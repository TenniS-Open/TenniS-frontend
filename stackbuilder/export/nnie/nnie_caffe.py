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

    for n in nodes[::-1]:
        if isinstance(n, CaffeNode):
            if str(n.proto.type) in {"ELU", "Exp", "Log", "Power", "PReLU", "ReLU", "Sigmoid", "TanH", "RReLU"}:
                if n.bottoms[0] not in inputs:
                    map_node_top_name[n.bottoms[0]] = map_node_top_name[n]

    for n in nodes:
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

    update_blob(blobs.add(), (10, 20, 30, 40))

    return cn


register_node_converter("conv2d", convert_conv2d)
