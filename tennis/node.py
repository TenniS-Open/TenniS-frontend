#!/usr/bin/env python

"""
:author Kier
"""

import struct
from .tensor import compatible_string
from collections import OrderedDict

from .tensor import write_tensor
from .tensor import read_tensor
from .tensor import from_any
from .tensor import to_int
from .tensor import to_str

from .dtype import VOID

import numpy
from typing import List, Tuple, Union


class Node(object):
    Parameter = "<param>"
    Const = "<const>"
    Variable = "<var>"

    class RetentionParam(object):
        name = "#name"
        op = "#op"
        output_count = "#output_count"
        shape = "#shape"
        dtype = "#dtype"

    def __init__(self, op=None, name=None, shape=None):
        self.__op = "" if op is None else op
        self.__name = "" if name is None else name
        # self.__output_count = 1 if output_count is None else output_count
        # self.__output_count = numpy.asarray(self.__output_count, numpy.int32)
        self.__shape = shape
        self.__params = {
            self.RetentionParam.name: self.__name,
            self.RetentionParam.op: self.__op,
            # self.RetentionParam.output_count: self.__output_count,
        }
        if self.__shape is not None:
            self.__shape = numpy.asarray(self.__shape, numpy.int32)
            self.__params[self.RetentionParam.shape] = self.__shape
        self.__inputs = []
        self.__outputs = []

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value
        self.__params[self.RetentionParam.name] = self.__name

    @property
    def op(self):
        return self.__op

    @op.setter
    def op(self, value):
        self.__op = value
        self.__params[self.RetentionParam.op] = self.__op

    @property
    def output_count(self):
        return 1

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        self.__shape = from_any(value, dtype=numpy.int32)
        self.__params[self.RetentionParam.shape] = self.__shape

    @property
    def params(self):
        return self.__params

    @property
    def dtype(self):
        if self.RetentionParam.dtype in self.__params:
            return int(self.__params[self.RetentionParam.dtype])
        return VOID

    @dtype.setter
    def dtype(self, value):
        self.__dtype = from_any(value, dtype=numpy.int32)
        self.__params[self.RetentionParam.dtype] = self.__dtype

    def has(self, param):
        return param in self.__params

    def set(self, param, value, dtype=None):
        self.__params[param] = from_any(value, dtype=dtype)

    def get(self, param):
        return self.__params[param]

    def try_get(self, param, value):
        if param in self.__params:
            return self.params[param]
        return value

    def clear(self, param):
        del self.__params[param]

    def clear_params(self):
        self.__params.clear()

    @property
    def inputs(self):
        # type: () -> List[Node]
        return self.__inputs

    @property
    def outputs(self):
        # type: () -> List[Node]
        return self.__outputs

    @staticmethod
    def Link(node, inputs):
        # type: (Node, Union[Node, List[Node]]) -> None
        """
        :param node: Node
        :param inputs: single Node of list of Node
        :return: None
        """
        assert isinstance(inputs, (Node, tuple, list)), "Input nodes must be node or list of nodes"

        # in case of link the node twice
        for i in node.inputs:
            if node in i.__outputs:
                i.__outputs.remove(node)
        node.__inputs = []

        if isinstance(inputs, (list, tuple)):
            node.__inputs = list(inputs)
            for input in inputs:
                assert isinstance(input, Node)
                input.__outputs.append(node)
        elif isinstance(inputs, Node):
            input = inputs
            node.__inputs = [input]
            input.__outputs.append(node)
        else:
            raise Exception("Input nodes must be node or list of nodes")

    @staticmethod
    def _list_replace(a, old, new):
        # type: (List, object, object) -> List
        for i in range(len(a)):
            if a[i] == old:
                a[i] = new
        return a

    @staticmethod
    def Replace(old, new):
        # type: (Node, Node) -> None
        assert isinstance(old, Node)
        assert isinstance(new, Node)
        if old == new:
            return

        if old in new.inputs:
            for o in old.outputs:
                if o == new:
                    continue
                Node._list_replace(o.__inputs, old, new)
            for o in old.outputs:
                if o == new:
                    continue
                if o not in new.outputs:
                    new.__outputs.append(o)
            old.__outputs = [new, ]
            return

        if new in old.inputs:
            pass

        # common case: three things to be done,
        # 1. change each old.outputs' inputs.
        # 2. merge old.outputs to new.outputs
        # 3. clear old.outputs
        for o in old.outputs:
            Node._list_replace(o.__inputs, old, new)
        for o in old.outputs:
            if o not in new.outputs:
                new.__outputs.append(o)
        old.__outputs = []

    def __str__(self):
        return str(self.__params)

    def __repr__(self):
        return str(self.__params)


def write_string(stream, s):
    # type: (file, [str, bytes]) -> None
    s = compatible_string(s)
    if isinstance(s, str):
        s = s.encode()
    elif isinstance(s, bytes):
        pass
    else:
        raise Exception("Can not write type={} as string".format(type(s)))
    stream.write(struct.pack("=i%ds" % len(s), len(s), s))


def __write_int(stream, i):
    # type: (file, int) -> None
    stream.write(struct.pack("=i", i))


def __read_int(stream):
    # type: (file) -> int
    return int(struct.unpack('=i', stream.read(4))[0])


def read_string(stream):
    # type: (file) -> str
    size = __read_int(stream=stream)
    s = struct.unpack('=%ds' % size, stream.read(size))[0]
    return str(s.decode())


def write_bubble(stream, node):
    # type: (file, Node) -> None
    params = node.params
    stream.write(struct.pack("=i", len(params)))
    ordered_keys = list(params.keys())
    ordered_keys.sort()
    for k in ordered_keys:
        v = params[k]
        write_string(stream=stream, s=k)
        write_tensor(stream=stream, tensor=from_any(v))


def read_bubble(stream):
    # type: (file) -> Node
    params = {}
    size = __read_int(stream=stream)
    while size > 0:
        k = read_string(stream=stream)
        v = read_tensor(stream=stream)
        params[k] = v
        size -= 1
    output_count = 1 if Node.RetentionParam.output_count not in params else params[Node.RetentionParam.output_count]
    if output_count != 1:
        raise Exception("All operators' output count must be 1.")
    node = Node(op=to_str(params[Node.RetentionParam.op]),
                name=to_str(params[Node.RetentionParam.name]),
                shape=None if Node.RetentionParam.shape not in params else params[Node.RetentionParam.shape])
    for k in params.keys():
        node.set(k, params[k])
    return node


if __name__ == '__main__':
    node = Node(op='sum', name='C')
    node.set("str", "v:str")
    node.set("int", 16)
    node.set("float", 3.4)

    with open("bubble.txt", "wb") as fo:
        write_bubble(fo, node)

    with open("bubble.txt", "rb") as fi:
        local_node = read_bubble(fi)
        print(local_node.params)
