#!/usr/bin/env python

"""
:author Kier
"""

from typing import List, Dict

from .node import Node

from .binio import read_int
from .binio import read_int_list
from .binio import write_int
from .binio import write_int_list

from .node import write_bubble
from .node import read_bubble

import numpy
from typing import Union, List, Tuple, Set
import copy


class Graph(object):
    def __init__(self):
        self.__nodes = []

    def make(self, op=None, name=None, output_count=None):
        node = Node(op=op, name=name, output_count=output_count)
        self.__nodes.append(node)
        return node

    def clear(self):
        self.__nodes = []

    @property
    def nodes(self):
        return self.__nodes


def write_nodes(stream, nodes, base=0):
    # type: (file, list[Node], int) -> None
    # build node_index_map
    index = base
    node_index_map = {}
    for node in nodes:
        node_index_map[node] = index
        index += 1

    # write number of nodes and each nodes
    write_int(stream=stream, i=len(nodes))
    for node in nodes:
        write_bubble(stream=stream, node=node)
        write_int_list(stream=stream, a=[node_index_map[input] for input in node.inputs])


def read_nodes(stream, base=0):
    # type: (file, int) -> list[Node]
    # read number of nodes
    nodes = []
    list_of_inputs = []
    size = read_int(stream=stream)
    while size > 0:
        nodes.append(read_bubble(stream=stream))
        list_of_inputs.append(read_int_list(stream=stream))
        size -= 1

    for i in range(len(nodes)):
        node = nodes[i]
        inputs = [nodes[j - base] for j in list_of_inputs[i]]
        Node.Link(node=node, inputs=inputs)

    return nodes


def plot_graph(node, plot=None):
    # type: (Union[Node, List[Node]], Set[Node]) -> None
    if plot is None:
        plot = set()

    if not isinstance(node, (tuple, list)):
        node = [node,]

    for x in node:
        assert isinstance(x, Node)
        if x in plot:
            continue
        plot_graph(x.inputs, plot)

    for x in node:
        assert isinstance(x, Node)
        if x in plot:
            continue
        if x.op == Node.Const:
            data = x.get("value")
            data = numpy.asarray(data)
            print("{}: {} = {}".format(x.op, x.name, data.shape))
        else:
            print("{}: {} -> {}".format(x.op, [i.name for i in x.inputs], x.name))
        plot.add(x)


def clone_node(node, cache=None, tips=None):
    # type: (Node, Dict[Node, Node], Dict[Node, Node]) -> Node
    """
    :param node:
    :param cache: cache means each cloned nodes
    :param tips: contain some key-value pairs, if clone key, return value's copy instead
    :return:
    """
    if cache is None:
        cache = {}
    if tips is None:
        tips = {}

    if node in cache:
        return cache[node]

    if node in tips:
        dolly = clone_node(tips[node], cache=cache, tips=tips)
        cache[node] = dolly
        return dolly

    dolly = Node(op=node.op, name=node.name, shape=node.shape)

    for k in node.params.keys():
        v = node.params[k]
        dolly.set(k, copy.copy(v))

    dolly_inputs = [cache[node] if node in cache else clone_node(node, cache=cache, tips=tips) for node in node.inputs]
    Node.Link(dolly, dolly_inputs)

    cache[node] = dolly

    return dolly


def clone_graph(nodes, cache=None, tips=None):
    # type: (List[Node], Dict[Node, Node], Dict[Node, Node]) -> List[Node]
    if cache is None:
        cache = {}
    if tips is None:
        tips = {}
    return [cache[node] if node in cache else clone_node(node, cache=cache, tips=tips) for node in nodes]


def clone_bubble(node):
    # type: (Node) -> Node
    """
    :param node:
    :return:
    """
    dolly = Node(op=node.op, name=node.name, shape=node.shape)

    for k in node.params.keys():
        v = node.params[k]
        dolly.set(k, copy.copy(v))

    return dolly


def walk_graph(nodes, endpoints=None, cache=None):
    # type: (Union[ts.Node, List[ts.Node]], Set[ts.Node], Set[ts.Node]) -> Tuple[Set[ts.Node], Set[ts.Node]]
    """
    :param nodes: start walk nodes
    :param endpoints: end nodes set, if walking on end nodes set, return the node in return value's second value
    :return: tuple of 2 set: (in graph nodes, in endpoints nodes or input nodes)
    """
    if endpoints is None:
        endpoints = set()
    if cache is None:
        cache = set()
    if isinstance(nodes, Node):
        nodes = [nodes, ]

    in_graph = set()
    in_ends = set()
    for node in nodes:
        # .1 check cache
        if node in cache:
            continue
        # .2 check if end
        if node in endpoints:
            in_ends.add(node)
            cache.add(node)
            continue
        # .3 recursively traversing each input
        cache.add(node)
        in_graph.add(node)
        sub_in_graph, sub_in_ends = walk_graph(node.inputs, endpoints, cache)
        in_graph |= sub_in_graph
        in_ends |= sub_in_ends
    return in_graph, in_ends
