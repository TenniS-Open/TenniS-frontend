from typing import CallableMeta, Union, List, Dict, Set, Tuple

import tensorstack as ts


class PriorityList(object):
    def __init__(self, index=None):
        if index is None:
            index = lambda x: x
        self.__index = index
        self.__list = []

    def append(self, x):
        # type: (object) -> int
        x_i = self.__index(x)
        N = len(self.__list)
        i = N
        for j in range(N):
            item = self.__list[j]
            item_i = self.__index(item)
            if x_i > item_i:
                i = j
                break
        self.__list.insert(i, x)
        return i

    def __len__(self):
        return len(self.__list)

    def __iter__(self):
        return self.__list.__iter__()

    def clear(self):
        self.__list = []


class Fence(object):
    def __init__(self):
        # list of tuple, contain tuple of priority, checker and converter
        self.__fence = PriorityList(lambda x: x[0])

    def clear(self):
        self.__fence.clear()

    def register(self, checker, converter, priority=0):
        # type: (CallableMeta, CallableMeta, int) -> None
        """
        :param checker: (Node) -> bool, check if Node is target node
        :param converter: (Node) -> Union[Node, None], None for not converted, Node for converted Node
        :param priority: priority, bigger number first
        :return: None
        Notice: if node converted, must return new node, can not build new nod on it
        """
        self.__fence.append((priority, checker, converter))

    def _walk_graph(self, nodes, endset=None, cache=None):
        # type: (Union[ts.Node, List[ts.Node]], Set[ts.Node], Set[ts.Node]) -> Tuple[Set[ts.Node], Set[ts.Node]]
        """
        :param nodes: start walk nodes
        :param endset: end nodes set, if walking on end nodes set, return the node in return value's second value
        :return: tuple of 2 set: (in graph nodes, in endset nodes)
        """
        if endset is None:
            endset = set()
        if cache is None:
            cache = set()
        if isinstance(nodes, ts.Node):
            nodes = [nodes, ]

        in_graph = set()
        in_ends = set()
        for node in nodes:
            # .1 check cache
            if node in cache:
                continue
            # .2 check if end
            if node in endset:
                in_ends.add(node)
                cache.add(node)
                continue
            # .3 recursively traversing each input
            cache.add(node)
            in_graph.add(node)
            sub_in_graph, sub_in_ends = self._walk_graph(node.inputs, endset, cache)
            in_graph |= sub_in_graph
            in_ends |= sub_in_ends
        return in_graph, in_ends

    def _can_update2(self, node, cvt_node, graph, update=True):
        # type: (ts.Node, ts.Node, Set[ts.Node], bool) -> bool
        # Check if updated graph including middle graph output.
        # If updated graph has middle output, than do not updated
        cvt_graph, cvt_graph_inputs = self._walk_graph(cvt_node, graph)
        origin_graph, origin_graph_inputs = self._walk_graph(node, cvt_graph_inputs)
        # outer link check
        for n in graph:
            if n in origin_graph:
                continue
            for i in n.inputs:
                if i != node and i in origin_graph:
                    return False
        if update:
            graph -= origin_graph
            graph |= cvt_graph
        return True

    def _convert_hard(self, node, graph):
        # type: (ts.Node, Set[ts.Node]) -> Union[ts.Node, None]
        # find first checker
        for i, chc, cvt in self.__fence:
            if chc(node):
                cvt_node = cvt(node)
                if cvt_node is not None and cvt_node != node:
                    if self._can_update2(node, cvt_node, graph):
                        # update node only if converted node has no effect
                        return cvt_node
        return None

    def _convert_cached(self, node, graph, cache=None):
        # type: (ts.Node, Set[ts.Node], Dict[ts.Node, ts.Node]) -> Union[ts.Node, None]
        if cache is not None and node in cache:
            return cache[node]
        cvt_node = self._convert_hard(node, graph=graph)
        if cvt_node is None:
            cvt_node = node
        if cache is not None:
            cache[node] = cvt_node
        return cvt_node

    def _convert_updated(self, node, graph, cache=None):
        # type: (ts.Node, Set[ts.Node], Dict[ts.Node, ts.Node]) -> Union[ts.Node, None]
        # Replace Node in graph
        cvt_node = self._convert_cached(node, graph, cache=cache)
        if cvt_node == node:
            return cvt_node
        # Careful replace node with cvt_node
        ts.Node.Replace(node, cvt_node)
        return cvt_node

    def _convert(self, node, graph, cache=None):
        # type: (ts.Node, Set[ts.Node], Dict[ts.Node, ts.Node]) -> Union[ts.Node, None]
        assert isinstance(node, ts.Node)
        if cache is None:
            cache = {}

        # 1. try convert node firstly
        cvt_node = self._convert_updated(node, graph, cache)

        # 2. loop convert node inplace if node converted
        if cvt_node is not None and cvt_node != node:
            # Recursive call self, loop update graph
            cvt_cvt_node = self._convert(cvt_node, graph, cache)
            if cvt_cvt_node is not None and cvt_cvt_node != cvt_node:
                return cvt_cvt_node

        for i in range(len(cvt_node.inputs)):
            # convert input, no need to return
            # no need to re-link node as that _convert_updated used
            self._convert(cvt_node.inputs[i], graph, cache)

        return cvt_node if cvt_node != node else None

    def _convert_check_none(self, node, graph, cache=None):
        # type: (ts.Node, Set[ts.Node], Dict[ts.Node, ts.Node]) -> ts.Node
        cvt_node = self._convert(node, graph, cache=cache)
        if cvt_node is None:
            return node
        return cvt_node

    def convert(self, node, cache=None):
        # type: (Union[ts.Node, List[ts.Node]], Dict[ts.Node, ts.Node]) -> Union[ts.Node, List[ts.Node]]
        """
        Convert node and list of node
        :param node: ts.Node of list of ts.Node, ready to convert
        :param cache: map of original node to converted node
        :return: converted node or list of node
        """
        graph, _ = self._walk_graph(node)
        if isinstance(node, ts.Node):
            return self._convert_check_none(node, graph, cache)
        elif isinstance(node, (list, tuple)):
            return [self._convert_check_none(i, graph, cache) for i in node]
        else:
            raise ValueError()


if __name__ == "__main__":
    from stackfence.metanode import *
    a = ts.menu.param("a")
    b = ts.menu.op("b", "b", [a, ])
    c = ts.menu.op("c", "c", [b, ])
    d = ts.menu.op("d", "d", [c, ])
    e = ts.menu.op("e", "e", [c, d])
    # a
    # |
    # b
    # |
    # c
    # |\
    # | d
    # |/
    # e

    print("=================")
    ts.graph.plot_graph(e)

    f = Fence()
    f.register(MetaGraph([
        {"#op": "c"}
    ]), lambda x: ts.menu.op("x", "x", x.inputs))
    f.register(MetaGraph([
        {"#op": "t"},
        ({"#op": "t"}, -1),
    ]), lambda x: ts.menu.op(x.name, "t", x.inputs[0].inputs))
    f.register(MetaGraph([
        {"#op": "b"},
        ({"#op": "c"}, -1),
        ({"#op": "d"}, -1),
    ]), lambda x: ts.menu.op(x.name, "d", x.inputs[0].inputs))

    e = f.convert(e)

    print("=================")
    ts.graph.plot_graph(e)
