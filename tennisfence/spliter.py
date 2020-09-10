from typing import Callable


from tennisfence.metanode import *

import sys
if sys.version > '3':
    from queue import Queue
else:
    from Queue import Queue

from collections import deque as Deque
from typing import Optional, Iterable

import logging


class SubGraph(object):
    def __init__(self, outputs, endpoints=None):
        # type: (Union[ts.Node, List[ts.Node]], Set[ts.Node]) -> None
        """
        Sub graph is not full graph, the input node is concat of output
        :param outputs: are sub-graph node nodes
        :param endpoints: a set of may endpoints, not equals the outputs
        """
        self.__outputs = list(outputs)
        nodes, inputs = self._walk_graph(outputs, endpoints)
        self.__inputs = inputs
        self.__nodes = nodes

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
            if node in endset or node.op == ts.Node.Parameter:
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

    @property
    def nodes(self):
        # type: () -> List[ts.Node]
        return self.__nodes

    @property
    def inputs(self):
        # type: () -> List[ts.Node]
        return self.__inputs

    @property
    def outputs(self):
        # type: () -> List[ts.Node]
        return self.__outputs

    def sort_inputs(self, inputs):
        inputs = list(inputs)
        old_inputs = set(self.__inputs)
        new_inputs = set(inputs)
        old_not_in_new = old_inputs - new_inputs
        if len(old_not_in_new) > 0:
            raise ValueError("param 1 must contain {}".format(old_not_in_new))
        new_not_in_old = new_inputs - old_inputs
        if len(new_not_in_old) > 0:
            raise ValueError("{} not in graph inputs".format(new_not_in_old))
        self.__inputs = inputs


class MainGraph(SubGraph):
    SubGraphOp = "tmp::submodule"

    def __init__(self, outputs, graphs, submodule_op=None):
        # type: (Union[ts.Node, List[ts.Node]], List[SubGraph], str) -> None
        """
        Sub graph is not full graph, the input node is concat of output
        :param outputs: are sub-graph node nodes
        :param endpoints: a set of may endpoints, not equals the outputs
        """
        if submodule_op is None:
            submodule_op = self.SubGraphOp
        super(MainGraph, self).__init__(outputs=outputs)
        self.__subnodes = [None, ] * len(graphs)
        self.__submodules = [g for g in graphs]
        for node in self.nodes:
            if node.op == submodule_op:
                index = int(node.get("#index"))
                if index >= len(graphs):
                    raise ValueError("Submodule count mismatch")
                self.__subnodes[index] = node
        if not all(self.__subnodes):
            raise ValueError("Submodule count mismatch")

    def sub_count(self):
        # type: () -> int
        return len(self.__submodules)

    def sub_node(self, i):
        # type: (int) -> ts.Node
        node = self.__subnodes[i]
        assert isinstance(node, ts.Node)
        return node

    def sub_graph(self, i):
        # type: (int) -> SubGraph
        return self.__submodules[i]


class RefCache(object):
    def __init__(self, outputs):
        # type: (List[ts.Node]) -> None
        # 1. build schedule
        schedule = self._get_computation_schedule(outputs)
        # 2. build full ref map[A, nodes relied by A]
        map_node_refs = {}
        for node in schedule:
            refs = set()
            for i in node.inputs:
                refs.add(i)
                refs |= map_node_refs[i]
            map_node_refs[node] = refs
        self.__map_node_refs = map_node_refs

    def _get_computation_schedule(self, outputs):
        # type: (List[ts.Node]) -> List[ts.Node]
        schedule = []
        walked = set()

        def add_schedule(node):
            if node in walked:
                return
            for i in node.inputs:
                if i not in walked:
                    add_schedule(i)
            schedule.append(node)
            walked.add(node)

        for node in outputs:
            add_schedule(node)

        return schedule

    def ref(self, top, bottom):
        # type: (ts.Node, ts.Node) -> bool
        """
        Return if top relies on bottom
        :param top:
        :param bottom:
        :return:
        """
        if top not in self.__map_node_refs:
            raise ValueError("param 1 not in graph")
        return bottom in self.__map_node_refs[top]

    def get_refs(self, node):
        # type: (ts.Node) -> Set[ts.Node]
        if node not in self.__map_node_refs:
            raise ValueError("param 1 not in graph")
        return self.__map_node_refs[node]


class TopFirstQueue(object):
    def __init__(self, ref):
        # type: (RefCache) -> None
        self.ref = ref
        self.__list = []

    def __contains__(self, item):
        return item in self.__list

    def extend(self, iter):
        # type: (Iterable) -> None
        for item in iter:
            self.append(item)

    def append(self, node):
        # type: (ts.Node) -> None
        self.__list.append(node)

    def empty(self):
        # type: () -> bool
        return len(self.__list) == 0

    def __len__(self):
        # type: () -> int
        return len(self.__list)

    def pop(self):
        # type: () -> Optional[ts.Node]
        if self.empty():
            return None
        L = self.__list
        i = 0
        n = L[i]
        for j in range(1, len(L)):
            if self.ref.ref(L[j], n):
                i = j
                n = L[i]
        del self.__list[i]
        return n


class BottomFirstQueue(object):
    def __init__(self, ref):
        # type: (RefCache) -> None
        self.ref = ref
        self.__list = []

    def __contains__(self, item):
        return item in self.__list

    def extend(self, iter):
        # type: (Iterable) -> None
        for item in iter:
            self.append(item)

    def append(self, node):
        # type: (ts.Node) -> None
        self.__list.append(node)

    def empty(self):
        # type: () -> bool
        return len(self.__list) == 0

    def __len__(self):
        # type: () -> int
        return len(self.__list)

    def pop(self):
        # type: () -> Optional[ts.Node]
        if self.empty():
            return None
        L = self.__list
        i = 0
        n = L[i]
        for j in range(1, len(L)):
            if self.ref.ref(n, L[j]):
                i = j
                n = L[i]
        del self.__list[i]
        return n


class OrderRecoder(object):
    def __init__(self):
        self.__map_item_order = {}
        self.__serial = 0

    def append(self, item):
        self.__map_item_order[item] = self.__serial
        self.__serial += 1

    def extend(self, iter):
        for item in iter:
            self.append(item)

    def clear(self):
        self.__map_item_order = {}
        self.__serial = 0

    def __contains__(self, item):
        return item in self.__map_item_order

    def __getitem__(self, item):
        return self.__map_item_order[item]


class GraphSpliter(object):
    def __init__(self, single_input=False, single_output=False,
                 logging_level=logging.INFO,
                 min_graph_size=0,
                 only_max_graph_out=False,
                 log_end_nodes=False):
        logging.basicConfig(level=logging_level, format='%(asctime)s %(name)s [%(levelname)s]: %(message)s')
        self.logger = logging.getLogger("GraphSpliter")

        self.__support_set = []
        self.__route_set = []

        self.__single_input = single_input
        self.__single_output = single_output
        self.__min_graph_size = min_graph_size
        self.__only_max_graph_out = only_max_graph_out
        self.__log_end_nodes = log_end_nodes

    @property
    def single_input(self):
        # type: () -> bool
        return self.__single_input

    @property
    def single_output(self):
        # type: () -> bool
        return self.__single_output

    def support(self, checker):
        # type: (Union[List[Union[str, Callable]], str, Callable]) -> GraphSpliter
        if not isinstance(checker, (tuple, list)):
            checker = [checker, ]
        for i in checker:
            if callable(i):
                self.__support_set.append(i)
            elif isinstance(i, basestring):
                self.__support_set.append(MetaNode(i))
            else:
                raise ValueError("param must be str of check function, or list of above")

    def route(self, checker):
        # type: (Union[List[Union[str, Callable]], str, Callable]) -> GraphSpliter
        if not isinstance(checker, (tuple, list)):
            checker = [checker, ]
        for i in checker:
            if callable(i):
                self.__route_set.append(i)
            elif isinstance(i, basestring):
                self.__route_set.append(MetaNode(i))
            else:
                raise ValueError("param must be str of check function, or list of above")

    def is_support(self, node):
        # type: (ts.Node) -> bool
        for checker in self.__support_set:
            if checker(node):
                return True
        return False

    def is_route(self, node):
        # type: (ts.Node) -> bool
        for checker in self.__route_set:
            if checker(node):
                return True
        return False

    def is_route_or_support(self, node):
        # type: (ts.Node) -> bool
        return self.is_route(node) or self.is_support(node)

    def _clone_node(self, node, cache=None, tips=None):
        # type: (ts.Node, Dict[ts.Node, ts.Node], Dict[ts.Node, ts.Node]) -> ts.Node
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
            dolly = self._clone_node(tips[node], cache=cache, tips=tips)
            cache[node] = dolly
            return dolly

        dolly = ts.Node(op=node.op, name=node.name, shape=node.shape)

        for k in node.params.keys():
            v = node.params[k]
            dolly.set(k, v)

        dolly_inputs = [cache[node] if node in cache else self._clone_node(node, cache=cache, tips=tips)
                        for node in node.inputs]
        ts.Node.Link(dolly, dolly_inputs)

        cache[node] = dolly

        return dolly

    def _clone_graph(self, nodes, cache=None, tips=None):
        # type: (List[ts.Node], Dict[ts.Node, ts.Node], Dict[ts.Node, ts.Node]) -> List[ts.Node]
        if cache is None:
            cache = {}
        if tips is None:
            tips = {}
        return [cache[node] if node in cache else self._clone_node(node, cache=cache, tips=tips) for node in nodes]

    def _explore(self, node, ref, dead=None):
        # type: (ts.Node, RefCache, Set[ts.Node]) -> Tuple[Optional[List[ts.Node]], Optional[List[ts.Node]]]
        """
        :param node: start nodes
        :param ref: graph nodes refs
        :param dead: node ready in other graph, or walked
        :return: sub graph's output nodes, sub graph's input nodes(end points)
        Notice, return None, None for the node is not suitable for split
        """
        MAX_GRAPH = sys.maxsize

        if dead is None:
            dead = {}

        if node in dead or not self.is_route_or_support(node):
            return None, None

        start = node
        # contains graph input nodes
        graph_inputs = set()
        # contains graph output nodes
        graph_outputs = set()
        # contains all graph node
        graph_nodes = set()

        recoder = OrderRecoder()

        if self.single_output:
            graph_set = {start, }   # contains all nodes in graph
            input_set = set()     # not support nodes, as graph inputs
            output_set = {start, }

            walked = set()
            walked.add(start)
            recoder.append(start)
            walking = TopFirstQueue(ref)
            walking.extend(start.inputs)
            recoder.extend(start.inputs)

            while len(walking) > 0:
                n = walking.pop()
                if n in walked:
                    continue
                walked.add(n)

                if n not in dead and \
                        self.is_route_or_support(n) and \
                        not any([ref.ref(t, n) for t in input_set]) and \
                        not any([ref.ref(n, t) for t in output_set]) and \
                        len(set(n.outputs) - graph_set) == 0 and \
                        len(graph_set) < MAX_GRAPH:
                    # not output node should be included in graph
                    graph_set.add(n)    # add to graph set, already check before in walking
                    for i in n.inputs:
                        if i not in walked:
                            walking.append(i)
                            recoder.append(i)
                else:
                    input_set.add(n)

            graph_outputs = {start}
            graph_inputs = input_set
            graph_nodes = graph_set
        else:
            # multi outputs explore, walk output
            graph_set = {start, }   # contains all nodes in graph
            input_set = set()     # not support nodes, as graph inputs
            output_set = {start, }  # may output later, filter after all found

            next_input_deque = TopFirstQueue(ref)
            next_output_deque = BottomFirstQueue(ref)
            next_input_empty = False
            next_output_empty = False

            walked = set()
            walked.add(start)
            recoder.append(start)
            next_input_deque.extend(start.inputs)
            recoder.extend(start.inputs)

            while True:
                next_input_empty = len(next_input_deque) == 0
                if next_input_empty and next_output_empty:
                    break

                while len(next_input_deque) > 0:
                    n = next_input_deque.pop()
                    assert isinstance(n, ts.Node)
                    if n in walked:
                        continue
                    walked.add(n)
                    if n not in dead and \
                            self.is_route_or_support(n) and \
                            not any([ref.ref(t, n) for t in input_set]) and \
                            not any([ref.ref(n, t) for t in output_set]) and \
                            len(graph_set) < MAX_GRAPH:
                        graph_set.add(n)    # add to graph set, already check before in walking
                        for i in n.inputs:
                            if i not in walked:
                                next_input_deque.append(i)
                                recoder.append(i)
                        for o in n.outputs:
                            if o not in walked:
                                next_output_deque.append(o)
                                recoder.append(o)
                    else:
                        input_set.add(n)

                next_output_empty = len(next_output_deque) == 0
                if next_input_empty and next_output_empty:
                    break

                while len(next_output_deque) > 0:
                    n = next_output_deque.pop()
                    assert isinstance(n, ts.Node)
                    if n in walked:
                        continue
                    walked.add(n)
                    if n not in dead and \
                            self.is_route_or_support(n) and \
                            not any([ref.ref(t, n) for t in input_set]) and \
                            not any([ref.ref(n, t) for t in output_set]) and \
                            len(graph_set) < MAX_GRAPH:
                        graph_set.add(n)    # add to graph set, already check before in walking
                        for i in n.inputs:
                            if i not in walked:
                                next_input_deque.append(i)
                                recoder.append(i)
                        for o in n.outputs:
                            if o not in walked:
                                next_output_deque.append(o)
                                recoder.append(o)
                    else:
                        output_set.add(n)

            # now input_set and graph_set ready, check output_set
            graph_outputs = {start}
            for o in output_set:
                if o == start:  # start must be output
                    continue
                for i in o.inputs:
                    if i in graph_set:
                        graph_outputs.add(i)    # end point in graph node is output node
            graph_inputs = input_set
            graph_nodes = graph_set

        # base walked, first check if satisfied single input or single single output
        if self.single_input and len(graph_inputs) > 1:
            # change sub graph to single input
            while len(graph_inputs) > 1:
                if len(graph_nodes) == 1:
                    # last node, not satisfied, return no sub graph
                    return None, None
                # find an node can be remove from graph
                # a.1 gather node can may remove
                may_remove = set()
                for n in graph_inputs:
                    for o in n.outputs:
                        if o in graph_nodes:
                            may_remove.add(o)
                # a.1.1 sort may_remove list
                last_remove = may_remove & SubGraph([start, ], graph_inputs).nodes
                first_remove = may_remove - last_remove
                may_remove = list(first_remove) + list(last_remove)
                # a.2 find target, input's count first and ref node first, output node last
                target = may_remove[0]
                for i in range(1, len(may_remove)):
                    if ref.ref(target, may_remove[i]):
                        target = may_remove[i]
                # a.3 dual with remove node: edit graph set, inputs
                must_remain_inputs = set()
                for n in (set(may_remove) - {target}):
                    must_remain_inputs |= set(n.inputs)
                after_inputs = graph_inputs & must_remain_inputs | {target}
                graph_nodes.remove(target)
                graph_inputs = after_inputs
                if target in graph_outputs:
                    graph_outputs.remove(target)

        # check if sub graph still has nodes
        if len(graph_nodes) == 0:
            return None, None

        # check if all nodes are route nodes
        has_support = False
        for n in graph_nodes:
            if self.is_support(n):
                has_support = True
                break

        if not has_support:
            return None, None

        graph_inputs = list(graph_inputs)
        graph_outputs = list(graph_outputs)

        if len(graph_inputs) > 1:
            graph_inputs.sort(key=lambda x: recoder[x])
        if len(graph_outputs) > 1:
            graph_outputs.sort(key=lambda x: recoder[x], reverse=True)

        # sort outputs, ensure bottom node first
        sorted_graph_outputs = []
        while len(graph_outputs) > 0:
            i = 0
            n = graph_outputs[i]
            for j in range(1, len(graph_outputs)):
                if ref.ref(n, graph_outputs[j]):
                    i = j
                    n = graph_outputs[i]
            sorted_graph_outputs.append(n)
            del graph_outputs[i]
        graph_outputs = sorted_graph_outputs

        return graph_outputs, graph_inputs

    def _count(self, iter, func):
        n = 0
        for i in iter:
            if func(i):
                n += 1
        return n

    def _be_related_core(self, ref, a, b):
        # type: (RefCache, SubGraph, SubGraph) -> bool
        a_graph_refs = set(a.inputs)
        for i in a.inputs:
            a_graph_refs |= ref.get_refs(i)
        b_graph_outputs = set(b.outputs)
        return len(a_graph_refs & b_graph_outputs) > 0

    def _be_related(self, ref, a, b):
        # type: (RefCache, SubGraph, SubGraph) -> bool
        return self._be_related_core(ref, a, b) or self._be_related_core(ref, b, a)

    def split(self, outputs, inputs=None):
        # type: (Union[ts.Node, List[ts.Node]], List[ts.Node]) -> MainGraph
        """
        :param outputs: output nodes
        :param inputs: input nodes, equals two parameters pass to Module.sort_inputs
        :return: MainGraph, contains List of SubGraph (Length N) contains the module operator:
        operator attrs has: `#index` `Int`, the sub-graph offset in return array, must be in [0, N - 1]
                            `#name`: `String` is sub graph output name, or list of output names concat by `&`
                            `#op`: `String` MainGraph.SubGraphOp
        """
        if isinstance(outputs, ts.Node):
            outputs = [outputs, ]
        elif isinstance(outputs, (tuple, list)):
            pass
        else:
            raise ValueError("param 1 must ts.Node or list of ts.Node")

        # ========================================================================================= #
        # 0. clone nodes, make sure all outputs in graph
        cache = {}
        outputs = self._clone_graph(nodes=outputs, cache=cache)
        if inputs is not None:
            inputs = [cache[i] if isinstance(i, ts.Node) else i for i in inputs]

        logic_output = ts.Node("output", "output")
        ts.Node.Link(logic_output, outputs)

        ref = RefCache(outputs=[logic_output])

        # ========================================================================================= #
        walked_set = set()   # contains all split nodes
        walked_set.add(logic_output)
        supported_graphs = []   # list of SubGraph
        # 1. walk each node, split it to supported graph or original graph
        #     If found an supported graph, build a SubGraph
        #     Use Queue check each node, if an node is supported, than walk all linked node, summery sub graph
        walking = TopFirstQueue(ref)
        for n in outputs:
            walking.append(n)
        while not walking.empty():
            n = walking.pop()
            assert isinstance(n, ts.Node)
            if n in walked_set:
                continue

            # self.logger.debug("Walking on {}".format(n))

            if n.op != ts.Node.Const and self.is_route_or_support(n):
                # now, walk all supported nodes
                sub_outputs, sub_inputs = self._explore(n, ref, walked_set)
                # check if this node can be split to an sub-graph
                if sub_outputs is not None:
                    # build SubGraph
                    sub_graph = SubGraph(sub_outputs, set(sub_inputs))
                    sub_graph.sort_inputs(sub_inputs)
                    walked_set |= set(sub_graph.nodes)
                    for i in sub_graph.inputs:
                        walking.append(i)
                    supported_graphs.append(sub_graph)
                    continue

            # here, n is not support node
            walked_set.add(n)
            for i in n.inputs:
                walking.append(i)
            # next loop

        # ========================================================================================= #
        # Optimize sub graphs, merge each two not related graph
        if not self.single_input and not self.single_output:
            # merge not loop graph
            while len(supported_graphs) > 1:
                N = len(supported_graphs)
                merged = False
                for i in range(N - 1):
                    for j in range(i + 1, N):
                        # check if i, j nodes can be merge
                        g1 = supported_graphs[i]
                        g2 = supported_graphs[j]
                        if not self._be_related(ref, g1, g2):
                            merge_inputs = set(g1.inputs) | set(g2.inputs)
                            merge_outputs = set(g1.outputs) | set(g2.outputs)
                            merge_graph = SubGraph(list(merge_outputs), merge_inputs)
                            merge_graph.sort_inputs(list(merge_inputs))
                            supported_graphs[i] = merge_graph
                            del supported_graphs[j]
                            merged = True
                            break
                    if merged:
                        break
                if not merged:
                    break
            pass

        # filter graph size
        if self.__min_graph_size > 1:
            # check if satisfied the min graph size
            great_supported_graphs = []
            for sub_graph in supported_graphs:
                if self._count(sub_graph.nodes, self.is_support) >= self.__min_graph_size:
                    great_supported_graphs.append(sub_graph)
            supported_graphs = great_supported_graphs

        # if only save one sub graph
        if self.__only_max_graph_out:
            # deal with max graph out
            if len(supported_graphs) > 1:
                i = 0
                g = supported_graphs[i]
                max_count = self._count(g.nodes, self.is_support)
                for j in range(1, len(supported_graphs)):
                    g = supported_graphs[j]
                    count = self._count(g.nodes, self.is_support)
                    if count > max_count:
                        i = j
                        max_count = count
                supported_graphs = [supported_graphs[i], ]

        # ========================================================================================= #
        # sort sub graphs, relied first
        # no sort now, use clone tips parameter fix the clone bug
        # remove logic output
        ts.Node.Link(logic_output, [])

        # 2. build each graph to submodule,
        # 2.1 build node in map, use clone graph method
        # 2.1.1. clone sub graph
        main_sub_graphs = []
        for i, sub_graph in enumerate(supported_graphs):
            sub_cache = {}
            for sub_graph_input in sub_graph.inputs:
                sub_cache[sub_graph_input] = ts.menu.param(
                    sub_graph_input.name, sub_graph_input.shape)
            sub_outputs = self._clone_graph(sub_graph.outputs, sub_cache)
            sub_inputs = [sub_cache[i] for i in sub_graph.inputs]
            g = SubGraph(sub_outputs, sub_inputs)
            g.sort_inputs(sub_inputs)
            main_sub_graphs.append(g)

        if self.__log_end_nodes:
            end_plot = set()

            for n, sub_graph in enumerate(supported_graphs):
                print("[====]: Sub graph {}: input={}, output={}".format(
                    n, len(sub_graph.inputs), len(sub_graph.outputs)))
                for g_i in sub_graph.inputs:
                    if g_i in end_plot:
                        continue
                    end_plot.add(g_i)
                    if g_i.op == ts.Node.Parameter:
                        continue
                    print("[====]: Sub graph bottom: {}:{}".format(g_i.op, g_i.name))
                for g_o in sub_graph.outputs:
                    for o in g_o.outputs:
                        if o in end_plot:
                            continue
                        end_plot.add(o)
                        if o in sub_graph.nodes:
                            continue
                        print("[====]: Sub graph top: {}:{}".format(o.op, o.name))

            if len(end_plot) == 0:
                print("[====]: Sub graph = main graph.")
                pass

        # 2.1.2 clone main graph
        main_tips = {}
        for i, sub_graph in enumerate(supported_graphs):
            if len(sub_graph.outputs) == 1:
                module = ts.Node(MainGraph.SubGraphOp,
                                 sub_graph.outputs[0].name,
                                 sub_graph.outputs[0].shape)
                if sub_graph.outputs[0].has("#dtype"):
                    module.dtype = sub_graph.outputs[0].dtype
                if sub_graph.outputs[0].has("#shape"):
                    module.shape = sub_graph.outputs[0].shape
                module_inputs = sub_graph.inputs
                ts.Node.Link(module, module_inputs)
                module.set("#index", i)
                main_tips[sub_graph.outputs[0]] = module
            else:
                module = ts.Node(MainGraph.SubGraphOp,
                                 "&".join([n.name for n in sub_graph.outputs]))
                module_inputs = sub_graph.inputs
                ts.Node.Link(module, module_inputs)
                module.set("#index", i)
                for j, o in enumerate(sub_graph.outputs):
                    output_field = ts.menu.field(o.name, module, j)
                    if o.has("#dtype"):
                        output_field.dtype = o.dtype
                    if o.has("#shape"):
                        output_field.shape = o.shape
                    main_tips[o] = output_field

            # self.logger.info(module)

        main_cache = {}
        main_outputs = self._clone_graph(outputs, cache=main_cache, tips=main_tips)

        main_graph = MainGraph(outputs=main_outputs, graphs=main_sub_graphs)
        if inputs:
            main_graph.sort_inputs([main_cache[i] for i in inputs])

        # 3. return main graph
        return main_graph


if __name__ == "__main__":
    a = ts.menu.param("a")
    b = ts.menu.data("b", 0)
    c = ts.menu.op("c", "c", [a, b])
    d = ts.menu.op("d", "d", [c, ])
    e = ts.menu.op("e", "e", [c, d])
    f = ts.menu.op("f", "f", [e, ])
    g = ts.menu.op("g", "g", [e, ])
    # a b
    # |/
    # c
    # |\
    # | d
    # |/
    # e
    # |\
    # f g

    in_list = [a]
    out_list = [f, g]
    op_list = ["c", "d", "e", "f", "g"]
    single_input = False
    single_output = False
    min_graph_size = 0

    debug_sup_list = None
    # debug_sup_list = ['e', 'f', 'd', 'c']

    print("=================")
    ts.graph.plot_graph(out_list)
    for flag in range(0, 1 << len(op_list)):
        sup_list = []
        for i in range(len(op_list)):
            if flag & (1 << i):
                sup_list.append(op_list[i])

        if debug_sup_list is not None:
            sup_list = debug_sup_list

        print("============== Support: {} =============".format(sup_list))
        spliter = GraphSpliter(logging_level=logging.DEBUG,
                               single_input=single_input, single_output=single_output,
                               min_graph_size=min_graph_size)
        spliter.route(ts.Node.Const)
        for op in sup_list:
            spliter.support(op)
        graph = spliter.split(out_list, in_list)
        print("+++++++++++++++++")
        ts.graph.plot_graph(graph.outputs)
        for i in range(graph.sub_count()):
            print("---------------")
            ts.graph.plot_graph(graph.sub_graph(i).outputs)

        if debug_sup_list is not None:
            break
