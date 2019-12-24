from typing import CallableMeta


from stackfence.metanode import *

import sys
if sys.version > '3':
    from queue import Queue
else:
    from Queue import Queue

from collections import deque as Deque

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


class GraphSpliter(object):
    def __init__(self, single_input=False, single_output=False,
                 logging_level=logging.INFO,
                 min_graph_size=0,
                 only_max_graph_out=False):
        logging.basicConfig(level=logging_level, format='%(asctime)s %(name)s [%(levelname)s]: %(message)s')
        self.logger = logging.getLogger("GraphSpliter")

        self.__support_set = []
        self.__route_set = []

        self.__single_input = single_input
        self.__single_output = single_output
        self.__min_graph_size = min_graph_size
        self.__only_max_graph_out = only_max_graph_out

    @property
    def single_input(self):
        # type: () -> bool
        return self.__single_input

    @property
    def single_output(self):
        # type: () -> bool
        return self.__single_output

    def support(self, checker):
        # type: (Union[List[Union[str, CallableMeta]], str, CallableMeta]) -> GraphSpliter
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
        # type: (Union[List[Union[str, CallableMeta]], str, CallableMeta]) -> GraphSpliter
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

        dolly_inputs = self._clone_graph(node.inputs, cache=cache, tips=tips)
        ts.Node.Link(dolly, dolly_inputs)

        cache[node] = dolly

        return dolly

    def _clone_graph(self, nodes, cache=None, tips=None):
        # type: (List[ts.Node], Dict[ts.Node, ts.Node], Dict[ts.Node, ts.Node]) -> List[ts.Node]
        if cache is None:
            cache = {}
        if tips is None:
            tips = {}
        return [self._clone_node(node, cache=cache, tips=tips) for node in nodes]

    def _explore(self, node, ref, dead=None, final=None):
        # type: (ts.Node, RefCache, Set[ts.Node], Set[ts.Node]) -> Tuple[List[ts.Node], List[ts.Node]]
        """
        :param node: start nodes
        :param ref: graph nodes refs
        :param dead: node ready in other graph, or walked
        :return: sub graph's output nodes, sub graph's input nodes(end points)
        Notice, return None, None for the node is not suitable for split
        """
        if dead is None:
            dead = {}
        if final is None:
            final = set()
        final.add(node)

        start = node
        # contains graph input nodes
        graph_inputs = set()
        # contains graph output nodes
        graph_outputs = set()
        # contains all graph node
        graph_nodes = set()

        if self.single_output:
            graph_set = set()   # contains all nodes in graph
            input_set = set()     # not support nodes, as graph inputs

            walked = set()
            re_walk_count = 0   # means now many re_walk node in walking set

            walking = Deque()
            walking.append(node)
            while len(walking) > 0 and re_walk_count < len(walking):
                n = walking.popleft()
                if n in walked:
                    continue
                walked.add(n)

                if n not in dead and \
                        self.is_route_or_support(n) and \
                        not any([ref.ref(t, n) for t in input_set]):
                    # check if n can be output node
                    if n != start:
                        n_outputs = set(n.outputs)
                        n_not_in_graph_outputs = n_outputs - graph_set
                        if len(n_not_in_graph_outputs) > 0:
                            walked.remove(n)
                            walking.append(n)  # re-walk
                            re_walk_count += 1
                            continue
                    re_walk_count = 0

                    # not output node should be included in graph
                    graph_set.add(n)    # add to graph set, already check before in walking
                    for i in n.inputs:
                        if self.is_route_or_support(i):
                            walking.append(i)
                        else:
                            input_set.add(i)
                            walked.add(i)
                else:
                    input_set.add(n)

            while len(walking) > 0:
                n = walking.popleft()
                if n in walked:
                    continue
                walked.add(n)
                input_set.add(n)

            graph_outputs = {start}
            graph_inputs = input_set
            graph_nodes = graph_set
        else:
            # multi outputs explore, walk output
            graph_set = set()   # contains all nodes in graph
            input_set = set()     # not support nodes, as graph inputs
            output_set = set()  # may output later, filter after all found

            walked = set()

            next_input_deque = Deque()
            next_output_deque = Deque()
            next_input_empty = False
            next_output_empty = False

            next_input_deque.append(start)
            output_set.add(start)

            while True:
                next_input_empty = len(next_input_deque) == 0
                if next_input_empty and next_output_empty:
                    break

                while len(next_input_deque) > 0:
                    n = next_input_deque.popleft()
                    assert isinstance(n, ts.Node)
                    if n in walked:
                        continue
                    walked.add(n)
                    if n not in dead and \
                            self.is_route_or_support(n) and \
                            not any([ref.ref(t, n) for t in input_set]):
                        # check if n can be output node
                        n_outputs = set(n.outputs)
                        n_not_in_graph_outputs = n_outputs - graph_set
                        if len(n_not_in_graph_outputs) > 0:
                            output_set.add(n)

                        graph_set.add(n)    # add to graph set, already check before in walking
                        for i in n.inputs:
                            if self.is_route_or_support(i):
                                next_input_deque.append(i)
                            else:
                                input_set.add(i)
                                walked.add(i)
                        for o in n.outputs:
                            if o not in dead and o not in walked:
                                next_output_deque.append(o)
                    else:
                        input_set.add(n)

                next_output_empty = len(next_output_deque) == 0
                if next_input_empty and next_output_empty:
                    break

                not_sure_output_list =[]
                sure_output_count = 0
                while len(next_output_deque) > 0:
                    n = next_output_deque.pop()
                    assert isinstance(n, ts.Node)
                    if n in walked:
                        continue
                    walked.add(n)
                    if n not in dead and \
                            self.is_route_or_support(n) and \
                            not any([ref.ref(t, n) for t in input_set]):
                        n_inputs = set(n.inputs)
                        n_unsatisfied_inputs = n_inputs - input_set - graph_set
                        if len(n_unsatisfied_inputs) == 0:
                            output_set.add(n)
                            graph_set.add(n)
                            sure_output_count += 1
                            for o in n.outputs:
                                if o not in dead and o not in walked:
                                    next_output_deque.append(o)
                            for i in n.inputs:
                                if i not in dead and i not in walked:
                                    next_input_deque.appendleft(i)
                        else:
                            not_sure_output_list.append(n)
                    else:
                        # if not support nothing to do
                        pass
                if sure_output_count > 0:
                    for o in not_sure_output_list:
                        next_output_deque.append(o)

            # now input_set and graph_set ready, check output_set
            graph_outputs = set()
            for o in output_set:
                o_outputs = set(o.outputs)
                n_not_in_graph_outputs = o_outputs - graph_set
                if len(n_not_in_graph_outputs) > 0 or o in final:
                    graph_outputs.add(o)
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

        return list(graph_outputs), list(graph_inputs)

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

        ref = RefCache(outputs=outputs)

        # ========================================================================================= #
        walked_set = set()   # contains all split nodes
        supported_graphs = []   # list of SubGraph
        all_graph_nodes = []    # contains graph's all outputs
        # 1. walk each node, split it to supported graph or original graph
        #     If found an supported graph, build a SubGraph
        #     Use Queue check each node, if an node is supported, than walk all linked node, summery sub graph
        walking = Queue()
        for n in outputs:
            walking.put(n)
        while not walking.empty():
            n = walking.get()
            assert isinstance(n, ts.Node)
            if n in walked_set:
                continue

            # self.logger.debug("Walking on {}".format(n))

            if n.op != ts.Node.Const and self.is_route_or_support(n):
                # now, walk all supported nodes
                sub_outputs, sub_inputs = self._explore(n, ref, walked_set, set(outputs))
                # check if this node can be split to an sub-graph
                if sub_outputs is not None:
                    # build SubGraph
                    sub_graph = SubGraph(sub_outputs, set(sub_inputs))
                    sub_graph.sort_inputs(sub_inputs)
                    walked_set |= set(sub_graph.nodes)
                    for i in sub_graph.inputs:
                        walking.put(i)
                    supported_graphs.append(sub_graph)
                    continue

            walked_set.add(n)
            for i in n.inputs:
                walking.put(i)
            # next loop

        # ========================================================================================= #
        # Optimize sub graphs
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

        # 2.1.2 clone main graph
        main_tips = {}
        for i, sub_graph in enumerate(supported_graphs):
            if len(sub_graph.outputs) == 1:
                module = ts.Node(MainGraph.SubGraphOp,
                                 sub_graph.outputs[0].name,
                                 sub_graph.outputs[0].shape)
                module_inputs = sub_graph.inputs
                ts.Node.Link(module, module_inputs)
                module.set("#index", i)
                main_tips[sub_graph.outputs[0]] = module
            else:
                module = ts.Node(MainGraph.SubGraphOp,
                                 "&".join([n.name for n in sub_graph.outputs]),
                                 sub_graph.outputs[0].shape)
                module_inputs = sub_graph.inputs
                ts.Node.Link(module, module_inputs)
                module.set("#index", i)
                for j, o in enumerate(sub_graph.outputs):
                    main_tips[o] = ts.menu.field(o.name, module, j)
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
