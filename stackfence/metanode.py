from typing import CallableMeta, Union, List, Tuple, Set, SupportsInt, Dict

import numpy
import tensorstack as ts

Tensor = Union[int, float, numpy.ndarray, ts.tensor.StringTensor, ts.tensor.PackedTensor]


class MetaRule(object):
    def __init__(self, rule, msg=None):
        # type: (CallableMeta, str) -> None
        if not callable(rule):
            raise ValueError("param 1 must be callable")
        self.__rule = rule
        if msg is not None and not isinstance(msg, str):
            msg = str(msg)
        self.__msg = msg

    def __call__(self, value):
        # type: (Union[Tensor, None]) -> bool
        """
        None if attr not defined
        :param value:
        :return:
        """
        try:
            return self.__rule(value)
        except:
            return False

    def __and__(self, other):
        # type: (MetaRule) -> MetaRule
        def logic(x):
            return self.__call__(x) and other(x)
        return MetaRule(logic, "({} and {})".format(self, other))

    def __or__(self, other):
        # type: (MetaRule) -> MetaRule
        def logic(x):
            return self.__call__(x) or other(x)
        return MetaRule(logic, "({} or {})".format(self, other))

    def __invert__(self):
        # type: (MetaRule) -> MetaRule
        def logic(x):
            return not self.__call__(x)
        return MetaRule(logic, "not ".format(self))

    def __repr__(self):
        if self.__msg is None:
            # return super(MetaRule, self).__repr__()
            return "CustomRule"
        return self.__msg

    def __str__(self):
        if self.__msg is None:
            # return super(MetaRule, self).__str__()
            return "CustomRule"
        return self.__msg


class HasShape(MetaRule):
    @classmethod
    def GetShape(cls, value):
        # type: (Tensor) -> tuple
        if isinstance(value, (basestring, ts.tensor.StringTensor)):
            return len(str(value)),
        if isinstance(value, (int, str)):
            return tuple()
        if isinstance(value, numpy.ndarray):
            return value.shape
        if isinstance(value, (list, tuple)):
            try:
                array = numpy.asarray(value)
                return array.shape
            except:
                raise MetaValueMismatch(type(value), "tensor", value)
        raise MetaValueMismatch(type(value),
                                "[int, float, list, tuple, numpy.ndarray, basestring, ts.tensor.StringTensor]")

    def __init__(self, *shape):
        # type: (List[int]) -> None
        if len(shape) == 1:
            if not isinstance(shape[0], int):
                shape = shape[0]
        if not isinstance(shape, (list, tuple)):
            raise ValueError("shape must be list of Int")
        for dim in shape:
            if not isinstance(dim, int):
                raise ValueError("shape must be list of Int")
        shape = tuple(shape)
        super(HasShape, self).__init__(lambda x: self.GetShape(x) == shape,
                                       "shape{}".format(shape))


class MetaValueMismatch(Exception):
    def __init__(self, got, expected, value=None):
        if value is None:
            super(MetaValueMismatch, self).__init__("{} expected, got {}".format(expected, got))
        else:
            super(MetaValueMismatch, self).__init__("{}: {} expected, got {}".format(value, expected, got))


NotSet = MetaRule(lambda x: x is None, "notset")
HasSet = MetaRule(lambda x: x is not None, "hasset")


class _ValuedMetaRuleBase(MetaRule):
    @classmethod
    def Cast(cls, value):
        # type: (Tensor) -> object
        raise NotImplementedError

    @classmethod
    def EQ(cls, value):
        # type: (object) -> MetaRule
        return MetaRule(lambda x: cls.Cast(x) == value,
                        "== {}".format(repr(value)))

    @classmethod
    def IN(cls, value):
        # type: (Union[List, Tuple, Set]) -> MetaRule
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("param 1 must be [List, Tuple, Set], got ".format(type(value)))
        return MetaRule(lambda x: cls.Cast(x) in value,
                        "in {}".format(repr(list(value))))

    @classmethod
    def GT(cls, value):
        # type: (object) -> MetaRule
        return MetaRule(lambda x: cls.Cast(x) > value,
                        "> {}".format(repr(value)))

    @classmethod
    def LT(cls, value):
        # type: (object) -> MetaRule
        return MetaRule(lambda x: cls.Cast(x) < value,
                        "< {}".format(repr(value)))

    @classmethod
    def NE(cls, value):
        # type: (object) -> MetaRule
        return MetaRule(lambda x: cls.Cast(x) != value,
                        "!= {}".format(repr(value)))

    @classmethod
    def GE(cls, value):
        # type: (object) -> MetaRule
        return MetaRule(lambda x: cls.Cast(x) >= value,
                        ">= {}".format(repr(value)))

    @classmethod
    def LE(cls, value):
        # type: (object) -> MetaRule
        return MetaRule(lambda x: cls.Cast(x) <= value,
                        "<= {}".format(repr(value)))


class String(_ValuedMetaRuleBase):
    @classmethod
    def Cast(cls, value):
        # type: (Tensor) -> str
        if not isinstance(value, (basestring, ts.tensor.StringTensor)):
            raise MetaValueMismatch(type(value), "[basestring, ts.tensor.StringTensor]")
        return str(value)


class Scalar(_ValuedMetaRuleBase):
    @classmethod
    def Cast(cls, value):
        # type: (Tensor) -> int
        if not isinstance(value, (int, float, numpy.ndarray, basestring, ts.tensor.StringTensor)):
            raise MetaValueMismatch(type(value),
                                    "[int, float, numpy.ndarray, basestring, ts.tensor.StringTensor]")
        if isinstance(value, (basestring, ts.tensor.StringTensor)):
            value = str(value)
        elif isinstance(value, numpy.ndarray):
            if value.shape != () and value.shape != (1,):
                raise MetaValueMismatch(type(value), "[int, float, numpy.ndarray]", "tensor.shape")
            value = value.reshape([1])[0]
        return value


class Ignore(object):
    def __eq__(self, other): return True
    def __ne__(self, other): return True
    def __gt__(self, other): return True
    def __lt__(self, other): return True
    def __ge__(self, other): return True
    def __le__(self, other): return True
    def __repr__(self): return "_"
    def __str__(self): return "_"


class _ListedMetaRule(MetaRule):
    @classmethod
    def Cast(cls, value):
        # type: (Tensor) -> List
        if not isinstance(value, (int, float, list, tuple, numpy.ndarray, basestring, ts.tensor.StringTensor)):
            raise MetaValueMismatch(type(value),
                                    "[int, float, list, tuple, numpy.ndarray, basestring, ts.tensor.StringTensor]")
        if isinstance(value, (basestring, ts.tensor.StringTensor)):
            value = list(str(value))
        elif isinstance(value, numpy.ndarray):
            value = list(value.reshape([-1]))
        elif isinstance(value, (list, tuple)):
            value = list(value)
        else:
            value = [value,]
        return value

    def __init__(self, value, cmp, cmp_str="with"):
        # type: (List[Union[int, float]], CallableMeta, str) -> None

        if not isinstance(value, (list, tuple)):
            raise MetaValueMismatch(type(value), "[list, tuple]", "param 1")

        value = [Ignore() if v is None else v for v in value]

        # Rule of control check list
        def rule(x):
            x = self.Cast(x)    # cast tensor to list
            if len(x) != len(value):
                return False
            n = len(x)
            for i in range(n):
                lhs = x[i]
                rhs = value[i]
                if isinstance(rhs, Ignore):
                    continue
                if not cmp(lhs, rhs):
                    return False
            return True

        super(_ListedMetaRule, self).__init__(rule,
                                              "{} {}".format(cmp_str, value))


class Array(MetaRule):
    @classmethod
    def EQ(cls, value):
        # type: (Union[List, Tuple]) -> MetaRule
        return _ListedMetaRule(value, lambda x, y: x == y, "==")

    @classmethod
    def GT(cls, value):
        # type: (Union[List, Tuple]) -> MetaRule
        return _ListedMetaRule(value, lambda x, y: x > y, ">")

    @classmethod
    def LT(cls, value):
        # type: (Union[List, Tuple]) -> MetaRule
        return _ListedMetaRule(value, lambda x, y: x < y, "<")

    @classmethod
    def NE(cls, value):
        # type: (Union[List, Tuple]) -> MetaRule
        return _ListedMetaRule(value, lambda x, y: x != y, "!=")

    @classmethod
    def GE(cls, value):
        # type: (Union[List, Tuple]) -> MetaRule
        return _ListedMetaRule(value, lambda x, y: x >= y, ">=")

    @classmethod
    def LE(cls, value):
        # type: (Union[List, Tuple]) -> MetaRule
        return _ListedMetaRule(value, lambda x, y: x <= y, "<=")


def EQ(value):
    # type: (Union[str, int, float, list, tuple]) -> MetaRule
    return Array.EQ(value) if isinstance(value, (list, tuple)) else Scalar.EQ(value)


def NE(value):
    # type: (Union[str, int, float, list, tuple]) -> MetaRule
    return Array.NE(value) if isinstance(value, (list, tuple)) else Scalar.NE(value)


def GT(value):
    # type: (Union[str, int, float, list, tuple]) -> MetaRule
    return Array.GT(value) if isinstance(value, (list, tuple)) else Scalar.GT(value)


def LT(value):
    # type: (Union[str, int, float, list, tuple]) -> MetaRule
    return Array.LT(value) if isinstance(value, (list, tuple)) else Scalar.LT(value)


def GE(value):
    # type: (Union[str, int, float, list, tuple]) -> MetaRule
    return Array.GE(value) if isinstance(value, (list, tuple)) else Scalar.GE(value)


def LE(value):
    # type: (Union[str, int, float, list, tuple]) -> MetaRule
    return Array.LE(value) if isinstance(value, (list, tuple)) else Scalar.LE(value)


def IN(value):
    # type: (Union[list, tuple]) -> MetaRule
    return Scalar.IN(value)


class MetaAttr(object):
    def __init__(self, rule):
        # type: (Union[str, int, float, list, tuple, MetaRule, MetaAttr]) -> None
        """
        :param rule:
        Example:
        1. MetaAttr(12) equals MetaAttr(Scalar.EQ(12))
        2. MetaAttr([1, 2, None]) equals MetaAttr(Array.EQ([1, 2, None]))
        3. MetaAttr(Scalar.EQ(12)) check value by meta rule
        """
        if isinstance(rule, (basestring, int, float)):
            self.__rule = Scalar.EQ(rule)
        elif isinstance(rule, (list, tuple)):
            self.__rule = Array.EQ(rule)
        elif isinstance(rule, MetaRule):
            self.__rule = rule
        elif isinstance(rule, MetaAttr):
            self.__rule = rule.__rule
        else:
            raise ValueError("param 1 got {}, but [str, int, float, list, tuple, MetaRule, MetaAttr] expected.")

    def check(self, value):
        # type: (Union[Tensor, None]) -> bool
        return self.__rule.__call__(value=value)

    def __call__(self, value):
        # type: (Union[Tensor, None]) -> bool
        return self.check(value)

    def __repr__(self):
        msg = repr(self.__rule)
        if len(msg) > 1 and msg[0] == '(' and msg[-1] == ')':
            msg = msg[1:-1]
        return msg

    def __str__(self):
        msg = str(self.__rule)
        if len(msg) > 1 and msg[0] == '(' and msg[-1] == ')':
            msg = msg[1:-1]
        return msg


class MetaNode(object):
    def __init__(self, attrs=None, **kwargs):
        # type: (Union[str, dict, MetaNode], dict[str, object]) -> None
        """
        :param attrs: str for node op, or dict for node attrs
        :param kwargs: node attrs, [Optional]
        Example:
        1. MetaNode("conv2d") equals to
            MetaNode({"#op": "conv2d"})
        2. MetaNode("conv2d", stride=[1, 1, 1, 1]) equals to
            MetaNode({"#op": "conv2d", "stride": [1, 1, 1, 1]})
        Notice:
            The dict's value must can be passed to MetaAttr
        """
        if attrs is None:
            self.__meta_attr = {}
            self.__input_count = None
            return

        if isinstance(attrs, MetaNode):
            self.__meta_attr = attrs.__meta_attr
            self.__input_count = attrs.__input_count
            return

        if isinstance(attrs, basestring):
            attrs = {"#op": str(attrs)}
        elif isinstance(attrs, dict):
            pass
        else:
            raise ValueError("param 1 must be [str, dict], got {}".format(type(attrs)))

        attrs.update(kwargs)

        self.__meta_attr = dict()

        for k, v in attrs.items():
            self.__meta_attr[k] = MetaAttr(v)

        self.__input_count = None

    def __repr__(self):
        items = []
        for k, v in self.__meta_attr.items():
            items.append("{} {}".format(k, v))
        return "{{{}}}".format(", ".join(items))

    def __str__(self):
        return self.__repr__()

    def check(self, node):
        # type: (Union[dict, ts.Node]) -> bool
        """
        Return if params satisfied meta node
        :param node:
        :return:
        """
        params = node
        input_count = 0
        if isinstance(node, ts.Node):
            input_count = len(node.inputs)
            params = node.params
        if not isinstance(params, dict):
            raise ValueError("param 1 must be dict or ts.Node")

        if self.__input_count is not None and input_count != self.__input_count:
            return False

        for k, v in self.__meta_attr.items():
            assert isinstance(v, MetaAttr)
            if k in params:
                checked = v.check(params[k])
            else:
                checked = v.check(None)
            if not checked:
                return False

        return True

    def __call__(self, node):
        # type: (Union[dict, ts.Node]) -> bool
        return self.check(node)

    def input(self, count):
        # type: (Union[int, SupportsInt]) -> MetaNode
        """
        Set node input count check
        :param count:
        :return:
        """
        self.__input_count = int(count)
        return self


class ABS(object):
    def __init__(self, i):
        self.i = i

    def __int__(self):
        return self.i

    def __repr__(self):
        return "ABS({})".format(self.i)

    def __str__(self):
        return "ABS({})".format(self.i)


class LinkedMetaNode(MetaNode):
    def __init__(self, attrs=None, **kwargs):
        # type: (Union[str, dict, MetaNode], dict[str, object]) -> None
        """
        :param attrs: str for node op, or dict for node attrs
        :param kwargs: node attrs, [Optional]
        Example:
        1. MetaNode("conv2d") equals to
            MetaNode({"#op": "conv2d"})
        2. MetaNode("conv2d", stride=[1, 1, 1, 1]) equals to
            MetaNode({"#op": "conv2d", "stride": [1, 1, 1, 1]})
        Notice:
            The dict's value must can be passed to MetaAttr
        """
        super(LinkedMetaNode, self).__init__(attrs=attrs, **kwargs)
        self.__inputs = []

    def link(self, inputs):
        # type: (List[Union[MetaNode, LinkedMetaNode]]) -> LinkedMetaNode
        if not isinstance(inputs, (tuple, list)):
            raise ValueError("param 1 must be list of LinkedMetaNode")
        fixed_inputs = []
        for i in inputs:
            if isinstance(i, LinkedMetaNode):
                fixed_inputs.append(i)
            elif isinstance(i, MetaNode):
                fixed_inputs.append(LinkedMetaNode(i))
            else:
                raise ValueError("param 1 must be list of LinkedMetaNode")
        inputs = fixed_inputs
        self.__inputs = list(inputs)

    @property
    def inputs(self):
        # type: () -> List[LinkedMetaNode]
        return self.__inputs

    def check(self, node, cache=None):
        # type: (Union[ts.Node], dict) -> bool
        """
        Return if params satisfied meta node
        :param node:
        :param cache:
        :return:
        """
        if not isinstance(node, ts.Node):
            raise ValueError("param 1 must be ts.Node")
        if cache is None:
            cache = {}
        if node in cache:
            checker = cache[node]
            if self != checker:
                return False
        else:
            cache[node] = self
        if not super(LinkedMetaNode, self).check(node):
            return False
        for i, v in enumerate(self.inputs):
            if i >= len(node.inputs):
                return False
            if not v.check(node.inputs[i]):
                return False
        return True


def _dict2list(obj):
    # type: (Dict) -> Union[List, None]
    arr = []
    for k, v in obj.items():
        if k < 0 or k > 65536:
            return None
        while len(arr) < k + 1:
            arr.append(None)
        arr[k] = v
    return arr


class MetaGraph(object):
    """
    Define meta graph, including nodes and their's links
    """
    NodeType = List[Union[str, MetaNode, Tuple[MetaNode, Union[List[int], Dict[int, int]]]]]

    def __init__(self, nodes, index=-1):
        # type: (Union[str, NodeType], int) -> None
        """
        Build meta graph, by node with links
        :param nodes: list of int, MetaNode or Tuple[MetaNode, List[Int]]
        :param index: the output flag
        Notice: the link index is related index, using ABS(index) for absolute index
        """
        # origin_nodes = nodes
        e = ValueError("param 1 must be str, or list of: str, dict, MetaNsode"
                       ", or list of (MetaNode, list[int] or dict[int, int])")
        """
        init attrs
        """
        self.__index = index
        self.__graph = []
        self.__links = []
        self.__anchor = None    # equals self.__graph[self.__index] later
        """
        Start check value, and update parameters
        """
        if isinstance(nodes, (basestring, dict)):
            nodes = [nodes, ]
        if not isinstance(nodes, (tuple, list)):
            raise e
        nodes = [MetaNode(i) if isinstance(i, (basestring, dict)) else i for i in nodes]
        nodes = [(i, None) if isinstance(i, MetaNode) else i for i in nodes]
        for node in nodes:
            if not isinstance(node, (tuple, list)):
                raise e
            if len(node) == 0:
                raise e
        nodes = [(i, None) if len(i) < 2 else i for i in nodes]
        nodes = [(MetaNode(i[0]), i[1]) if isinstance(i, (basestring, dict)) else i for i in nodes]
        nodes = [(MetaNode(i[0]), i[1]) if isinstance(i[0], (basestring, dict)) else i for i in nodes]
        for node in nodes:
            if not isinstance(node[0], MetaNode):
                raise e
            link = node[1]
            if isinstance(link, (int, ABS, type(None))):
                pass
            elif isinstance(link, (list, tuple)):
                check_link_type = [isinstance(i, (int, ABS, type(None))) for i in link]
                if not all(check_link_type):
                    raise e
            elif isinstance(link, dict):
                check_link_type = [isinstance(i[0], int) and isinstance(i[1], (int, ABS, type(None)))
                                   for i in link.items()]
                if not all(check_link_type):
                    raise e
            else:
                raise e
        self.__links = [i[1] for i in nodes]
        """
        Start build graph.
        """
        linked_nodes = [LinkedMetaNode(i[0]) for i in nodes]
        N = len(nodes)
        for i in range(N):
            node = linked_nodes[i]
            link = nodes[i][1]
            if link is None:
                continue
            if isinstance(link, (int, ABS)):
                link = [link, ]
            if isinstance(link, dict):
                link = _dict2list(link)
                if link is None:
                    raise e
            inputs = []
            for index in link:
                if index is None:
                    inputs.append(LinkedMetaNode())
                    continue
                if isinstance(index, ABS):
                    offset = index.i
                elif isinstance(index, int):
                    offset = i + index
                else:
                    raise e

                if offset >= N or offset < 0:
                    raise IndexError("index out of range: N={}, i={}, link={}".format(N, i, index))

                inputs.append(linked_nodes[offset])

            node.link(inputs=inputs)

        if N == 0:
            raise e

        self.__graph = linked_nodes
        self.__anchor = self.__graph[self.__index]

    def check(self, node):
        # type: (ts.Node) -> bool
        return self.__anchor.check(node)

    def __call__(self, node):
        # type: (ts.Node) -> bool
        return self.check(node)

    def __repr__(self):
        # type: () -> str
        N = len(self.__graph)

        lines = []
        for i in range(N):
            node = self.__graph[i]
            link = self.__links[i]
            lines.append("{}, {}".format(node, link))

        return "[{}]".format("\n ".join(lines))

    def __str__(self):
        # type: () -> str
        return self.__repr__()


if __name__ == "__main__":
    def print_rule(rule, value):
        if isinstance(value, numpy.ndarray):
            print("{}: {} {}".format(rule(value), value, rule))
        else:
            print("{}: {} {}".format(rule(value), repr(value), rule))

    a = MetaAttr(EQ(11) | EQ(12))
    print_rule(a, 11)
    print_rule(a, 12)
    print_rule(a, 13)
    print_rule(a, numpy.asarray(12))
    print_rule(a, numpy.asarray([12]))
    b = HasShape((1, 2, 3))
    print_rule(b, numpy.zeros((1, 2, 3)))
    c = HasShape()
    print_rule(c, 1)
    e = EQ("123")
    print_rule(e, 123)
    print_rule(e, "123")
    f = IN(["123"])
    print_rule(f, "123")
    f1 = GT([1, 2, None])
    f2 = HasShape([4])
    f3 = MetaAttr(f1 & f2)
    t = [2, 3, 4]
    print_rule(f1, t)
    print_rule(f2, t)
    print_rule(f3, t)

    m = MetaNode("conv2d", stride=[None, None, 3, 3], format="HCHW").input(3)

    checked = m.check({"#op": "conv2d", "stride": [1, 1, 3, 3], "format": "HCHW", "dilation": [1, 1, 1, 1]})

    print (checked)

    print(m)

    g = MetaGraph([
        {"#op": IN(["conv2d", "conv2d_v2"])},
        ({"#op": "add_bias"}, -1)
    ])

    print(g)

