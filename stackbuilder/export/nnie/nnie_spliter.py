from stackfence.spliter import GraphSpliter
from stackfence.metanode import *

from typing import Optional


def _get_shape(node):
    # type: (ts.Node) -> Optional[List[int]]
    if node.has("#shape"):
        return node.shape
    if node.op == ts.Node.Const:
        return node.get("value").shape
    return None


def if_no_broadcast_reduce(op):
    # type: (str) -> CallableMeta
    def checker(node):
        # type: (ts.Node) -> bool
        if node.op != op:
            return False
        lhs = node.inputs[0]
        rhs = node.inputs[1]
        lhs_shape = _get_shape(lhs)
        rhs_shape = _get_shape(rhs)
        if lhs_shape is None or rhs_shape is None:
            return False
        if len(lhs_shape) != len(rhs_shape):
            return False
        for i, j in zip(lhs_shape, rhs_shape):
            if i != j:
                return False
        return True
    return checker


def get_spliter():
    # type: () -> GraphSpliter
    gs = GraphSpliter(only_max_graph_out=True, single_input=False, single_output=False,
                      log_end_nodes=True)
    gs.route(ts.Node.Const)
    gs.support(MetaGraph([
        ts.Node.Const,
        ("conv2d", {1: -1})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ("add_bias", {1: -1})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ("depthwise_conv2d", {1: -1})
    ]))
    gs.support("pooling2d")
    gs.support(MetaGraph([
        ts.Node.Const,
        ts.Node.Const,
        ts.Node.Const,
        ts.Node.Const,
        ("fused_batch_norm", {1: -1, 2: -2, 3: -3, 4: -4})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ts.Node.Const,
        ("batch_norm", {1: -1, 2: -2})
    ]))
    gs.support(MetaGraph([
        ts.Node.Const,
        ts.Node.Const,
        ("batch_scale", {1: -1, 2: -2})
    ]))
    gs.support(if_no_broadcast_reduce("add"))
    gs.support("relu")
    # gs.route("flatten")
    gs.support("inner_prod")
    gs.route(MetaNode("_reshape", shape=HasShape(4)))
    gs.support(MetaGraph([
        {"#op": "concat",
         "dim": GE(-3) & LE(3) & NE(0),
         "#shape": HasShape(4)}
    ]))
    gs.support(if_no_broadcast_reduce("sub"))
    gs.support(MetaGraph([
        {"#op": ts.Node.Const, "value": EQ(0)},
        ({"#op": "sub", "#shape": HasShape(4)}, {0: -1})
    ]))
    gs.support(MetaNode({
        "#op": "_transpose",
        "permute": EQ([0, 2, 3, 1])
    }))
    gs.support(lambda x: x.op[:6] == "caffe:")
    gs.support(MetaNode({
        "#op": "softmax",
        "dim": GE(-3) & LE(3) & NE(0),
        "#shape": HasShape(4)
    }))
    gs.support(MetaNode({
        "#op": "relu_max",
        "max": 6,
        "#shape": GT([None, 0, 0, 0])
    }))
    gs.route("_copy")
    return gs