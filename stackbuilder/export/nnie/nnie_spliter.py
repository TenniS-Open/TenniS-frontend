from stackfence.spliter import GraphSpliter
from stackfence.metanode import *


def get_spliter():
    # type: () -> GraphSpliter
    gs = GraphSpliter(only_max_graph_out=True, single_input=True, single_output=True)
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
    return gs