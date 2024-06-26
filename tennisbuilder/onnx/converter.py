import onnx
from onnx import numpy_helper

import tennis as ts
from . import onnx_dtype as dtype
import tennis.frontend.onnx as onnx_node

import numpy


VoidNode = ts.menu.data('', numpy.asarray(0, dtype=numpy.float32))


def to_tensor_shape(tensor_shape):
    shape = []
    for dim in tensor_shape.dim:
        v = dim.dim_value if dim.HasField('dim_value') else -1
        shape.append(v)
    return shape


def get_tensor_stack_passes():
    return [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_dropout",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
        # "extract_constant_to_initializer",
        "fuse_add_bias_into_conv",
        # "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        # "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_transpose_into_gemm",
        "lift_lexical_references",
        "nop",
        # "split_init",
        # "split_predict",
    ]


class Name(object):
    class Attr(object):
        group = "group"
        auto_pad = "auto_pad"
        dilations = "dilations"
        kernel_shape = "kernel_shape"
        pads = "pads"
        strides = "strides"
        storage_order = "storage_order"

        axis = "axis"
        axes = "axes"

        alpha = "alpha"
        beta = "beta"
        transA = "transA"
        transB = "transB"
        epsilon = "epsilon"

        mode = "mode"
        value = "value"

        count_include_pad = "count_include_pad"
        ceil_mode = "ceil_mode"

        output_padding = "output_padding"
        output_shape = "output_shape"

        max = "max"
        min = "min"

        axes = "axes"
        keepdims = "keepdims"

    NOTSET = "NOTSET"
    SAME_UPPER = "SAME_UPPER"
    SAME_LOWER = "SAME_LOWER"
    VALID = "VALID"

    constant = "constant"
    reflect = "reflect"
    edge = "edge"


layer2converter = {
}


def register_layer_converter(layer, converter):
    layer2converter[layer] = converter


layer2version2converter = {}


def register_layer_version_converter(layer, version, converter):
    if layer in layer2version2converter:
        layer2version2converter[layer][version] = converter
    else:
        layer2version2converter[layer] = {version: converter}


def query_version_converter(layer, version):
    if layer not in layer2version2converter:
        return layer2converter[layer] if layer in layer2converter else None

    version2converter = layer2version2converter[layer]
    assert isinstance(version2converter, dict)
    found_v = 0
    found_converter = None
    for v in version2converter.keys():
        if found_v < v <= version:
            found_v = v
            found_converter = version2converter[v]

    if found_converter is None:
        return layer2converter[layer] if layer in layer2converter else None

    return found_converter


def unique_names(onnx_model, export_model=None):
    # type: (Union[str, onnx.ModelProto], Opitonal[str]) -> onnx.ModelProto
    if isinstance(onnx_model, onnx.ModelProto):
        pass
    elif isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)
    else:
        raise ValueError("onnx_model must be str or onnx.ModelProto")

    import copy
    onnx_model = copy.deepcopy(onnx_model)
    cache = {}  # map original name to new Name
    initializer_count = 0
    node_count = 0

    def get_initializer(name):
        if not name:
            return ""
        nonlocal initializer_count
        if name in cache:
            return cache[name]
        initializer_count += 1
        new_name = "Initializer_{}".format(initializer_count)
        cache[name] = new_name
        return new_name

    def get_input(name):
        if not name:
            return ""
        nonlocal node_count
        if name in cache:
            return cache[name]
        node_count += 1
        new_name = "Input_{}".format(node_count)
        cache[name] = new_name
        return new_name

    def get_output(name):
        if not name:
            return ""
        assert name in cache
        return cache[name]

    def get_node(name, op_type):
        # type: (str, str) -> str
        if not name:
            return ""
        nonlocal node_count
        if name in cache:
            return cache[name]
        node_count += 1
        new_name = "{}_{}".format(op_type, node_count)
        cache[name] = new_name
        return new_name

    onnx_graph = onnx_model.graph
    for tensor in onnx_graph.initializer:
        tensor.name = get_initializer(tensor.name)

    for value_info in onnx_graph.input:
        value_info.name = get_input(value_info.name)

    for node in onnx_graph.node:
        op_type = node.op_type
        name = node.name
        node_input = list(node.input)
        while len(node.input):
            node.input.pop()
        node.input.extend([get_node(s, op_type) for s in node_input])
        node_output = list(node.output)
        while len(node.output):
            node.output.pop()
        node.output.extend([get_node(s, op_type) for s in node_output])

        node.name = "{}_{}".format(op_type, "_".join([s.split("_")[-1] for s in node.output]))

    for value_info in onnx_graph.output:
        value_info.name = get_output(value_info.name)

    for value_info in onnx_graph.value_info:
        value_info.name = get_node(value_info.name, "Value")

    onnx.checker.check_model(onnx_model)
    if export_model is not None:
        assert isinstance(export_model, str)
        onnx.save(onnx_model, export_model)

    return onnx_model


def convert(input_file, output_file, check_graph=False, specific=None):
    """
    convert onnx
    :param input_file: onnx.ModelProto or param can parse into onnx.load(param)
    :param output_file: str of path to file
    :param check_graph: if call onnx.checker.check_graph
    :param specific: dict of converter
    :return: ts.Module
    """
    onnx_model = None
    if isinstance(input_file, onnx.ModelProto):
        onnx_model = input_file
    else:
        onnx_model = onnx.load(input_file)

    if onnx_model is None:
        raise Exception("Can not load {}:{} to onnx model".format(type(input_file), input_file))

    if check_graph:
        onnx.checker.check_graph(onnx_model.graph)
    else:
        try:
            onnx.checker.check_graph(onnx_model.graph)
        except Exception as e:
            import sys
            sys.stderr.write("[WARNING]: Check graph failed with: {}\n".format(e))

    opset_domain = "ai.onnx"
    opset_version = 0
    for opset in onnx_model.opset_import:
        this_domain = opset.domain if opset.HasField("domain") else "ai.onnx"
        this_version = opset.version if opset.HasField("version") else None

        if this_version is None:
            continue

        if opset_version is None or this_version > opset_version:
            opset_domain = this_domain
            opset_version = this_version

    if len(opset_domain) == 0:
        opset_domain = "ai.onnx"

    ir_version = onnx_model.ir_version if onnx_model.HasField("ir_version") else 0

    producer_name = onnx_model.producer_name if onnx_model.HasField("producer_name") else "unknown"
    producer_version = onnx_model.producer_version if onnx_model.HasField("producer_version") else 0

    print("[INFO] format: ONNX v{}".format(ir_version))
    print("[INFO] producer: {} {}".format(producer_name, producer_version))
    print("[INFO] imports: {} v{}".format(opset_domain, opset_version))

    onnx_graph = onnx_model.graph

    # op
    nodes = []
    print("==================== Node ====================")
    for node in onnx_graph.node:
        op_type = node.op_type
        attribute = node.attribute
        # print("{}: {} => {}".format(node.op_type, list(node.input), list(node.output)))
        # print("{}".format(attribute))
        nodes.append(node)
    print ("Got {} nodes.".format(len(nodes)))

    # init
    initialized = {}    # str: numpy.array
    print("==================== Initializer ====================")
    for tensor in onnx_graph.initializer:
        name = tensor.name
        array = numpy_helper.to_array(tensor)
        # print("{}: {}, {}".format(name, array.dtype, array.shape))
        initialized[name] = array
    print ("Got {} initializer.".format(len(initialized)))

    input = {}  # str, shape
    graph_input_names = []
    # input
    print("==================== Input ====================")
    for value_info in onnx_graph.input:
        name = value_info.name
        if name in initialized:
            continue
        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type
        shape = to_tensor_shape(tensor_type.shape)
        print("{}: {}, {}".format(name, elem_type, shape))
        input[name] = (elem_type, shape)
        graph_input_names.append(name)

    output = {} # str, shape
    graph_output_names = []
    # output
    print("==================== Output ====================")
    for value_info in onnx_graph.output:
        name = value_info.name
        if name in initialized:
            continue
        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type
        shape = to_tensor_shape(tensor_type.shape)
        print("{}: {}, {}".format(name, elem_type, shape))
        output[name] = (elem_type, shape)
        graph_output_names.append(name)

    # set all initialized node
    name2node = {}  # str -> ts.Node
    name2node[''] = VoidNode    # support void node to support optional input
    # get ts_inputs
    ts_inputs = []
    # no loop in graph
    for name in graph_input_names:
        value = input[name]
        elem_type = value[0]
        shape = value[1]
        ts_dtype = dtype.from_onnx(elem_type)
        ts_input_node = ts.menu.param("_input_" + name, shape=shape)
        ts_node = ts.zoo.cast(name, x=ts_input_node, dtype=ts_dtype)
        name2node[name] = ts_node
        ts_inputs.append(ts_input_node)

    for name in initialized.keys():
        value = initialized[name]
        ts_node = ts.menu.data(name, value=value)
        name2node[name] = ts_node

    builtin_layer_converters = {
        "Conv": convert_conv_layer,
        "Relu": convert_relu_layer,
        "MaxPool": convert_pooling2d_layer,
        "Add": convert_add_layer,
        "AveragePool": convert_pooling2d_layer,
        "Shape": convert_shape_layer,
        "Concat": convert_concat_layer,
        "BatchNormalization": convert_bn_layer,
        "Pad": convert_pad_layer,
        "Constant": convert_constant_layer,
        # about new operator
        "Gather": convert_gather_layer,
        "Unsqueeze": convert_unsqueeze_layer,
        "Reshape": convert_reshape_layer,
        "Gemm": convert_gemm_layer,
        "GlobalAveragePool": convert_global_pooling2d_layer,
        "Sigmoid": convert_sigmoid_layer,
        "Neg": convert_neg_layer,
        "Transpose": convert_transpose_layer,
        # "Softmax": convert_softmax_layer,
    }

    if specific is not None:
        if not isinstance(specific, dict):
            raise Exception("specific must be dict, got {}".format(type(specific)))
    else:
        specific = {}

    print("==================== Converting ====================")
    # convert each node
    for node in nodes:
        op_type = node.op_type
        # attribute = node.attribute
        node_input = node.input
        node_output = node.output

        ts_converter = None

        # check specific
        ts_converter = specific[op_type] if op_type in specific else None

        # check outer register
        if ts_converter is None:
            ts_converter = query_version_converter(op_type, opset_version)

        #  check built-in converter
        if ts_converter is None:
            ts_converter = builtin_layer_converters[op_type] if op_type in builtin_layer_converters else None

        # convert layer
        if ts_converter is None:
            raise Exception("Not supported ONNX Layer {}-{}".format(op_type, opset_version))

        input_ts_nodes = []
        for name in node_input:
            input_ts_nodes.append(name2node[name])

        output_names = []
        for name in node_output:
            output_names.append(name)

        output_ts_nodes = ts_converter(node, input_ts_nodes, output_names)

        if isinstance(output_ts_nodes, ts.Node):
            output_ts_nodes = (output_ts_nodes, )

        assert len(output_names) == len(output_ts_nodes)

        for i in range(len(output_ts_nodes)):
            # update blob2nodes
            name2node[node_output[i]] = output_ts_nodes[i]

    # get outputs from outout_blobs
    ts_outputs = []
    for name in graph_output_names:
        if name not in name2node:
            raise Exception("Not computed node: {}".format(name))
        ts_outputs.append(name2node[name])

    module = ts.Module()

    # load module
    module.load(ts_outputs)

    # sort inputs
    module.sort_inputs(ts_inputs)

    with open(output_file, "wb") as fo:
        ts.Module.Save(stream=fo, module=module)

    print("============ Summary ============")
    print("Input file: {}".format(input_file))
    print("Output file: {}".format(output_file))
    index = 0
    print("Input node: ")
    for node in module.inputs:
        assert isinstance(node, ts.Node)
        print("{}: {}, shape={}".format(index, node.name, node.shape))
        index += 1
    index = 0
    print("Output node: ")
    for node in module.outputs:
        assert isinstance(node, ts.Node)
        print("{}: {}".format(index, node.name))
        index += 1

    return module


def topy(attr):
    # type: (onnx.AttributeProto) -> object
    type = attr.type
    if type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    elif type == onnx.AttributeProto.STRING:
        return bytes(attr.s).decode("UTF-8")
    elif type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif type == onnx.AttributeProto.INT:
        return attr.i
    else:
        raise Exception("Can not convert attribute: {}".format(attr))


def convert_conv_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2 or len(input_nodes) == 3
    assert len(output_names) == 1

    conv2d_name = "_conv2d_" + output_names[0]
    bias_name = "_bias_" + output_names[0]
    node_name = output_names[0]

    X = input_nodes[0]
    W = input_nodes[1]  # (M x C/group x kH x kW)
    B = None
    if len(input_nodes) > 2:
        B = input_nodes[2]

    auto_pad = Name.NOTSET
    if Name.Attr.auto_pad in attr_dict:
        auto_pad = attr_dict[Name.Attr.auto_pad]
        print("--##    AutoPad: {}".format(auto_pad))

    dilations = attr_dict[Name.Attr.dilations]
    print("--##    Dilations: {}".format(dilations))

    group = 1
    if Name.Attr.group in attr_dict:
        group = attr_dict[Name.Attr.group]
        print("--##    Group: {}".format(group))

    kernel_shape = attr_dict[Name.Attr.kernel_shape]
    print("--##    KernelShape: {}".format(kernel_shape))

    pads = attr_dict[Name.Attr.pads]
    print("--##    Pads: {}".format(pads))

    strides = attr_dict[Name.Attr.strides]
    print("--##    Strides: {}".format(strides))

    if auto_pad != Name.NOTSET:
        raise NotImplementedError("auto_pad = {}".format(auto_pad))

    if len(dilations) != 2:
        raise NotImplementedError("dilations = {}".format(dilations))

    if len(kernel_shape) != 2:
        raise NotImplementedError("kernel_shape = {}".format(kernel_shape))

    W_array = ts.zoo.to_const(W, "W")

    if len(W_array.shape) != 4:
        raise NotImplementedError("W.shape = {}".format(W_array.shape))

    if group != 1 and W_array.shape[1] != 1:
        raise NotImplementedError("group = {} with weights.shape[1] = {}".format(group, W_array.shape[1]))

    if kernel_shape[0] != W_array.shape[2] or kernel_shape[1] != W_array.shape[3]:
        raise NotImplementedError("kernel_shape = {} with W.shape = {}".format(kernel_shape, W_array.shape))

    if len(pads) != 4:
        raise NotImplementedError("pads = {}".format(pads))

    if len(strides) != 2:
        raise NotImplementedError("strides = {}".format(strides))

    is_conv2d = group == 1
    is_depthwise_conv2d = W_array.shape[1] == 1

    ts_node = None

    if is_conv2d:
        ts_node = ts.zoo.conv2d(conv2d_name, x=input_nodes[0], w=W, format=ts.zoo.Name.NCHW,
                                padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                padding_value=0,
                                stride=[1, 1, strides[0], strides[1]],
                                dilation=[1, 1, dilations[0], dilations[1]])
    elif is_depthwise_conv2d:
        weights_shape = W_array.shape
        depthwise_weights_shape = (weights_shape[1], weights_shape[0], weights_shape[2], weights_shape[3])
        weights_blob = W_array.reshape(depthwise_weights_shape)
        ts_node = ts.zoo.depthwise_conv2d(conv2d_name, x=input_nodes[0], w=weights_blob, format=ts.zoo.Name.NCHW,
                                          padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                          padding_value=0,
                                          stride=[1, 1, strides[0], strides[1]],
                                          dilation=[1, 1, dilations[0], dilations[1]])

    if ts_node is None:
        raise NotImplementedError(node)

    if B is not None:
        ts_node = ts.zoo.add_bias(bias_name, x=ts_node, b=B, format=ts.zoo.Name.NCHW)

    ts_node.name = node_name

    return ts_node,


def convert_relu_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.relu(node_name, x=x)

    return ts_node,


def convert_neg_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.sub(name=node_name, lhs=numpy.asarray(0, dtype=numpy.float32), rhs=x)

    return ts_node,


def convert_pooling2d_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    op_type = node.op_type
    onnx_op_type_to_ts_pool_type = {
        "MaxPool": ts.zoo.Type.pooling_type.max,
        "AveragePool": ts.zoo.Type.pooling_type.avg,
    }

    auto_pad = Name.NOTSET
    if Name.Attr.auto_pad in attr_dict:
        auto_pad = attr_dict[Name.Attr.auto_pad]
        print("--##    AutoPad: {}".format(auto_pad))

    kernel_shape = attr_dict[Name.Attr.kernel_shape]
    print("--##    KernelShape: {}".format(kernel_shape))

    pads = [0, 0, 0, 0]
    if Name.Attr.pads in attr_dict:
        pads = attr_dict[Name.Attr.pads]
    print("--##    Pads: {}".format(pads))

    storage_order = 0
    if Name.Attr.storage_order in attr_dict:
        storage_order = attr_dict[Name.Attr.storage_order]
        print("--##    StorageOrder: {}".format(storage_order))

    strides = attr_dict[Name.Attr.strides]
    print("--##    Strides: {}".format(strides))

    count_include_pad = False
    if Name.Attr.count_include_pad in attr_dict:
        count_include_pad = attr_dict[Name.Attr.count_include_pad] != 0
    ceil_mode = False
    if Name.Attr.ceil_mode in attr_dict:
        ceil_mode = attr_dict[Name.Attr.ceil_mode] != 0

    if auto_pad != Name.NOTSET:
        raise NotImplementedError("auto_pad = {}".format(auto_pad))

    if storage_order != 0:
        raise NotImplementedError("storage_order = {}".format(storage_order))

    if len(kernel_shape) != 2:
        raise NotImplementedError("kernel_shape = {}".format(kernel_shape))

    if len(pads) != 4:
        raise NotImplementedError("pads = {}".format(pads))

    if len(strides) != 2:
        raise NotImplementedError("strides = {}".format(strides))

    if op_type not in onnx_op_type_to_ts_pool_type:
        raise NotImplementedError("pooling type = {}".format(op_type))
    pool_type = onnx_op_type_to_ts_pool_type[op_type]

    ts_padding_type = ts.zoo.Type.padding_type.black
    if count_include_pad:
        ts_padding_type = ts.zoo.Type.padding_type.white

    ts_node = onnx_node.pooling2d(node_name, x=x,
                                  ksize=[1, 1, kernel_shape[0], kernel_shape[1]],
                                  stride=[1, 1, strides[0], strides[1]],
                                  type=pool_type,
                                  format=ts.zoo.Name.NCHW,
                                  padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                  padding_type=ts_padding_type,
                                  auto_pad=auto_pad,
                                  ceil_mode=ceil_mode)
    return ts_node,


def convert_add_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]

    ts_node = ts.zoo.add(node_name, lhs=x, rhs=y)

    return ts_node,


def convert_shape_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.shape("_int32_" + node_name, x=x)
    ts_node.set('#dtype', ts.dtype.INT32)
    ts_node = ts.zoo.cast(node_name, ts_node, dtype=ts.dtype.INT64)

    return ts_node,


def convert_gather_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    indices = input_nodes[1]

    axis = 0
    if Name.Attr.axis in attr_dict:
        axis = attr_dict[Name.Attr.axis]
        print("--##    axis: {}".format(axis))

    ts_node = onnx_node.gather(node_name, x=x, indices=indices, axis=axis)

    return ts_node,


def convert_unsqueeze_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axes = attr_dict[Name.Attr.axes]
    print("--##    axes: {}".format(axes))

    ts_node = onnx_node.unsqueeze(node_name, x=x, axes=axes)

    return ts_node,


def convert_concat_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(output_names) == 1

    node_name = output_names[0]
    print("--##    input number: {}".format(len(input_nodes)))

    axis = attr_dict[Name.Attr.axis]
    print("--##    axis: {}".format(axis))

    ts_node = ts.zoo.concat(node_name, inputs=input_nodes, dim=axis)

    return ts_node,


def __whose_flatten_shape(shape):
    # type: (ts.Node) -> Union[ts.Node, None]
    """
    :return: return flatten tensor if it's flatten shape like(x.number, -1)
    """
    if not isinstance(shape, ts.Node):
        return None

    if shape.op != ts.zoo.Name.Layer.concat:
        return None

    if len(shape.inputs) != 2:
        return None

    unsqueeze_x_number = shape.inputs[0]
    unsqueeze_neg_one = shape.inputs[1]

    if unsqueeze_x_number.op != onnx_node.Name.Layer.unsqueeze:
        return None

    if unsqueeze_neg_one.op != onnx_node.Name.Layer.unsqueeze:
        return None

    if list(unsqueeze_x_number.get(onnx_node.Name.axes)) != [0]:
        return None

    if list(unsqueeze_neg_one.get(onnx_node.Name.axes)) != [0]:
        return None

    neg_one = unsqueeze_neg_one.inputs[0]
    x_number = unsqueeze_x_number.inputs[0]

    if neg_one.op != ts.Node.Const:
        return None
    elif int(neg_one.get(ts.menu.Name.value)) != -1:
        return None

    if x_number.op != onnx_node.Name.Layer.gather:
        return None
    elif int(x_number.get(onnx_node.Name.axis)) != 0:
        return None

    x_shape = x_number.inputs[0]

    if x_shape.op == ts.zoo.Name.Layer.cast:
        x_shape = x_shape.inputs[0]

    if x_shape.op != ts.zoo.Name.Layer.shape:
        return None

    x = x_shape.inputs[0]

    return x


def convert_reshape_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    new_shape = input_nodes[1]

    flatten_x = __whose_flatten_shape(new_shape)

    if x == flatten_x:
        print("--##    IsFlatten: {}".format(True))
        return ts.zoo.flatten(node_name, x)

    ts_node = ts.zoo.reshape(node_name, x, new_shape)

    return ts_node,


def convert_gemm_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 3
    assert len(output_names) == 1

    node_name = output_names[0]

    A = input_nodes[0]
    B = input_nodes[1]
    C = input_nodes[2]

    alpha = 1.0
    if Name.Attr.alpha in attr_dict:
        alpha = attr_dict[Name.Attr.alpha]
    print("--##    alpha: {}".format(alpha))

    beta = 1.0
    if Name.Attr.beta in attr_dict:
        beta = attr_dict[Name.Attr.beta]
    print("--##    beta: {}".format(beta))

    transA = 0
    if Name.Attr.transA in attr_dict:
        transA = attr_dict[Name.Attr.transA]
    print("--##    transA: {}".format(transA))

    transB = 0
    if Name.Attr.transB in attr_dict:
        transB = attr_dict[Name.Attr.transB]
    print("--##    transB: {}".format(transB))

    ts_node = onnx_node.gemm(node_name, A=A, B=B, C=C, alpha=alpha, beta=beta, transA=transA, transB=transB)

    return ts_node,


def convert_bn_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 5
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    scale = input_nodes[1]
    B = input_nodes[2]
    mean = input_nodes[3]
    var = input_nodes[4]

    epsilon = 1e-5
    if Name.Attr.epsilon in attr_dict:
        epsilon = attr_dict[Name.Attr.epsilon]
        print("--##    epsilon: {}".format(epsilon))

    ts_node = ts.zoo.fused_batch_norm(node_name, x=x,
                                      mean=mean, variance=var, scale=scale, bias=B,
                                      dim=1, epsilon=epsilon)

    return ts_node,


def convert_pad_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    mode = Name.constant
    if Name.Attr.mode in attr_dict:
        mode = attr_dict[Name.Attr.mode]
        print("--##    mode: {}".format(mode))

    if mode != Name.constant:
        raise NotImplementedError("mode={}".format(mode))

    pads = attr_dict[Name.Attr.pads]
    print("--##    pads: {}".format(pads))

    value = 0
    if Name.Attr.value in attr_dict:
        value = attr_dict[Name.Attr.value]
        print("--##    value: {}".format(value))

    pads = numpy.asarray(pads, dtype=numpy.int32).reshape((2, -1)).T

    ts_node = ts.zoo.pad(node_name, x=x, padding=pads, padding_value=value)

    return ts_node,


def convert_constant_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 0
    assert len(output_names) == 1

    node_name = output_names[0]

    value = attr_dict[Name.Attr.value]

    ts_node = ts.menu.data(node_name, value=value)

    return ts_node,


def convert_global_pooling2d_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    op_type = node.op_type
    onnx_op_type_to_ts_pool_type = {
        "GlobalMaxPool": ts.zoo.Type.pooling_type.max,
        "GlobalAveragePool": ts.zoo.Type.pooling_type.avg,
    }

    if op_type not in onnx_op_type_to_ts_pool_type:
        raise NotImplementedError("pooling type = {}".format(op_type))
    pool_type = onnx_op_type_to_ts_pool_type[op_type]

    ts_node = ts.zoo.global_pooling2d(node_name, x=x, type=pool_type, format=ts.zoo.Name.NCHW)

    return ts_node,


def convert_sigmoid_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.sigmoid(node_name, x=x)

    return ts_node,


def convert_transpose_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.transpose(name=node_name, x=x, permute=attr_dict["perm"])

    return ts_node,


def convert_softmax_v1_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axis = 1
    if Name.Attr.axis in attr_dict:
        axis = int(attr_dict[Name.Attr.axis])

    x_shape = ts.zoo.shape(name=node_name + "_shape", x=x)

    flatten_x = ts.zoo.flatten(name=node_name + "_flatten", x=x, dim=axis)

    softmax_flatten_x = ts.zoo.softmax(name=node_name + "_softmax", x=flatten_x, dim=-1)

    y = ts.zoo.reshape_v2(name=node_name, x=softmax_flatten_x, shape=x_shape)

    return y,


register_layer_version_converter("Softmax", 11, convert_softmax_v1_layer)
register_layer_version_converter("Softmax", 1, convert_softmax_v1_layer)


def convert_softmax_v13_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axis = -1
    if Name.Attr.axis in attr_dict:
        axis = int(attr_dict[Name.Attr.axis])

    y = ts.zoo.softmax(name=node_name, x=x, dim=axis)

    return y,


register_layer_version_converter("Softmax", 13, convert_softmax_v13_layer)


def convert_sub_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]

    ts_node = ts.zoo.sub(node_name, lhs=x, rhs=y)

    return ts_node,


register_layer_converter("Sub", convert_sub_layer)


def convert_div_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]

    ts_node = ts.zoo.div(node_name, lhs=x, rhs=y)

    return ts_node,


register_layer_converter("Div", convert_div_layer)


def convert_mul_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]

    ts_node = ts.zoo.mul(node_name, lhs=x, rhs=y)

    return ts_node,


register_layer_converter("Mul", convert_mul_layer)


def convert_reduce_sum_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axes = attr_dict["axes"]
    keepdims = bool(attr_dict["keepdims"]) if "keepdims" in attr_dict else True

    if len(axes) == 0:
        node = ts.zoo.reshape(node_name + "_flatten", x=x, shape=[-1])
        ts_node = ts.zoo.reduce_sum(node_name, x=node, reduce_dims=0, keep_dims=keepdims)
    elif len(axes) == 1:
        ts_node = ts.zoo.reduce_sum(node_name, x=x, reduce_dims=axes[0], keep_dims=keepdims)
    else:
        raise NotImplementedError("axes = {}".format(axes))

    return ts_node,


register_layer_converter("ReduceSum", convert_reduce_sum_layer)


def convert_sqrt_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.sqrt(node_name, x)

    return ts_node,


register_layer_converter("Sqrt", convert_sqrt_layer)


def convert_flatten_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axis = 1
    if "axis" in attr_dict:
        axis = attr_dict["axis"]

    if axis == 1:
        ts_node = ts.zoo.flatten(node_name, x=x, dim=axis)
    else:
        ts_node = ts.zoo.flatten2d(node_name, x=x, dim=axis)

    return ts_node,


register_layer_converter("Flatten", convert_flatten_layer)


def convert_conv_traspose_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2 or len(input_nodes) == 3
    assert len(output_names) == 1

    conv2d_name = "_conv2d_" + output_names[0]
    bias_name = "_bias_" + output_names[0]
    node_name = output_names[0]

    X = input_nodes[0]
    W = input_nodes[1]  # (M x C/group x kH x kW)
    B = None
    if len(input_nodes) > 2:
        B = input_nodes[2]

    auto_pad = Name.NOTSET
    if Name.Attr.auto_pad in attr_dict:
        auto_pad = attr_dict[Name.Attr.auto_pad]
        print("--##    AutoPad: {}".format(auto_pad))

    dilations = attr_dict[Name.Attr.dilations]
    print("--##    Dilations: {}".format(dilations))

    group = 1
    if Name.Attr.group in attr_dict:
        group = attr_dict[Name.Attr.group]
        print("--##    Group: {}".format(group))

    kernel_shape = attr_dict[Name.Attr.kernel_shape]
    print("--##    KernelShape: {}".format(kernel_shape))

    pads = attr_dict[Name.Attr.pads]
    print("--##    Pads: {}".format(pads))

    strides = attr_dict[Name.Attr.strides]
    print("--##    Strides: {}".format(strides))

    output_padding = None
    if Name.Attr.output_padding in attr_dict:
        output_padding = attr_dict[Name.Attr.output_padding]
        print("--##    output_padding: {}".format(output_padding))

    output_shape = None
    if Name.Attr.output_shape in attr_dict:
        output_shape = attr_dict[Name.Attr.output_shape]
        print("--##    output_shape: {}".format(output_shape))

    if output_shape is not None:
        raise NotImplementedError("output_shape = {}".format(output_shape))

    if auto_pad != Name.NOTSET:
        raise NotImplementedError("auto_pad = {}".format(auto_pad))

    if group != 1:
        raise NotImplementedError("group = {}".format(group))

    if len(dilations) != 2:
        raise NotImplementedError("dilations = {}".format(dilations))

    if len(kernel_shape) != 2:
        raise NotImplementedError("kernel_shape = {}".format(kernel_shape))

    W_array = ts.zoo.to_const(W, "W")

    if len(W_array.shape) != 4:
        raise NotImplementedError("W.shape = {}".format(W_array.shape))

    if group != 1 and W_array.shape[1] != 1:
        raise NotImplementedError("group = {} with weights.shape[1] = {}".format(group, W_array.shape[1]))

    if kernel_shape[0] != W_array.shape[2] or kernel_shape[1] != W_array.shape[3]:
        raise NotImplementedError("kernel_shape = {} with W.shape = {}".format(kernel_shape, W_array.shape))

    if len(pads) != 4:
        raise NotImplementedError("pads = {}".format(pads))

    if len(strides) != 2:
        raise NotImplementedError("strides = {}".format(strides))

    is_conv2d = group == 1
    # is_depthwise_conv2d = W_array.shape[1] == 1

    ts_node = None

    if is_conv2d:
        ts_node = ts.zoo.transpose_conv2d(conv2d_name, x=input_nodes[0], w=W, format=ts.zoo.Name.NCHW,
                                          padding=[[0, 0], [0, 0], [pads[0], pads[2]], [pads[1], pads[3]]],
                                          padding_value=0,
                                          stride=[1, 1, strides[0], strides[1]],
                                          dilation=[1, 1, dilations[0], dilations[1]])

    if output_padding is not None:
        assert len(output_padding) == 4
        ts_node = ts.zoo.pad(node_name + "_out_pad", x=ts_node,
                             padding=[[0, 0], [0, 0], [output_padding[0], output_padding[2]], [output_padding[1], output_padding[3]]])

    if ts_node is None:
        raise NotImplementedError(node)

    if B is not None:
        ts_node = ts.zoo.add_bias(bias_name, x=ts_node, b=B, format=ts.zoo.Name.NCHW)

    ts_node.name = node_name

    return ts_node,


register_layer_converter("ConvTranspose", convert_conv_traspose_layer)


def convert_tile_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    repeats = input_nodes[1]

    ts_node = ts.zoo.tile(node_name, x=x, repeats=repeats)

    return ts_node,


register_layer_converter("Tile", convert_tile_layer)


def convert_dropout_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.copy(node_name, x=x)

    return ts_node,


register_layer_converter("Dropout", convert_dropout_layer)


def convert_tanh_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.tanh(node_name, x=x)

    return ts_node,


register_layer_converter("Tanh", convert_tanh_layer)


def convert_abs_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.abs(node_name, x=x)

    return ts_node,


register_layer_converter("Abs", convert_abs_layer)


def convert_upsample_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    scales = input_nodes[1]

    mode = attr_dict["mode"]
    mode2type = {
        "nearest": ts.zoo.Type.resize2d_type.hard,      # nearest means hard in TS
        "bilinear": ts.zoo.Type.resize2d_type.linear,
    }
    if mode not in mode2type:
        raise NotImplementedError("mode={}".format(mode))
    type = mode2type[mode]

    try:
        scales = ts.zoo.to_const(scales, "scales")
    except:
        # do common sample image
        x_shape = ts.zoo.shape(name=x.name + "_shape", x=x)
        float_x_shape = ts.zoo.cast(name=x.name + "_float_shape", x=x_shape, dtype=ts.dtype.FLOAT32)
        scaled_size = ts.zoo.mul(name=x.name + "_scale_size", lhs=float_x_shape, rhs=scales)
        int_scaled_size = ts.zoo.cast(name=x.name + "_int_size", x=scaled_size, dtype=dtype.ts.dtype.INT32)
        return ts.zoo.resize2d(name=node_name, x=x, size=int_scaled_size, type=type)

    # do static scale image
    scales = numpy.asarray(scales)

    if scales.shape != (4,):
        raise NotImplementedError("scales={}".format(scales))

    if scales[0] != 1 or scales[1] != 1:
        raise NotImplementedError("scales={}".format(scales))

    if scales[2] != scales[3]:
        raise NotImplementedError("scales={}".format(scales))

    scale = scales[3]

    ts_node = ts.zoo.sample2d(name=node_name, x=x, scale=scale, type=type)

    return ts_node,


register_layer_converter("Upsample", convert_upsample_layer)


def convert_reduce_mean_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    reduce_dims = attr_dict[Name.Attr.axes]
    print("--##    reduce_dims: {}".format(reduce_dims))
    keepdims = attr_dict[Name.Attr.keepdims]
    print("--##    keep_dims: {}".format(keepdims))

    keepdims = True if keepdims == 1 else False

    ts_node = ts.zoo.reduce_mean(node_name, x=x, reduce_dims=reduce_dims, keep_dims=keepdims)

    return ts_node


register_layer_converter("ReduceMean", convert_reduce_mean_layer)


def convert_matmul(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    A = input_nodes[0]
    B = input_nodes[1]

    ts_node = ts.zoo.matmul(name=node_name, lhs=A, rhs=B)

    return ts_node,


register_layer_converter("MatMul", convert_matmul)


def convert_expand(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    shape = input_nodes[1]

    ts_node = ts.zoo.broadcast(name=node_name, x=x, shape=shape)

    return ts_node,


register_layer_converter("Expand", convert_expand)


def convert_exp_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.exp(node_name, x=x)

    return ts_node,


register_layer_converter("Exp", convert_exp_layer)


def convert_slice_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)
    assert len(output_names) == 1
    assert len(input_nodes) >= 1

    node_name = output_names[0]
    x = input_nodes[0]

    starts = None
    ends = None
    axes = None
    steps = None

    if len(input_nodes) == 1:
        # is Slice-1
        starts = attr_dict["starts"]
        ends = attr_dict["ends"]
        starts = ts.zoo.to_node(starts, node_name + "_starts", device=ts.device.CPU, dtype=numpy.int64)
        ends = ts.zoo.to_node(ends, node_name + "_ends", device=ts.device.CPU, dtype=numpy.int64)
        if "axes" in attr_dict:
            axes = attr_dict["axes"]
            axes = ts.zoo.to_node(axes, node_name + "_axes", device=ts.device.CPU, dtype=numpy.int64)
    elif len(input_nodes) >= 3 and len(input_nodes) <= 5:
        # is Slice-10 or Slice-11
        starts = input_nodes[1]
        ends = input_nodes[2]
        if len(input_nodes) > 3:
            axes = input_nodes[3]
        if len(input_nodes) > 4:
            steps = input_nodes[4]
    else:
        raise NotImplementedError("{}".format(node))

    ts_node = None

    if axes is None:
        ts_node = ts.frontend.onnx.slice_v3(node_name, x, starts, ends)
    elif steps is None:
        ts_node = ts.frontend.onnx.slice_v3(node_name, x, starts, ends, axes)
    else:
        ts_node = ts.frontend.onnx.slice_v3(node_name, x, starts, ends, axes, steps)

    return ts_node,


register_layer_converter("Slice", convert_slice_layer)


def convert_clip_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 3 or len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    if len(input_nodes) == 3:
        min_value = ts.zoo.to_const(input_nodes[1], "min")
        max_value = ts.zoo.to_const(input_nodes[2], "max")
    else:
        min_value = attr_dict["min"]
        max_value = attr_dict["max"]

    min_value = float(min_value)
    max_value = float(max_value)

    print("--##    min: {}".format(min_value))
    print("--##    max: {}".format(max_value))

    if abs(min_value - 0) < 1e-6:
        ts_node = ts.zoo.relu_max(node_name, x=x, max=max_value)
    else:
        raise NotImplementedError("Clip min={}, max={}".format(min_value, max_value))

    return ts_node,


register_layer_converter("Clip", convert_clip_layer)


def convert_leaky_relu_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    alpha = 0.01
    if "alpha" in attr_dict:
        alpha = attr_dict["alpha"]
        print("--##    alpha: {}".format(alpha))

    ts_node = ts.zoo.leaky_relu(node_name, x=x, scale=alpha)

    return ts_node,


register_layer_converter("LeakyRelu", convert_leaky_relu_layer)


def convert_cast_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    to = attr_dict["to"]

    dtype2dtype = {
        onnx.TensorProto.UNDEFINED: ts.dtype.VOID,
        # Basic types.
        onnx.TensorProto.FLOAT: ts.dtype.FLOAT32,   # float
        onnx.TensorProto.UINT8: ts.dtype.UINT8,   # uint8_t
        onnx.TensorProto.INT8: ts.dtype.INT8,    # int8_t
        onnx.TensorProto.UINT16: ts.dtype.UINT16,  # uint16_t
        onnx.TensorProto.INT16: ts.dtype.INT16,   # int16_t
        onnx.TensorProto.INT32: ts.dtype.INT32,   # int32_t
        onnx.TensorProto.INT64: ts.dtype.INT64,   # int64_t
        onnx.TensorProto.STRING: ts.dtype.VOID,  # string   # not support string cast
        onnx.TensorProto.BOOL: ts.dtype.BOOLEAN,    # bool

        # IEEE754 half-precision floating-point format (16 bits wide).
        # This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
        onnx.TensorProto.FLOAT16: ts.dtype.FLOAT16,

        onnx.TensorProto.DOUBLE: ts.dtype.FLOAT64,
        onnx.TensorProto.UINT32: ts.dtype.UINT32,
        onnx.TensorProto.UINT64: ts.dtype.UINT64,
        onnx.TensorProto.COMPLEX64: ts.dtype.COMPLEX64,     # complex with float32 real and imaginary components
        onnx.TensorProto.COMPLEX128: ts.dtype.COMPLEX128,    # complex with float64 real and imaginary components

        # Non-IEEE floating-point format based on IEEE754 single-precision
        # floating-point number truncated to 16 bits.
        # This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
        onnx.TensorProto.BFLOAT16: ts.dtype.VOID,   # not support for now

    }

    if to not in dtype2dtype:
        raise NotImplementedError("Unknown to={}".format(to))

    dtype = dtype2dtype[to]

    if dtype == ts.dtype.VOID:
        raise NotImplementedError("Unsupported to={}".format(to))

    ts_node = ts.zoo.cast(name=node_name, x=x, dtype=dtype)

    return ts_node,


register_layer_converter("Cast", convert_cast_layer)


def convert_floor(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.menu.op(node_name, "floor", [x])

    return ts_node,


register_layer_converter("Floor", convert_floor)


def convert_equal_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    lhs = input_nodes[0]
    rhs = input_nodes[1]

    ts_node = ts.zoo.equal(node_name, lhs=lhs, rhs=rhs)

    return ts_node,


register_layer_converter("Equal", convert_equal_layer)


def convert_softplus_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.softplus(node_name, x=x)

    return ts_node,


register_layer_converter("Softplus", convert_softplus_layer)


def convert_where_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    assert len(input_nodes) == 3
    assert len(output_names) == 1

    node_name = output_names[0]

    cond = input_nodes[0]
    x = input_nodes[1]
    y = input_nodes[2]

    ts_node = ts.zoo.where(node_name, cond=cond, x=x, y=y)

    return ts_node,


register_layer_converter("Where", convert_where_layer)


def convert_constant_of_shape_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    value = None
    if "value" in attr_dict:
        value = attr_dict["value"]
        print("--##    value: {}".format(value))

    ts_node = ts.zoo.constant_of_shape(node_name, x=x, value=value)

    return ts_node,


register_layer_converter("ConstantOfShape", convert_constant_of_shape_layer)


def convert_resize_asymmetric(node, input_nodes, output_names, attr_dict):
    node_name = output_names[0]

    assert len(input_nodes) >= 3
    x = input_nodes[0]
    # roi = input_nodes[1]  # just ignore roi in non-tf_crop_and_resize mode
    scales = input_nodes[2]

    mode = attr_dict["mode"]
    mode2type = {
        "nearest": ts.zoo.Type.resize2d_type.hard,      # nearest means hard in TS
        "linear": ts.zoo.Type.resize2d_type.linear,
        "cubic": ts.zoo.Type.resize2d_type.cubic,
    }
    if mode not in mode2type:
        raise NotImplementedError("mode={}".format(mode))
    type = mode2type[mode]

    if_scale_empty = False
    try:
        scales_data = ts.zoo.to_const(scales, "scales")
        scales_count = numpy.prod(scales_data.shape)
        if_scale_empty = scales_count == 0
    except:
        pass

    if not if_scale_empty:
        try:
            scales = ts.zoo.to_const(scales, "scales")
            scales_val = scales[-2]
            return ts.zoo.sample2d(name=node_name, x=x, scale=scales_val, type=type)
        except:
            return ts.zoo.sample2d_v2(name=node_name, x=x, scale=scales, type=type)
    else:
        raise NotImplementedError("mode={}, scales={}".format(mode, scales))


def convert_resize_half_pixel(node, input_nodes, output_names, attr_dict):
    node_name = output_names[0]

    assert 3 <= len(input_nodes) <= 4
    x = input_nodes[0]
    # roi = input_nodes[1]  # just ignore roi in non-tf_crop_and_resize mode
    scales = input_nodes[2]

    sizes = None
    if len(input_nodes) == 4:
        sizes = input_nodes[3]
    else:
        scales_val = ts.zoo.to_const(value=scales, name="scales")
        input_shape = ts.zoo.to_const(value=x, name="x").shape
        assert len(scales_val) == len(input_shape)
        sizes = input_shape * scales_val

    mode = attr_dict["mode"]
    mode2type = {
        "nearest": ts.zoo.Type.resize2d_type.hard,      # nearest means hard in TS
        "linear": ts.zoo.Type.resize2d_type.linear,
        "cubic": ts.zoo.Type.resize2d_type.cubic,
    }
    if mode not in mode2type:
        raise NotImplementedError("mode={}".format(mode))
    type = mode2type[mode]

    return ts.zoo.resize2d(name=node_name, x=x, size=sizes, type=type)


def convert_resize_v11_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))
    opset_version = 11

    attribute = node.attribute
    attr_dict = {
        "coordinate_transformation_mode": "half_pixel",
        "cubic_coeff_a": -0.75,
        "exclude_outside": 0,
        "extrapolation_value": 0,
        "mode": "nearest",  # linear, cubic
        "nearest_mode": "round_prefer_floor",
    }
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) in {3, 4}
    assert len(output_names) == 1

    coordinate_transformation_mode = attr_dict["coordinate_transformation_mode"]

    if coordinate_transformation_mode == "half_pixel":
        return convert_resize_half_pixel(node, input_nodes, output_names, attr_dict)
    elif coordinate_transformation_mode == "asymmetric":
        return convert_resize_asymmetric(node, input_nodes, output_names, attr_dict)
    else:
        raise NotImplementedError("coordinate_transformation_mode={}".format(coordinate_transformation_mode))


register_layer_version_converter("Resize", 11, convert_resize_v11_layer)


def convert_resize_v10_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    scales = input_nodes[1]

    mode = attr_dict["mode"]
    mode2type = {
        "nearest": ts.zoo.Type.resize2d_type.hard,      # nearest means hard in TS
        "linear": ts.zoo.Type.resize2d_type.linear,
        "bilinear": ts.zoo.Type.resize2d_type.linear,
    }
    if mode not in mode2type:
        raise NotImplementedError("mode={}".format(mode))
    type = mode2type[mode]

    try:
        scales = ts.zoo.to_const(scales, "scales")
    except:
        # do common sample image
        x_shape = ts.zoo.shape(name=x.name + "_shape", x=x)
        float_x_shape = ts.zoo.cast(name=x.name + "_float_shape", x=x_shape, dtype=ts.dtype.FLOAT32)
        scaled_size = ts.zoo.mul(name=x.name + "_scale_size", lhs=float_x_shape, rhs=scales)
        int_scaled_size = ts.zoo.cast(name=x.name + "_int_size", x=scaled_size, dtype=dtype.ts.dtype.INT32)
        return ts.zoo.resize2d(name=node_name, x=x, size=int_scaled_size, type=type)

    # do static scale image
    scales = numpy.asarray(scales)

    if scales.shape != (4,):
        raise NotImplementedError("scales={}".format(scales))

    if scales[0] != 1 or scales[1] != 1:
        raise NotImplementedError("scales={}".format(scales))

    if scales[2] != scales[3]:
        raise NotImplementedError("scales={}".format(scales))

    scale = scales[3]

    ts_node = ts.zoo.sample2d(name=node_name, x=x, scale=scale, type=type)

    return ts_node,


register_layer_version_converter("Resize", 10, convert_resize_v10_layer)


def convert_lstm_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {
        # "activation_alpha": [],
        # "activation_beta": [],
        # "activations": [],
        # "clip": 0,
        "direction": "forward",
        "hidden_size": 0,
        # "input_forget": 0,
    }

    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)
        print("--##    {}: {}".format(str(attr.name), attr_dict[str(attr.name)]))

    assert 3 <= len(input_nodes) <= 8
    assert 0 <= len(output_names) <= 3

    x = input_nodes[0]
    w = input_nodes[1]
    r = input_nodes[2]
    b = input_nodes[3]
    sequence_lens = input_nodes[4] if len(input_nodes) >= 4 else None
    initial_h = input_nodes[5] if len(input_nodes) >= 5 else 0
    initial_c = input_nodes[6] if len(input_nodes) >= 6 else 0

    assert sequence_lens == VoidNode, NotImplementedError("Not support sequence_lens = {}".format(sequence_lens))

    if initial_h == 0:
        set_init_h = numpy.zeros(shape=(w.shape(0), x.shape(1), r.shape(2)), dtype=numpy.float32)
        initial_h = ts.zoo.to_node(value=set_init_h, name="initial_h", device=ts.device.CPU, dtype=numpy.float32)

    if initial_c == 0:
        set_init_c = numpy.zeros(shape=(w.shape(0), x.shape(1), r.shape(2)), dtype=numpy.float32)
        initial_c = ts.zoo.to_node(value=set_init_c, name="initial_c", device=ts.device.CPU, dtype=numpy.float32)

    node_name = output_names[0]
    node = ts.zoo.LSTM(node_name, x, w, r, b, initial_h, initial_c, attr_dict["direction"], attr_dict["hidden_size"])

    return [ts.menu.field(name=output_names[i], input=node, offset=i) for i in range(len(output_names))]


register_layer_version_converter("LSTM", 7, convert_lstm_layer)


def convert_hard_sigmoid_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)
        print("--##    {}: {}".format(str(attr.name), attr_dict[str(attr.name)]))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    alpha = 0.2
    beta = 0.5
    if "alpha" in attr_dict:
        alpha = attr_dict["alpha"]
        print("--##    alpha: {}".format(alpha))
    if "beta" in attr_dict:
        beta = attr_dict["beta"]
        print("--##    beta: {}".format(beta))

    ts_node = ts.zoo.hard_sigmoid(node_name, x=x, alpha=alpha, beta=beta)

    return ts_node,


register_layer_converter("HardSigmoid", convert_hard_sigmoid_layer)


def convert_identity_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    ts_node = ts.zoo.copy(node_name, x)

    return ts_node,


register_layer_converter("Identity", convert_identity_layer)


def convert_scatter_nd_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)
        print("--##    {}: {}".format(str(attr.name), attr_dict[str(attr.name)]))

    assert len(input_nodes) == 3
    assert len(output_names) == 1

    data = input_nodes[0]
    indices = input_nodes[1]
    updates = input_nodes[2]

    node_name = output_names[0]

    ts_node = ts.zoo.scatter_nd(node_name, data, indices, updates)

    return ts_node,


register_layer_converter("ScatterND", convert_scatter_nd_layer)


def convert_pad_v11_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) in {2, 3}
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    mode = Name.constant
    if Name.Attr.mode in attr_dict:
        mode = attr_dict[Name.Attr.mode]
        print("--##    mode: {}".format(mode))

    if mode != Name.constant:
        raise NotImplementedError("mode={}".format(mode))

    pads = input_nodes[1]
    pads_name = pads.name

    pads = ts.zoo.reshape(pads_name + "_reshape", pads, [2, -1])
    pads = ts.zoo.transpose(pads_name + "_transpose", pads, [1, 0])

    value = 0
    if len(input_nodes) > 2:
        value = ts.zoo.to_const(input_nodes[2], "value")
        print("--##    value: {}".format(value))

    ts_node = ts.zoo.pad(node_name, x=x, padding=pads, padding_value=value)

    return ts_node,


register_layer_version_converter("Pad", 11, convert_pad_v11_layer)


def convert_prelu_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    slope = input_nodes[1]

    slope_arr = ts.zoo.to_const(value=slope, name="slope")
    max_dim_order = numpy.argmax(slope_arr.shape)

    slope_val = slope_arr.reshape(slope_arr.shape[max_dim_order], -1)
    slope_node_val = numpy.squeeze(slope_val, axis=1)

    ts_node = ts.zoo.prelu(name=node_name, x=x, dim=1, slope=slope_node_val)

    return ts_node,


register_layer_converter("PRelu", convert_prelu_layer)


def convert_squeeze_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axes = attr_dict[Name.Attr.axes]
    print("--##    axes: {}".format(axes))

    ts_node = ts.zoo.squeeze(name=node_name, x=x, axes=axes)

    return ts_node,


register_layer_converter("Squeeze", convert_squeeze_layer)


def convert_argmax_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 1
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]

    axis = attr_dict['axis']
    keepdims = attr_dict['keepdims']
    print("--##    axis: {}".format(axis))
    print("--##    keepdims: {}".format(keepdims))

    if keepdims:
        raise NotImplementedError("keepdims={}".format(keepdims))

    ts_node = ts.menu.op(node_name, "argmax", [x])
    ts_node.set('dim', axis, numpy.int32)

    return ts_node,


register_layer_version_converter("ArgMax", 11, convert_argmax_layer)


def convert_gather_elements_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    indices = input_nodes[1]

    axis = 0
    if Name.Attr.axis in attr_dict:
        axis = attr_dict[Name.Attr.axis]
        print("--##    axis: {}".format(axis))

    ts_node = ts.menu.op(node_name, "gather_elements", [x, indices])
    ts_node.set('axis', axis, numpy.int32)

    return ts_node,


register_layer_version_converter("GatherElements", 11, convert_gather_elements_layer)


def convert_split_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2

    # node_name = output_names[0]
    node_name = node.name

    x = input_nodes[0]
    split = input_nodes[1]

    axis = 0
    if Name.Attr.axis in attr_dict:
        axis = attr_dict[Name.Attr.axis]
        print("--##    axis: {}".format(axis))

    ts_node = ts.menu.op(node_name, "split", [x, split])
    ts_node.set('axis', axis, numpy.int32)

    output_nodes = [ts.menu.field(name, ts_node, i) for i, name in enumerate(output_names)]

    return output_nodes


register_layer_version_converter("Split", 17, convert_split_layer)


def convert_pow_layer(node, input_nodes, output_names):
    # type: (onnx.NodeProto, List[ts.Node], List[str]) -> List[ts.Node]
    print("--# -=[ Converting {} layer: {} -> {} ]=-".format(node.op_type, [n.name for n in input_nodes], output_names))

    attribute = node.attribute
    attr_dict = {}
    for attr in attribute:
        attr_dict[str(attr.name)] = topy(attr)

    assert len(input_nodes) == 2
    assert len(output_names) == 1

    node_name = output_names[0]

    x = input_nodes[0]
    y = input_nodes[1]


    try:
        y_value = ts.zoo.to_const(y, "y")
        ts_node = ts.menu.op(node_name, "pow", [x])
        ts_node.set("y", y_value, numpy.float32)
    except:
        ts_node = ts.menu.op(node_name, "pow_v2", [x, y])

    return ts_node,


register_layer_version_converter("Pow", 17, convert_pow_layer)
