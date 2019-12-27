#!/usr/bin/env python

from typing import Union
import tensorstack as ts

from typing import Tuple, List, Dict
import numpy
import os

from stackfence.spliter import MainGraph
from .. import fridge
from . import nnie_spliter
from . import nnie_fence

from . import nnie_caffe
from .. import dumper
from . import nnie_config

import sys
if sys.version > "3":
    basestring = str

"""
For add new converter, See .nnie_spliter.get_spliter for graph spliter; .nnie_caffe for graph converter
"""


def split_caffe(input_tsm, output_tsm, subdir=None, input_shape=None, export_main=False):
    # type: (Union[str, ts.Module], str, str, Union[List[Tuple[int]], Dict[str, Tuple[int]]], bool) -> [None, MainGraph]
    """
    Split support node to sub graph.
    :param input_tsm:
    :param output_tsm:
    :param subdir:
    :return:
    Notice: output main tsm module and sub caffe models
    Every output caffe named {output_tsm}.<i>.wk.prototxt and caffemodel
    """
    assert isinstance(subdir, (type(None), basestring))
    module = input_tsm
    if isinstance(module, basestring):
        with open(module, "rb") as f:
            module = ts.Module.Load(f)
    assert isinstance(module, ts.Module)
    filepath = os.path.abspath(output_tsm)
    output_root, filename_ext = os.path.split(filepath)
    filename, ext = os.path.splitext(filename_ext)

    output_caffe_root = output_root
    if subdir is not None:
        output_caffe_root = os.path.join(output_root, subdir)

    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    if not os.path.isdir(output_caffe_root):
        os.makedirs(output_caffe_root)

    outputs = module.outputs
    inputs = module.inputs
    print("[INFO]: Freezing graph...")
    outputs, inputs = fridge.freeze(outputs, inputs, input_shape)
    print("[INFO]: Split graph...")
    outputs, inputs = nnie_fence.get_fence().convert(outputs, after=inputs)
    main_graph = nnie_spliter.get_spliter().split(outputs, inputs)
    print("[INFO]: Convert graph...")
    nnie_count = main_graph.sub_count()
    for i in range(nnie_count):
        output_name_body = "{}.{}.wk".format(filename, i)
        print("[INFO]: Exporting... {}".format(
            os.path.relpath(os.path.join(output_caffe_root, output_name_body), output_root)))
        output_prototxt = "{}.prototxt".format(output_name_body)
        output_caffemodel = "{}.caffemodel".format(output_name_body)
        sub_node = main_graph.sub_node(i)
        sub_graph = main_graph.sub_graph(i)
        nnie_caffe.convert(sub_graph.outputs, sub_graph.inputs,
                           os.path.join(output_caffe_root, output_prototxt),
                           os.path.join(output_caffe_root, output_caffemodel))

    if export_main:
        print("[INFO]: Exporting... {}".format(filepath))
        main_module = ts.Module()
        main_module.load(main_graph.outputs)
        main_module.sort_inputs(main_graph.inputs)

        with open(filepath, "wb") as f:
            ts.Module.Save(f, main_module)

    return main_graph


def export_image_list(module, output_names, calibrator, main, output_root, cache=None, device="cpu", device_id=0):
    # type: (ts.Module, List[str], dumper.Calibrator, str, str, str, str, int) -> Dict[str, str]
    output_root = os.path.abspath(output_root)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)
    map_name_filenames = {}
    for name in output_names:
        fixed_name = name.replace("/", "=")
        fixed_name = fixed_name.replace("\\", "=")
        filename = os.path.join(output_root, "{}.{}.txt".format(main, fixed_name))
        map_name_filenames[name] = filename

    extractor = dumper.Dumper(module, output_names, calibrator, 1, cache=cache, device=device, device_id=device_id)

    map_name_features = {}
    map_name_file_stream = {}
    for name in output_names:
        map_name_features[name] = []
        filename = map_name_filenames[name]
        map_name_file_stream[name] = open(filename, "w")

    P = [0, 0, calibrator.number()]
    S = len(output_names)

    def process_show():
        sys.stdout.write("\r[{:.2g}/{}/{}]   ".format(float(P[0]) / S, P[1], P[2]))
        sys.stdout.flush()

    process_show()

    def flush():
        for name in output_names:
            features = map_name_features[name]
            stream = map_name_file_stream[name]
            lines = []
            for data in features:
                flatten_data = numpy.asarray(data).reshape([-1])
                # print("Data size: {}={}".format(len(flatten_data), "*".join([str(i) for i in data.shape])))
                lines.append(" ".join(["%.3g" % f for f in flatten_data]))
                lines.append("\n")
                P[0] += 1
                process_show()
            stream.writelines(lines)
            stream.flush()
            map_name_features[name] = []

    N = 100
    count = 0
    while True:
        features_list = extractor.next()
        if features_list is None:
            break
        for i, name in enumerate(output_names):
            map_name_features[name].append(features_list[i])

        P[1] += 1
        process_show()

        count += 1
        if count >= N:
            count = 0
            flush()

    flush()
    process_show()
    print("\n[INFO]: Build image list done.")

    for name in output_names:
        map_name_file_stream[name].close()

    return map_name_filenames


class Calibrator(object):
    """
    Return valid set
    """
    def next(self):
        # type: () -> Tuple[numpy.ndarray]
        """
        Get next sample for quantification, tuple means multi inputs
        :return:
        """
        raise NotImplementedError


def caffe2nnie(prototxt, caffemodel, wk, calibrator):
    # type: (str, str, str, Calibrator) -> None
    """
    Convert caffe model to wk file, use bench and dataset build
    :param prototxt:
    :param caffemodel:
    :param wk:
    :return:
    Only convert base CNN layer for now. so net type equals to 1
    """
    pass


def fuse_nnie(input_tsm, output_tsm):
    # type: (str, str) -> None
    """
    Fuse tsm + nnies to single tsm
    :param input_tsm:
    :param output_tsm:
    :return:
    """
    pass


def export(input_tsm, output_tsm, dataset):
    # type: (str, str, Calibrator) -> ts.Module
    """
    convert to nnie model, using dataset quantize
    :param input_tsm:
    :param output_tsm:
    :param dataset:
    :return:
    """
    # 1. export_split
    # 2. caffe2nnie
    # 3. fuse_nnie
    pass


class NetInferer(object):
    def run(self, inputs, outputs):
        # type: (List[numpy.ndarray], List[str]) -> List[numpy.ndarray]
        """
        :param inputs: length input count
        :param outputs: get output names
        :return:
        """
        raise NotImplementedError


class NNIEExporter(object):
    def __init__(self, nnie_version=None, host_device="cpu", host_device_id=0):
        self.__original_module = None   # update by load
        self.__input_shape = None       # update by load
        self.__cmd = None   # path to cmd line
        self.__cache = None # cache temp files
        self.__host_device = host_device
        self.__host_device_id = host_device_id
        if nnie_version is not None and nnie_version not in {"11", "12"}:
            raise ValueError("nnie version must be string 11 or 12")
        self.__nnie_version = nnie_version
        pass

    def load(self, module, input_shape=None):
        # type: (ts.Module, Union[List[Tuple[int]], Dict[str, Tuple[int]]]) -> None
        if isinstance(module, basestring):
            print("[INFO]: Loading... {}".format(module))
            with open(module, "rb") as f:
                module = ts.Module.Load(f)
        assert isinstance(module, ts.Module)
        self.__original_module = module
        self.__input_shape = input_shape

    def export_caffe(self, filename, subdir=None):
        # type: (str, str) -> None
        if self.__original_module is None:
            raise ValueError("call load fist be before export_caffe")
        split_caffe(self.__original_module, filename, subdir, self.__input_shape)

    def _split_root_name_ext(self, filename):
        # type: (str) -> Tuple[str, str, str]
        filepath = os.path.abspath(filename)
        root, name_ext = os.path.split(filepath)
        name, ext = os.path.splitext(name_ext)
        return root, name, ext

    def export_nnie_cfg(self, filename, calibrator):
        # type: (str, Calibrator) -> List[str]
        """
        :param filename:
        :param calibrator:
        :return: list of cfg filename
        nnie operator define:
        nnie(List[Tensor]) -> List[Tensor]
        attrs:
            `input_count` `Int` `Required`
            `output_count` `Int` `Required`
            `wk_file` `String` `Required` path to wk file
            `wk_buffer` `ByteArray` `Optional` load from this buffer if wk_buffer set
        ``
        """
        if self.__original_module is None:
            raise ValueError("call load fist be before export_nnie_cfg")

        output_root, output_name, output_ext = self._split_root_name_ext(filename)
        # 1. split caffe
        main_graph = split_caffe(input_tsm=self.__original_module,
                                 output_tsm=filename,
                                 subdir="model",
                                 input_shape=self.__input_shape,
                                 export_main=False)
        # 2. get image list
        sub_graph_inputs = set()
        sub_graph_count = main_graph.sub_count()
        for i in range(sub_graph_count):
            for input in main_graph.sub_graph(i).inputs:
                sub_graph_inputs.add(input.name)
        sub_graph_inputs = list(sub_graph_inputs)
        print("[INFO]: Building image list... ")
        map_name_image_list = export_image_list(module=self.__original_module,
                                                output_names=sub_graph_inputs,
                                                calibrator=calibrator,
                                                main=output_name,
                                                output_root=os.path.join(output_root, "data"),
                                                cache=self.__cache,
                                                device=self.__host_device,
                                                device_id=self.__host_device_id)
        summery_configs = []
        # 3. write nnie cfg file
        for i in range(sub_graph_count):
            node = main_graph.sub_node(i)
            graph = main_graph.sub_graph(i)

            # ref wk filename
            wk_instruction_name = os.path.join("inst", "{}.{}".format(output_name, i))
            wk_filename = wk_instruction_name + ".wk"
            print("[INFO]: Waiting... {}".format(wk_filename))

            # write config
            cfg = nnie_config.Config()
            cfg.prototxt_file = os.path.join("model", "{}.{}.wk.prototxt".format(output_name, i))
            cfg.caffemodel_file = os.path.join("model", "{}.{}.wk.caffemodel".format(output_name, i))
            cfg.instruction_name = wk_instruction_name
            cfg.batch_num = 0
            cfg.log_level = 0
            for graph_input in graph.inputs:
                iname = graph_input.name
                image_list = os.path.relpath(map_name_image_list[iname], output_root)
                cfg.image_list.append(image_list)
                cfg.image_type.append(0)

            full_cfg_path = os.path.join(output_root, "{}.{}.wk.cfg".format(output_name, i))
            cfg.write(full_cfg_path)
            summery_configs.append(full_cfg_path)

            # update node
            node.op = "nnie"
            node.set("input_count", len(graph.inputs), numpy.int32)     # required
            node.set("output_count", len(graph.outputs), numpy.int32)   # required
            node.set("wk_file", wk_filename)    # required

        # 4. write main tsm file
        main_module = ts.Module()
        main_module_outputs, main_module_inputs = \
            nnie_fence.back_fence().convert(main_graph.outputs, after=main_graph.inputs)
        main_module.load(main_module_outputs)
        main_module.sort_inputs(main_module_inputs)

        if not os.path.isdir(output_root):
            os.makedirs(output_root)

        with open(filename, "wb") as f:
            ts.Module.Save(f, main_module)

        # final. make inst dir for output file
        inst_root = os.path.join(output_root, "inst")
        if not os.path.isdir(inst_root):
            os.makedirs(inst_root)

        print("[INFO]: Writen file: {}".format(filename))

        # PS. than writen nnie wk file, has some pattern
        return summery_configs

    @staticmethod
    def FuseNNIE(input_filename, output_filename):
        # type: (str, str) -> None
        """
        Fuse all nnie operators' wk_file to wk_buffer
        :param input_filename:
        :param output_filename:
        :return:
        """
        pass

    def export_wk(self, calibrator):
        # type: (Calibrator) -> None
        pass



