"""
Dump temp output data to file, support cache
"""

from typing import Tuple, List, Set, Dict, Optional
import numpy

import tensorstack as ts
from tensorstack.backend.api import Module, Workbench, Device

from io import BytesIO
import hashlib
import os

import sys
if sys.version > '3':
    basestring = str


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

    def number(self):
        # type: () -> int
        """
        Get full data number
        :return:
        """
        raise NotImplementedError


def _count_node_name(outputs, cache=None, count=None):
    # type: (List[ts.Node], Set[ts.Node], Dict[ts.Node, int]) -> Dict[ts.Node, int]
    if cache is None:
        cache = set()
    if count is None:
        count = {}
    for node in outputs:
        if node in cache:
            continue
        cache.add(node)
        if node in count:
            count[node.name] += 1
        else:
            count[node.name] = 1
        _count_node_name(node.inputs, cache, count)
    return count


def _check_output_names(outputs, count_names):
    # type: (List[ts.Node], List[str]) -> None
    node_name_count = _count_node_name(outputs)
    for name in count_names:
        if name not in node_name_count:
            raise ValueError("name \"{}\" not in graph.".format(name))
        if node_name_count[name] > 1:
            raise ValueError("name \"{}\" not in graph.".format(name))


def _create_workbench(module, device="cpu", id=0):
    # type: (ts.Module, str, int) -> Tuple[_HookInputWorkbench, str]
    assert isinstance(module, ts.Module)
    buffer = BytesIO()
    ts.Module.Save(buffer, module)
    value = buffer.getvalue()
    md5 = hashlib.md5(value).hexdigest()
    buffer.seek(0)
    api_module = Module.Load(buffer)
    buffer.close()
    api_workbench = _HookInputWorkbench.Load(module, api_module, Device(device, id))
    api_module.dispose()
    return api_workbench, md5


class _HookInputWorkbench(object):
    def __init__(self, module, api_module, device):
        # type: (ts.Module, Module, Device) -> None
        self.__module = module
        self.__workbench = Workbench.Load(api_module, device)
        self.__input_names = [i.name for i in self.__module.inputs]
        self.__map_input_value = {}

    def dispose(self):
        self.__workbench.dispose()

    @staticmethod
    def Load(module, api_module, device):
        # type: (ts.Module, Module, Device) -> _HookInputWorkbench
        return _HookInputWorkbench(module, api_module, device)

    def input(self, slot, tensor):
        assert isinstance(slot, int)
        tensor_numpy = numpy.asarray(tensor)
        self.__map_input_value[self.__module.inputs[slot].name] = tensor_numpy
        self.__workbench.input(slot, tensor)

    def run_hook(self, outputs):
        output_names = []
        for output in outputs:
            if output not in self.__input_names:
                output_names.append(output)
        if len(output_names) > 0:
            self.__workbench.run_hook(output_names)

    def output(self, slot):
        assert isinstance(slot, basestring)
        if slot in self.__map_input_value:
            return self.__map_input_value[slot]
        else:
            output = self.__workbench.output(slot)
            output_numpy = output.numpy
            output.dispose()
            return output_numpy


class Dumper(object):
    def __init__(self, module, outputs, calibrator, batch_size=1, cache=None, device="cpu", device_id=0):
        # type: (ts.Module, List[str], Calibrator, int, str, str, int) -> None
        assert isinstance(module, ts.Module), "param 1 must be ts.Module"
        assert isinstance(outputs, (list, tuple)) and \
               all([isinstance(s, basestring) for s in outputs]), "param 2 must list of string"
        assert hasattr(calibrator, "next"), "param 3 must hasattr next"
        assert hasattr(calibrator, "number"), "param 3 must hasattr next"
        assert isinstance(batch_size, (int, type(None))),  "param 4 must be int"
        assert isinstance(cache, (basestring, type(None))),  "param 5 must be str"
        # step 1. check each name
        _check_output_names(module.outputs, outputs)
        # step 2. build ts.backend.api.Workbench
        workbench, md5 = _create_workbench(module, device, device_id)
        print("[INFO]: Load workbench: {}".format(md5))
        if batch_size < 1:
            batch_size = 1

        if cache is None:
            cache = "/tmp"

        if batch_size > 1:
            raise ValueError("batch_size only support 1 for now version")

        self.__workbench = workbench
        self.__md5 = md5
        self.__calibrator = calibrator
        self.__cache = cache
        self.__batch_size = batch_size
        self.__outputs = list(outputs)

        class SingleCalibrator(Calibrator):
            def __init__(self, calibrator):
                assert hasattr(calibrator, "next")
                self.__data = None
                self.__calibrator = calibrator
                self.__i = 0

            def number(self):  # type: () -> int
                return self.__calibrator.number()

            def next(self):
                while True:
                    if self.__data is None:
                        self.__i = 0
                        self.__data = self.__calibrator.next()
                        if self.__data is None:
                            return None
                    N = self.__data[0].shape[0]
                    if self.__i >= N:
                        self.__data = None
                        continue
                    i = self.__i
                    self.__i += 1
                    return [t[i:i+1] for t in self.__data]

        self.__single = SingleCalibrator(self.__calibrator)

    def dispose(self):
        self.__workbench.dispose()

    def _data_md5(self, inputs):
        # type: (List[numpy.ndarray]) -> None
        data = ts.tensor.PackedTensor(inputs)
        buffer = BytesIO()
        ts.tensor.write_tensor(buffer, data)
        value = buffer.getvalue()
        buffer.close()
        md5 = hashlib.md5(value).hexdigest()
        return md5

    def _cache_file_path(self, inputs, output):
        # type: (List[numpy.ndarray], str) -> str
        output = output.replace("/", "#")
        output = output.replace("\\", "#")
        filepath = os.path.join(self.__cache,
                                "tensorstack", "dump",
                                self.__md5,
                                "{}.{}.t".format(self._data_md5(inputs), output),
                                )
        return filepath

    def _cache_output(self, inputs, output):
        # type: (List[numpy.ndarray], str) -> Optional[numpy.ndarray]
        filepath = self._cache_file_path(inputs, output)
        if os.path.isfile(filepath):
            return ts.tensor.read(filepath)
        return None

    def _write_tensor(self, filepath, data):
        # type: (str, numpy.ndarray) -> None
        try:
            fileroot, filename = os.path.split(filepath)
            if not os.path.isdir(fileroot):
                os.makedirs(fileroot)
            ts.tensor.write(filepath, data)
        except:
            pass

    def _infer_output(self, inputs, outputs):
        # type: (List[numpy.ndarray], List[str]) -> List[numpy.ndarray]
        list_output_i = []
        output_tensors = []
        for i, output in enumerate(outputs):
            value = self._cache_output(inputs, output)
            if value is None:
                list_output_i.append((output, i))
            output_tensors.append(value)
        if len(list_output_i) > 0:
            for i, input in enumerate(inputs):
                self.__workbench.input(i, input)
            hook_outputs = set([oi[0] for oi in list_output_i])
            self.__workbench.run_hook(list(hook_outputs))

            for output, i in list_output_i:
                data_numpy = self.__workbench.output(output)

                output_tensors[i] = data_numpy

                if output not in hook_outputs:
                    continue
                hook_outputs.remove(output)
                self._write_tensor(self._cache_file_path(inputs, output), data_numpy)

        return output_tensors

    def next(self):
        inputs = self.__single.next()
        if inputs is None:
            return None
        return self._infer_output(inputs, self.__outputs)
