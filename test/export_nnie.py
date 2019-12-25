#!/usr/bin/env python

import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.mxnet.converter import convert
import tensorstack as ts
from stackbuilder.export.dumper import Dumper, Calibrator
from tensorstack.backend.api import *
import cv2
import numpy

from stackbuilder.export.nnie.exporter import NNIEExporter


class MyCalibrator(Calibrator):
    def __init__(self, dataset):
        # type: (str) -> None
        """
        root or image list
        :param dataset:
        """
        if sys.version > '3':
            basestring = str
        image_ext = {".jpg"}
        dataroot = ""
        filelist = []
        if os.path.isdir(dataset):
            dataroot = dataset
            listdir = os.listdir(dataroot)
            for filename in listdir:
                _, ext = os.path.splitext(filename)
                if ext in image_ext:
                    filelist.append(filename)
            filelist.sort()
        elif os.path.isfile(dataset):
            dataroot = os.getcwd()
            with open(dataset, "r") as f:
                for line in f.readlines():
                    assert isinstance(line, basestring)
                    filelist.append(line.strip())
        else:
            raise ValueError("param 1 must be existed path or file")
        self.__filelist = [path if os.path.isabs(path) else os.path.join(dataroot, path)
                           for path in filelist]
        self.__next_index = 0
        self.__image_filter = None

    @property
    def image_filter(self):
        return self.__image_filter

    @image_filter.setter
    def image_filter(self, value):
        assert callable(value)
        self.__image_filter = value

    def number(self):  # type: () -> int
        return len(self.__filelist)

    def next(self):
        # type: () -> Optional[List[numpy.ndarray]]
        while True:
            if self.__next_index >= len(self.__filelist):
                return None
            filepath = self.__filelist[self.__next_index]
            self.__next_index += 1
            image = cv2.imread(filepath)
            if image is None:
                print("[WARNING]: Fail to open: {}".format(filepath))
                continue
            data = numpy.expand_dims(image, 0)

            if self.image_filter is not None:
                data = self.image_filter(data)

            return [data, ]


class MyImageFilter(object):
    def __init__(self):
        device = Device("cpu")
        self.__workbench = Workbench(device=device)
        self.__image_filter = ImageFilter(device=device)
        self.__image_filter.center_crop(248, 248)
        self.__image_filter.to_float()
        self.__image_filter.to_chw()

    def dispose(self):
        self.__image_filter.dispose()
        self.__workbench.dispose()

    def __call__(self, image):
        self.__workbench.setup_context()
        input = Tensor(image)
        output = self.__image_filter.run(input)
        output_numpy = output.numpy
        input.dispose()
        output.dispose()
        return output_numpy


def test():
    input_module = "../rawmd/RN30.light.tsm"
    input_shape = [[1, 3, 248, 248]]
    output_module = "./temp/RN30.nnie.tsm"
    dataset_root = "/Users/seetadev/Documents/Files/nnie/quantization_248x248_2"

    print("=========== Start test ==============")
    exporter = NNIEExporter()
    calibrator = MyCalibrator(dataset_root)
    image_filter = MyImageFilter()
    calibrator.image_filter = image_filter

    exporter.load(input_module, input_shape)
    # exporter.export_caffe(output_module)
    exporter.export_nnie_cfg(output_module, calibrator)


if __name__ == '__main__':
    test()

