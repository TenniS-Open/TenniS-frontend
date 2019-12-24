#!/usr/bin/env python
# coding: UTF-8

import sys
import os

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from stackbuilder.export.dumper import Dumper, Calibrator
import tensorstack as ts
from tensorstack.backend.api import *

from typing import List, Optional
import numpy
import cv2
import math


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


if __name__ == "__main__":
    path = "../rawmd/RN30.light.tsm"
    dataroot = "/home"

    calibrator = MyCalibrator("/Users/seetadev/Documents/Files/nnie/quantization_248x248")
    image_filter = MyImageFilter()
    calibrator.image_filter = image_filter

    with open(path, "rb") as f:
        module = ts.Module.Load(f)
    blob_names = [o.name for o in module.outputs]
    dumper = Dumper(module, blob_names, calibrator, 1, "/tmp")

    while True:
        outputs = dumper.next()
        if outputs is None:
            break
        print([a.shape for a in outputs])

    image_filter.dispose()
    dumper.dispose()



