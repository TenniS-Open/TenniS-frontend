#!python
# coding: UTF-8
"""
author: kier
"""

from ..enum import *
from ..config import *
from ..darknet import Layer
from ..darknet import calloc
from ..darknet import SizeParams
from ..darknet import fprintf

import sys
import math


def make_output_layer(batch, h, w, c, output):
    # type(int, int, int, int, int) -> Layer
    l = Layer()
    l.type = OUTPUT
    l.batch = batch
    l.w = w
    l.h = h
    l.c = c
    l.out_w = w
    l.out_h = h
    l.out_c = c

    l.outputs = l.out_w*l.out_h*l.out_c
    l.inputs = l.w*l.h*l.c
    l.output = None # calloc(l.outputs*batch, sizeof(float))

    fprintf(sys.stderr, "output                  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",
            w, h, c, l.out_w, l.out_h, l.out_c)

    return l


def parse_output(options, params):
    output = option_find_int(options, "output", -1) # default output last layer, not used for now
    l = make_output_layer(params.batch, params.w, params.h, params.c, output)
    return l
