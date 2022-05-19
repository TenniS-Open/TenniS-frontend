#!python

import onnx

def version_tuple(s):
    def try_int(i):
        try:
            return int(i)
        except Exception as _:
            return 0
    return tuple(map(try_int, s.split('.')))

if version_tuple(onnx.__version__) < (1, 4, 0):
    raise ImportError("Please upgrade your onnx installation to v1.4.* or later! Got v{}.".format(onnx.__version__))