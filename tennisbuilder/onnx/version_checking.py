#!python

import onnx

####  bug: 现在最高"1.10.0"   下面这个判断会报错
if onnx.__version__ < "1.4.0":
    raise ImportError("Please upgrade your onnx installation to v1.4.* or later! Got v{}.".format(onnx.__version__))