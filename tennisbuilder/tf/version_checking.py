#!python

import tensorflow as tf

def version_tuple(s):
    def try_int(i):
        try:
            return int(i)
        except Exception as _:
            return 0
    return tuple(map(try_int, s.split('.')))

if version_tuple(tf.__version__) < (1, 10, 0):
    raise ImportError("Please upgrade your tensorflow installation to v1.4.* or later! Got v{}.".format(tf.__version__))