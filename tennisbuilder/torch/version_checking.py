#!python

import torch

def version_tuple(s):
    def try_int(i):
        try:
            return int(i)
        except Exception as _:
            return 0
    return tuple(map(try_int, s.split('.')))
    
if version_tuple(torch.__version__) < (1, 0, 0):
    raise ImportError("Please upgrade your torch installation to v1.0.* or later! Got v{}.".format(torch.__version__))