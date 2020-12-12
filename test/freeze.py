import sys
import os
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import tennis as ts
from tennisbuilder.export import fridge

if __name__ == '__main__':
    input_module = r"det.tsm"
    output_module = r"det.freeze.tsm"
    with open(input_module, "rb") as f:
        m = ts.Module.Load(f)

    outputs, inputs = fridge.freeze(m.outputs, m.inputs, [(1, 3, 448, 448)])
    m = ts.Module()
    m.load(outputs)
    m.sort_inputs(inputs)

    with open(output_module, "wb") as f:
        ts.Module.Save(f, m)
