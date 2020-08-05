import os
import sys
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

RuntimeRoot = "/home/kier/git/TensorStack/lib/"
sys.path.append(RuntimeRoot)

from tennis.backend.api import *
import tennis

import cv2
import random


def build_test_module(path):
    """
    Build graph like c = CustomOp(a, b)
    With CustomOp(a, b) = alpha * a <binary> beta * b + gama
    :param path:
    :return:
    """
    a = tennis.menu.param("a")
    b = tennis.menu.param("b")
    c = tennis.menu.op("c", op_name="CustomOp", inputs=[a, b])
    c.set("alpha", 1)
    c.set("beta", 2)
    c.set("gama", 3)
    c.set("binary", "mul")

    print(c)

    module = tennis.Module()
    module.load(c)

    with open(path, "wb") as f:
        tennis.Module.Save(f, module=module)


class CustomOp(Operator):
    def __init__(self):
        self.alpha = 1
        self.beta = 1
        self.gama = 0
        self.register_binary = {
            "mul": lambda x, y: x * y,
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "div": lambda x, y: x / y,
        }
        self.binary = self.register_binary["mul"]

    def init(self, params, context):  # type: (OperatorParams, OperatorContext) -> None
        if "alpha" in params:
            self.alpha = params["alpha"].numpy
        if "beta" in params:
            self.beta = params["beta"].numpy
        if "gama" in params:
            self.gama = params["gama"].numpy
        binary = params["binary"].str  # this is string parameter
        assert binary in self.register_binary
        self.binary = self.register_binary[binary]

    def run(self, args, context):  # type: (List[Tensor], OperatorContext) -> Union[Tensor, List[Tensor]]
        assert len(args) == 2
        a = args[0].numpy
        b = args[1].numpy
        c = self.binary(self.alpha * a, self.beta * b) + self.gama
        return Tensor(c)

RegisterOperator(CustomOp, "cpu", "CustomOp")


if __name__ == "__main__":
    build_test_module("customop.tsm")

    device = Device()

    module = Module.Load("customop.tsm")

    workbench = Workbench.Load(module=module, device=device)

    a = 3
    b = 4

    workbench.input(0, a)
    workbench.input(1, b)

    workbench.run()

    c = workbench.output(0).numpy

    print("a={}, b={}, c={}".format(a, b, c))
