import sys
import os
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import tennis as ts
from tennisbuilder.export import flops


def analysis_flops(filename, input_shape, freeze=True):
    with open(filename, "rb") as f:
        m = ts.Module.Load(f)
    val = flops.analysis(m.outputs, m.inputs, input_shape, freeze=freeze)
    return val


if __name__ == '__main__':
    def log_analysis_flops(filename, input_shape, freeze=True):
        val = analysis_flops(filename, input_shape, freeze)
        print("Load {}, GFLOPs: {:.3f}".format(filename, flops.format_gflops(val)))

    log_analysis_flops(r"F:\rt\models\retinaface_origin.tsm", [(1, 3, 320, 320)], freeze=True)
    log_analysis_flops(r"F:\rt\models\pfld_was_72928_boxsize_1.5_2.5_allbigshift_v2_1.tsm", [(1, 3, 112, 112)], freeze=True)
    log_analysis_flops(r"F:\rt\models\mobilefacenet-v1_arcface_lffd-plfd_0519.tsm", [(1, 3, 112, 112)], freeze=True)
    log_analysis_flops(r"F:\rt\models\model_arcface_2020-3-4.tsm", [(1, 3, 112, 112)], freeze=True)
    log_analysis_flops(r"F:\rt\_acc989_res18_suzhou_19k_1204_3.tsm", [(1, 224, 224, 3)], freeze=True)
    log_analysis_flops(r"F:\BaiduNetdiskDownload\yolov3.coco\yolov3.coco.tsm", [(1, 3, 416, 416)], freeze=True)