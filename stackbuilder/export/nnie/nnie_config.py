import sys
if sys.version > "3":
    import str as basestring


class Config(object):
    def __init__(self):
        self.__prototxt_file = None
        self.__caffemodel_file = None
        self.__instructions_name = None
        self.__image_list = []
        self.__batch_num = None
        self.__net_type = 0
        self.__sparse_rate = 0
        self.__compile_mode = 0
        self.__is_simulation = 0
        self.__RGB_order = None
        self.__data_scale = None
        self.__image_type = 0
        self.__norm_type = 0
        self.__internal_stride = 16
        self.__log_level = 0
        self.__mean_file = None
        self.__compress_mode = None

    @property
    def prototxt_file(self):
        return self.__prototxt_file

    @prototxt_file.setter
    def prototxt_file(self, value):
        assert isinstance(value, basestring)
        self.__prototxt_file = value

    @property
    def caffemodel_file(self):
        return self.__caffemodel_file

    @caffemodel_file.setter
    def caffemodel_file(self, value):
        assert isinstance(value, basestring)
        self.__caffemodel_file = value

    @property
    def instructions_name(self):
        return self.__instructions_name

    @instructions_name.setter
    def instructions_name(self, value):
        assert isinstance(value, basestring)
        self.__instructions_name = value

    @property
    def image_list(self):
        return self.__image_list

    @image_list.setter
    def image_list(self, value):
        assert isinstance(value, (basestring, tuple, list))
        if isinstance(value, basestring):
            value = [value, ]

        if not all([isinstance(s, basestring) for s in value]):
            raise ValueError("param 1 must be str or list of str")

        self.__image_list = list(value)

    @property
    def batch_num(self):
        return self.__batch_num

    @batch_num.setter
    def batch_num(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert 0 <= value <= 256
        self.__batch_num = value

    @property
    def net_type(self):
        return self.__net_type

    @net_type.setter
    def net_type(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert value in [0, 1, 2]
        self.__net_type = value

    @property
    def sparse_rate(self):
        return self.__sparse_rate

    @sparse_rate.setter
    def sparse_rate(self, value):
        try:
            value = float(value)
        except:
            raise ValueError("param 1 must be float")
        assert 0 <= value <= 1
        self.__sparse_rate = value

    @property
    def compile_mode(self):
        return self.__compile_mode

    @compile_mode.setter
    def compile_mode(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert value in [0, 1, 2]
        self.__compile_mode = value

    @property
    def is_simulation(self):
        return self.__is_simulation

    @is_simulation.setter
    def is_simulation(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert value in [0, 1]
        self.__is_simulation = value

    @property
    def RGB_order(self):
        return self.__RGB_order

    @RGB_order.setter
    def RGB_order(self, value):
        assert isinstance(value, basestring)
        assert value in {"RGB", "BGR"}
        self.__RGB_order = value

    @property
    def data_scale(self):
        return self.__data_scale

    @data_scale.setter
    def data_scale(self, value):
        try:
            value = float(value)
        except:
            raise ValueError("param 1 must be float")
        self.__data_scale = value

    @property
    def image_type(self):
        return self.__image_type

    @image_type.setter
    def image_type(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert value in {0, 1, 3, 5}
        self.__image_type = value

    @property
    def norm_type(self):
        return self.__norm_type

    @norm_type.setter
    def norm_type(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert value in {0, 1, 2, 3, 4, 5}
        self.__norm_type = value

    @property
    def log_level(self):
        return self.__log_level

    @log_level.setter
    def log_level(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert value in {0, 1, 2, 3}
        self.__log_level = value

    @property
    def internal_stride(self):
        return self.__internal_stride

    @internal_stride.setter
    def internal_stride(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert value in {16, 32}
        self.__internal_stride = value

    @property
    def mean_file(self):
        return self.__mean_file

    @mean_file.setter
    def mean_file(self, value):
        assert isinstance(value, basestring)
        self.__mean_file = value

    @property
    def compress_mode(self):
        return self.__compress_mode

    @compress_mode.setter
    def compress_mode(self, value):
        try:
            value = int(value)
        except:
            raise ValueError("param 1 must be int")
        assert value in {0, 1}
        self.__compress_mode = value

    def write(self, filepath):
        # type: (str) -> None
        if self.__prototxt_file is None:
            raise ValueError("prototxt_file must be set")
        if self.__caffemodel_file is None:
            raise ValueError("caffemodel_file must be set")
        with open(filepath, "w") as f:
            f.write("[net_type] {}\n".format(self.net_type))
            f.write("[is_simulation] {}\n".format(self.is_simulation))
            f.write("[compile_mode] {}\n".format(self.compile_mode))
            f.write("[log_level] {}\n".format(self.log_level))
            f.write("\n")

            if self.instructions_name is not None:
                f.write("[instructions_name] {}\n".format(self.instructions_name))
            f.write("[prototxt_file] {}\n".format(self.prototxt_file))
            f.write("[caffemodel_file] {}\n".format(self.caffemodel_file))
            f.write("\n")

            f.write("[image_type] {}\n".format(self.image_type))
            if self.RGB_order is not None:
                f.write("[RGB_order] {}\n".format(self.RGB_order))
            for s in self.__image_list:
                f.write("[image_list] {}\n".format(s))
            f.write("\n")

            f.write("[norm_type] {}\n".format(self.norm_type))
            if self.data_scale is not None:
                f.write("[data_scale] {}\n".format(self.data_scale))
            if self.mean_file is not None:
                f.write("[mean_file] {}\n".format(self.mean_file))
            f.write("\n")

            if self.batch_num is not None:
                f.write("[batch_num] {}\n".format(self.batch_num))
            if self.sparse_rate is not None:
                f.write("[sparse_rate] {}\n".format(self.sparse_rate))
            if self.compress_mode is not None:
                f.write("[compress_mode] {}\n".format(self.compress_mode))
            f.write("[internal_stride] {}\n".format(self.internal_stride))
            f.write("\n")

