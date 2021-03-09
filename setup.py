#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup
from setuptools import Command
import shutil
import sys

running_root = os.getcwd()
setup_root = os.path.dirname(os.path.abspath(__file__))

ext_in_prefix = '' if sys.platform == 'win32' else 'lib'
ext_in_suffix = ext_out_suffix = '.so'
if sys.platform == 'win32':
    ext_in_suffix, ext_out_suffix = '.dll', '.pyd'
elif sys.platform == 'darwin':
    ext_in_suffix = '.dylib'

libtennis_name = "{}tennis{}".format(ext_in_prefix, ext_in_suffix)

setup_libtennis_name = "{}tennis_{}".format(ext_in_prefix, ext_in_suffix)
setup_libtennis_path = "tennis/backend/{}".format(setup_libtennis_name)


def find_packages(include=None):
    def impl(root, dir, packages):
        root_dir = os.path.join(root, dir)
        filenames = os.listdir(root_dir)
        for filename in filenames:
            fullpath = os.path.join(root_dir, filename)
            if os.path.isdir(fullpath):
                subdir = os.path.join(dir, filename)
                impl(root, subdir, packages)
            else:
                if filename == '__init__.py':
                    packages.append(dir)
    packages = []
    for top in include:
        impl(setup_root, top, packages)
    return packages


def _clean():
    # rm cache
    egg_info = "tennis.egg-info"
    if os.path.exists(egg_info):
        shutil.rmtree(egg_info)
    # working dir is setup.py dir
    # if os.path.exists(libtennis_path):
    #     os.remove(libtennis_path)


def find_proto(top):
    def impl(root, dir, packages):
        root_dir = os.path.join(root, dir)
        filenames = os.listdir(root_dir)
        if dir == ".":
            dir = ""
        for filename in filenames:
            fullpath = os.path.join(root_dir, filename)
            if os.path.isdir(fullpath):
                subdir = os.path.join(dir, filename)
                impl(root, subdir, packages)
            else:
                name, ext = os.path.splitext(filename)
                if ext == '.proto':
                    packages.append(os.path.join(dir, filename))
    packages = []
    impl(os.path.join(setup_root, top), ".", packages)
    return packages


protoc_command = None


def do_protoc(proto):
    global protoc_command
    if protoc_command is None:
        protoc_command = "protoc"
        print("[INFO] assume protoc=protoc")
    proto_file = os.path.abspath(proto)
    proto_root, filename = os.path.split(proto_file)

    print("protoc {} --python_out={}".format(proto_file, proto_root))
    exit_status = os.system("protoc {} --proto_path={} --python_out={}".format(proto_file, proto_root, proto_root))
    if exit_status != 0:
        raise Exception("Can not compile proto: {}, please compile it by hand.".format(proto_file))


def timestamp(filename):
    if not os.path.exists(filename):
        return 0
    statinfo = os.stat(filename)
    return statinfo.st_mtime


def update_proto(root):
    global protoc_command
    protos = find_proto(root)
    protoc_command = None
    for proto in protos:
        name, ext = os.path.splitext(proto)
        pb2_py = name + "_pb2.py"
        full_proto = os.path.join(root, proto)
        full_pb2_py = os.path.join(root, pb2_py)
        if timestamp(full_pb2_py) < timestamp(full_proto):
            do_protoc(full_proto)
    return protos


configure_proto = []


def configure():
    # Build proto
    global configure_proto
    configure_proto = update_proto("tennisbuilder")
    _clean()
    # Find libtennis
    # ext_path = "../lib/{}".format(libtennis_name)
    # Copy libtennis
    # if os.path.exists(ext_path):
    #     shutil.copyfile(ext_path, libtennis_path)
    # else:
    #     print('ERROR: Unable to find built cpp extension.\n'
    #           'Please build tennis using cmake.')
    #     sys.exit().


class bind(Command):
    description = "Bind backend library"

    user_options = [
        # The format is (long option, short option, description).
        ("lib=", None, "Root or path to {}".format(libtennis_name)),
    ]

    def initialize_options(self):
        """Set default values for options."""
        self.lib = "../lib"

    def finalize_options(self):
        """Post-process options."""
        # if self.lib:
        #     print("Bind lib: {}".format(self.lib))

    def run(self):
        """Run command."""
        if os.path.isabs(self.lib):
            setting_lib = self.lib
        else:
            setting_lib = os.path.join(running_root, self.lib)
        if os.path.isfile(setting_lib):
            pass
        elif os.path.isdir(setting_lib):
            setting_lib = os.path.join(setting_lib, libtennis_name)
        else:
            print("[ERROR] {} is not existed dir or file.".format(setting_lib))
            exit(2001)
        shutil.copyfile(setting_lib, setup_libtennis_path)
        print("[INFO] Succeed bind: {}".format(setup_libtennis_path))
        exit(0)


class clean(Command):
    description = "Clean egg-info and backend libraries"

    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        _clean()
        if os.path.exists(setup_libtennis_path):
            os.remove(setup_libtennis_path)
        exit(0)


cwd = os.getcwd()
os.chdir(setup_root)

configure()
setup(
    name='tennis',
    version='0.5.0',
    author='seetatech',
    author_email='kaizhou.li@seetatech.com',
    url='https://gitlab.seetatech.com/TenniS/TenniS',
    keywords='tennis, edge, neural network, inference',
    description='TenniS: Tensor based Edge Neural Network Inference System',
    long_description='''TenniS: Tensor based Edge Neural Network Inference System
    This is frontend of TenniS, provide tsm module import and export.''',

    packages=find_packages(include=["tennis", "tennisbuilder", "tennisfence"]),
    package_data={
        'tennis': ["backend/{}".format(setup_libtennis_name)],
        'tennisbuilder': configure_proto,
    },
    package_dir={
        'tennis': 'tennis',
        'tennisbuilder': 'tennisbuilder',
        'tennisfence': 'tennisfence',
    },

    cmdclass={
        "bind": bind,
        "clean": clean,
    },

    install_requires=[
        'numpy >= 1.14',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        # 'Operating System :: Ubuntu 16.04'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',  # Support lang
        'Programming Language :: Python :: 3',  # Python version
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)

os.chdir(cwd)
