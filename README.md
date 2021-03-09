# Python API

The model garden. [See](./garden/README.md).

## Setup

**step 1**: bind backend library to support backend operations
> If only use pure fronted code, no need to bind libtennis.so, skip step 1.
> Compile TenniS cpp project to get `libtennis.so` or `tennis.dll`

bind `TenniS` to setup, must be compiled first,
```bash
python setup.py bind --lib=<path>/libtennis.so
```
or bind `tennis.dll` in Windows.
```bash
python setup.py bind --lib=<path>/tennis.dll
```

**step 2**: install package to path env

install in default path,
```bash
python setup.py install
```
or install into given path.
```bash
python setup.py install --prefix=/path/to/install
```

## Package

Build all code and library to package.
```bash
python setup.py bind --lib=<path>/libtennis.so
python setup.py sdist
```
Packaged file will placed in `<root>/dist`.
