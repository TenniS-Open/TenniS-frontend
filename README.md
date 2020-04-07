# Python API

The model garden. [See](./garden/README.md).

## Setup

```bash
# bind libtennis to setup, must be compiled first
python setup.py bind --lib=`pwd`/../lib/libtennis.so
# install in comman way
python setup.py install --prefix=/path/to/install
```

## Package

```bash
python setup.py bind --lib=`pwd`/../lib/libtennis.so
python setup.py sdist
```
Packaged file will placed in `<root>/dist`.
