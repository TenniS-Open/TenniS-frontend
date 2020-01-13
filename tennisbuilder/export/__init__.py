try:
    from . import nnie
except Exception as e:
    import sys
    sys.stderr.write("import nnie failed with: {}\n".format(e))

from . import fridge
