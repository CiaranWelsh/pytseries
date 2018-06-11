try:
    from . import _fastdtw
except ImportError:
    from . import fastdtw

major = 0
minor = 0
micro = 1

global __version__
__version__ = "{}.{}.{}".format(major, minor, micro)