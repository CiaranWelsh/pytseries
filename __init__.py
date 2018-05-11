import dtw, core, clust, inout
try:
    import _fastdtw
except ImportError:
    import fastdtw
