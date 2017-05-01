import pysl
import sys

# Error reporting
# ------------------------------------------------------------------------------


g_file: str = None
g_compilation_status: bool = None


def init(filename: str):
    global g_file, g_compilation_status
    g_file = filename
    g_compilation_status = True


def get_status() -> bool:
    return g_compilation_status


def error(loc: pysl.Locationable, msg: str):
    sys.stderr.write('pyslc: {0}:{1}:{2} {3}\n'.format(g_file,
                                                     loc.line, loc.col, msg))

    global g_compilation_status
    g_compilation_status = False
