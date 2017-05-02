import sys
from . import pysl

# Error reporting
# ------------------------------------------------------------------------------


g_file: str = None
g_compilation_status: bool = None


def init(filename: str):
    global g_file, g_compilation_status
    g_file = filename
    g_compilation_status = True


def get_status() -> bool:
    if g_compilation_status:
        return g_compilation_status
    return False


def error(loc: pysl.Locationable, msg: str):
    if loc:
        sys.stderr.write('error: {0}:{1}:{2}: {3}\n'.format(g_file, loc.line, loc.col, msg))
    else:
        sys.stderr.write('error: {0}\n'.format(msg))

    global g_compilation_status
    g_compilation_status = False

