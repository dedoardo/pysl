import os
import sys
import json
from . import pysl
from . import error


g_root: dict = None
g_json = None
g_cpp = None

CPP_HEADER = """#pragma once
#if defined(_MSC_VER) || defined(__GNUG_)
#   define PYSL_PACKED
#   pragma pack(push, 1)
#else
#   define PYSL_PACKED __attribute((packed))
#endif\n
"""

CPP_FOOTER = """
#if defined(_MSC_VER) || defined(__GNUG_)
#   pragma pack(pop)
#endif
"""


def init(json_path: str, cpp_path: str):
    global g_root, g_json, g_cpp
    if json_path:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        try:
            g_json = open(json_path, 'w')
        except IOError as e:
            sys.stderr.write("Failed to open file: {0} with error: {1}\n".format(json_path, e))
            return False

    if cpp_path:
        os.makedirs(os.path.dirname(cpp_path), exist_ok=True)
        try:
            g_cpp = open(cpp_path, 'w')
        except IOError as e:
            sys.stderr.write("Failed to open file: {0} with error: {1}\n".format(cpp_path, e))
            return False

        g_cpp.write(CPP_HEADER)

    g_root = {}
    g_root[pysl.Language.Export.OPTIONS] = []
    g_root[pysl.Language.Export.CONSTANT_BUFFERS] = []
    g_root[pysl.Language.Export.SAMPLERS] = []
    return True


def finalize():
    if g_json:
        g_json.write(json.dumps(g_root, indent=4, separators=(',', ': ')))

    if g_cpp:
        g_cpp.write(CPP_FOOTER)


def options(options: [str]):
    g_root[pysl.Language.Export.OPTIONS] += options


def struct(struct: pysl.Struct):
    pass


def stage_input(si: pysl.StageInput):
    pass


def constant_buffer(cbuffer: pysl.ConstantBuffer):
    state = {}
    state[pysl.Language.Export.NAME] = cbuffer.name
    if cbuffer.enforced_size:
        state[pysl.Language.Export.SIZE] = cbuffer.enforced_size
    g_root[pysl.Language.Export.CONSTANT_BUFFERS].append(state)

    if g_cpp:
        g_cpp.write('struct {0}\n{{\n'.format(cbuffer.name))
        cur_offset = 0
        paddings = 0
        for constant in cbuffer.constants:
            if constant.offset:
                diff = constant.offset - cur_offset
                if diff < 0:  # Error in offset calculation
                    error(cbuffer, "Invalid offset for constant: {0} in ConstantBuffer: {1}".format(constant.name, cbuffer.name))
                elif diff > 0:
                    g_cpp.write('\tfloat __pysl_padding{0}[{1}];\n'.format(paddings, diff))
                    paddings += 1

            g_cpp.write('\t{0} {1}'.format(pysl.scalar_to_cpp(constant.type.type), constant.name))
            if constant.type.dim0 > 1:
                g_cpp.write('[{0}]'.format(constant.type.dim0))
            if constant.type.dim1 > 1:
                g_cpp.write('[{0}]'.format(constant.type.dim1))
            g_cpp.write(';\n')
            cur_offset += constant.type.dim0 * constant.type.dim1

        if cbuffer.enforced_size:
            diff = cbuffer.enforced_size - cur_offset
            if diff < 0:
                error(cbuffer, "Invalid enforced size in ConstantBuffer: {0}".format(cbuffer.name))
            elif diff > 0:
                g_cpp.write('\tfloat __pysl_padding{0}[{1}];\n'.format(paddings, diff))

        g_cpp.write('} PYSL_PACKED;\n')
        for constant in cbuffer.constants:
            if constant.offset:
                g_cpp.write('static_assert(offsetof({0}, {1}) == {2}, "Invalid offset"));\n'.format(cbuffer.name, constant.name, constant.offset * 4))
        g_cpp.write('static_assert(sizeof({0}) == {1}, "Invalid size");\n\n'.format(cbuffer.name, cbuffer.enforced_size * 4))


def sampler(sampler: pysl.Sampler):
    state = {}
    state[pysl.Language.Export.NAME] = sampler.name
    for key, val in sampler.attributes:
        state[key] = val
    g_root[pysl.Language.Export.SAMPLERS].append(state)


def entry_point(func: pysl.Function):
    if func.stage == pysl.Language.Decorator.VERTEX_SHADER:
        if pysl.Language.Decorator.VERTEX_SHADER in g_root:
            error(func, "Trying to export multiple VertexShader entry points, {0} is already registered".format(g_root[pysl.Language.Export.VERTEX_SHADER]))
            return
        g_root[pysl.Language.Export.VERTEX_SHADER] = func.name
    elif func.stage == pysl.Language.Decorator.PIXEL_SHADER:
        if pysl.Language.Decorator.PIXEL_SHADER in g_root:
            error(func, "Trying to export multiple PixelShader entry points, {0} is already registered".format(g_root[pysl.Language.Export.PIXEL_SHADER]))
            return
        g_root[pysl.Language.Export.PIXEL_SHADER] = func.name


def query_entry_point(stage: str) -> str:
    if stage == pysl.Language.Decorator.VERTEX_SHADER:
        return g_root[pysl.Language.Export.VERTEX_SHADER]
    elif stage == pysl.Language.Decorator.PIXEL_SHADER:
        return g_root[pysl.Language.Export.PIXEL_SHADER]
    return None
