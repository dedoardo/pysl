import math
import os
import sys
import ast
from . import pysl

g_out = None

# Core
# ------------------------------------------------------------------------------


def init(path: str) -> bool:
    try:
        global g_out
        os.makedirs(os.path.dirname(path), exist_ok=True)
        g_out = open(path, 'w')
    except IOError as e:
        sys.stderr.write("Failed to open file: {0} with error: {1}\n".format(path, e))
        return False
    return True


def finalize():
    if g_out:
        g_out.close()


def write(string: str):
    g_out.write(string)


def TYPE(type: pysl.Type):
    return type.str


def OFFSET_TO_CONSTANT(offset: int):
    coff = 'c' + str(math.floor(offset / 4))
    if offset % 4 == 1:
        coff += '.y'
    elif offset % 4 == 2:
        coff += '.z'
    elif offset % 4 == 3:
        coff += '.w'
    return coff


def declaration(declaration: pysl.Declaration):
    for qualifier in declaration.qualifiers:
        write('{0} '.format(qualifier))
    write('{0} {1}'.format(declaration.type, declaration.name))


# Top-level
# ----------


def options(options: [str]):
    write('#if defined(__INTELLISENSE__)\n')
    for opt in options:
        write('#\tdefine {0}\n'.format(opt))
    write('#endif\n\n')


def struct(struct: pysl.Struct):
    write('struct {0}\n{{\n'.format(struct.name))
    for element in struct.elements:
        write('\t{0} {1};\n'.format(TYPE(element[0]), element[1]))
    write('};\n\n')


def stage_input(si: pysl.StageInput, prev_sis: [pysl.StageInput]):
    write('struct {0}\n{{\n'.format(si.name))
    for element in si.elements:
        for cond in element.conditions:
            write('{0}\n'.format(cond))
        write('\t{0} {1} : {2};\n'.format(TYPE(element.type), element.name, element.semantic))
    for cond in si.post_conditions:
        write('{0}\n'.format(cond))
    write('};\n\n')


def entry_point_beg(func: pysl.Function, sin: pysl.StageInput, sout: pysl.StageInput):
    write('{0} {1}('.format(sout.name, func.name))
    write('{0} {1}'.format(sin.name, pysl.Language.SpecialAttribute.INPUT))
    write(')\n{\n')
    write('\t{0} {1};\n'.format(sout.name, pysl.Language.SpecialAttribute.OUTPUT))


def entry_point_end(func: pysl.Function):
    write('\treturn {0};\n'.format(pysl.Language.SpecialAttribute.OUTPUT))
    write('}\n\n')


def constant_buffer(cbuffer: pysl.ConstantBuffer):
    write('cbuffer {0}\n{{\n'.format(cbuffer.name))
    for constant in cbuffer.constants:
        write('\t{0} {1}_{2} : packoffset({3});\n'.format(TYPE(constant.type), cbuffer.name, constant.name, OFFSET_TO_CONSTANT(constant.offset)))

    if cbuffer.enforced_size is not None:
        write('\tfloat _{0}_padding : packoffset({1});\n'.format(cbuffer.name, OFFSET_TO_CONSTANT(cbuffer.enforced_size - 1)))

    write('};\n\n')


def TEXTURE_NAME_FROM_SAMPLER(sampler_name: str) -> str:
    return sampler_name + '__tex'


def sampler(sampler: pysl.Sampler):
    texture_type_map = [
        'Texture1D',
        'Texture2D',
        'Texture3D',
        'TextureCube',
        'Texture1DArray',
        'Texture2DArray',
        'Texture2DMS',
        'Texture2DArray',
        'TextureCubeArray',
        'Texture1D',
        'Texture2D',
        'Texture1DArray',
        'Texture2DArray',
        'TextureCube',
        'TextureCubeArray',
    ]
    texture_type = texture_type_map[pysl.Language.Sampler.TYPES.index('Sampler' + sampler.type)]
    write('{0} {1} : register(t{2});\n'.format(texture_type, TEXTURE_NAME_FROM_SAMPLER(sampler.name), sampler.slot))
    write('SamplerState {0} : register(s{1});\n\n'.format(sampler.name, sampler.slot))


# Block-level
# ------------------------------------------------------------------------------


def arg_sep():
    write(', ')


def arg_end():
    write(')')


def _sampler_sample(sampler: pysl.Sampler, args):
    is_shadow = 'Shadow' in sampler.type
    is_bias = len(args) >= 2
    is_offset = len(args) >= 3

    uv = args[0]
    bias_cmp = args[1] if is_bias else None
    offset = args[2] if is_offset else None
    if is_shadow:
        write('{0}.SampleCmp({1}, '.format(TEXTURE_NAME_FROM_SAMPLER(sampler.name), sampler.name))
        uv()
        arg_sep()
        bias_cmp()
        if is_offset:
            arg_sep()
            offset()
        arg_end()
    else:
        write('{0}.SampleBias({1}, '.format(TEXTURE_NAME_FROM_SAMPLER(sampler.name), sampler.name))
        uv()
        if is_bias:
            arg_sep()
            bias_cmp()
        else:
            write('0.f, ')
        if is_offset:
            arg_sep()
            offset()
        arg_end()


def _merge_uv_mip(sampler_type: str, uv_arg, mip_arg) -> str:
    # Multisample textures have no miplevel
    if 'MS' in sampler_type:
        uv_arg()
        arg_end()
        return
    elif sampler_type == '1D':
        write('int2(')
    elif sampler_type in ['1DArray', '2D']:
        write('int3(')
    elif sampler_type in ['2DArray', '3D']:
        write('int4(')
    uv_arg()
    arg_sep()
    mip_arg()
    arg_end()


def _sampler_load(sampler: pysl.Sampler, args):
    is_multisample = 'MS' in sampler.type

    texel_coord = args[0]
    miplevel = args[1]
    offset_sample = args[2] if len(args) >= 3 else None

    write('{0}.Load('.format(TEXTURE_NAME_FROM_SAMPLER(sampler.name)))
    _merge_uv_mip(sampler.type, texel_coord, miplevel)
    if not is_multisample:
        arg_sep()
        write('0')
    arg_sep()
    offset_sample()
    arg_end()


def _sampler_sample_grad(sampler: pysl.Sampler, args):
    uv = args[0]
    ddx = args[1]
    ddy = args[2]

    write('{0}.SampleGrad({1}, '.format(TEXTURE_NAME_FROM_SAMPLER(sampler.name), sampler.name))
    uv()
    arg_sep()
    ddx()
    arg_sep()
    ddy()
    arg_end()


def _sampler_sample_level(sampler: pysl.Sampler, args):
    uv = args[0]
    miplevel = args[1]

    write('{0}.SampleLevel({1}, '.format(TEXTURE_NAME_FROM_SAMPLER(sampler.name), sampler.name))
    uv()
    arg_sep()
    miplevel()
    arg_end()


def _sampler_gather(sampler: pysl.Sampler, args):
    is_shadow = 'Shadow' in sampler.type
    is_offset = len(args) >= 3

    uv = args[0]
    channel_cmp = args[1]
    offset = args[2] if is_offset else None

    if is_shadow:
        write('{0}.GatherCmp({1}, '.format(TEXTURE_NAME_FROM_SAMPLER(sampler.name), sampler.name))
        uv()
        arg_sep()
        channel_cmp()
        arg_sep()
        if is_offset:
            offset()
        else:
            write('0')
        arg_end()
    else:
        # Small trick to avoid
        if not isinstance(channel_cmp.node, ast.Num):
            print("Gather() operations <channel> has to be a number literal")
            return

        idx = int(channel_cmp.node.n)
        if idx < 0 or idx > 3:
            print("Gather() <channel> has to be in the range [0, 3]")
            return

        ext = ['Red', 'Green', 'Blue', 'Alpha'][idx]
        write('{0}.Gather{1}({2}, '.format(TEXTURE_NAME_FROM_SAMPLER(sampler.name), ext, sampler.name))
        uv()
        if is_offset:
            arg_sep()
            offset()
        arg_end()


def method_call(caller: pysl.Object, method: str, args):
    if isinstance(caller, pysl.Sampler):
        if method == pysl.Language.Sampler.Method.SAMPLE:
            _sampler_sample(caller, args)
        elif method == pysl.Language.Sampler.Method.LOAD:
            _sampler_load(caller, args)
        elif method == pysl.Language.Sampler.Method.SAMPLE_GRAD:
            _sampler_sample_grad(caller, args)
        elif method == pysl.Language.Sampler.Method.SAMPLE_LEVEL:
            _sampler_sample_level(caller, args)
        elif method == pysl.Language.Sampler.Method.GATHER:
            _sampler_gather(caller, args)


def member(caller: pysl.Object, value: str):
    if isinstance(caller, pysl.ConstantBuffer):
        write('{0}_{1}'.format(caller.name, value))


def _args(args):
    for i in range(len(args) - 1):
        args[i]()
        write(', ')
    if args:
        args[-1]()


def constructor(type: str, args):
    write('{0}('.format(type))
    _args(args)
    write(')')


def intrinsic(type: str, args):
    if type.startswith(pysl.Language.Intrinsic.COL):
        mat = args[0]
        row = args[1]

        comps = int(type[-1])
        write('float{0}('.format(comps))
        for comp in range(comps):
            mat()
            write('[{0}]['.format(comp))
            row()
            write(']{0}'.format(', ' if comp < comps - 1 else ''))
        write(')')
    elif type.startswith(pysl.Language.Intrinsic.ROW):
        comps = int(type[-1])
        args[0]()
        write('[')
        args[1]()
        write('].{0}'.format('xyzw'[:comps]))
    else:
        write('{0}('.format(type))
        _args(args)
        write(')')


def cast(type: str, val):
    write('({0})'.format(type))
    val()


def special_attribute(stage: str, si: pysl.StageInput, attribute: str, value: str):
    write('{0}.{1}'.format(attribute, value))