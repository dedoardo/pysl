import os
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
        print("Failed to open file: {0} with error: {1}".format(path, e))
        return False
    return True


def finalize():
    if g_out:
        g_out.close()


def write(string: str):
    g_out.write(string)


def TYPE(type: str):
    type_map = [
        'bool', 'bvec2', 'bvec3', 'bvec4',
        'int', 'ivec3', 'ivec3', 'ivec4',
        'uint', 'uvec2', 'uvec3', 'uvec4',
        'float', 'vec2', 'vec3', 'vec4',
        'mat2x2', 'mat2x3', 'mat2x4',
        'mat3x2', 'mat3x3', 'mat3x4',
        'mat4x2', 'mat4x3', 'mat4x4'
    ]
    return type_map[pysl.Language.NativeType._ALL.index(type)]


def OFFSET_TO_CONSTANT(offset: int):
    return offset * 4


def declaration(declaration: pysl.Declaration):
    for qualifier in declaration.qualifiers:
        write('{0} '.format(qualifier))
    write('{0} {1}'.format(declaration.type, declaration.name))


# Top-level
# ------------------------------------------------------------------------------


def options(options: [str]):
    pass


def struct(struct: pysl.Struct):
    write('struct {0}\n{{\n'.format(struct.name))
    for element in struct.elements:
        write('\t{0} {1};\n'.format(TYPE(element[0].str), element[1]))
    write('};\n\n')


def SEMANTIC(semantic: str, stage: str) -> str:
    if stage == pysl.Language.Decorator.PIXEL_SHADER and semantic == pysl.Language.Semantic.POSITION:
        return 'gl_FragCoord'

    semantic_map = [
        'gl_ClipDistance',
        'gl_CullDistance',
        'gl_FragDepth',
        'gl_InstanceID',
        'gl_FrontFacing',
        'gl_Position',
        'gl_PrimitiveID',
        'gl_SampleID',
        'gl_FragStencilRef',
        'gl_Target',
        'gl_VertexID'
    ]
    return semantic_map[pysl.Language.Semantic._ALL.index(semantic)]


def LOCATION(element: pysl.InputElement, si: pysl.StageInput, prev_sis: [pysl.StageInput], stage: str) -> int:
    if stage[2:] in [pysl.Language.Decorator.OUT, pysl.Language.Decorator.IN]:
        return si.elements.index(element)
    else:
        # Linkings vs-ps
        prev_stage = pysl.Language.Decorator.VERTEX_SHADER_OUT
        for prev_si in prev_sis:
            if prev_stage in prev_si.stages:
                # Looking for the element with the matching semantic
                for prev_element in prev_si.elements:
                    if prev_element.semantic == element.semantic:
                        return prev_si.elements.index(prev_element)
                print("Unmatched semantic {0} in {1}".format(stage, si.name))
                return -1


def stage_input(si: pysl.StageInput, prev_sis: [pysl.StageInput]):
    for stage in si.stages:
        cur_stage = stage[:2]
        if cur_stage == pysl.Language.Decorator.VERTEX_SHADER:
            write('#if defined(PYSL_VERTEX_SHADER)\n')
        elif cur_stage == pysl.Language.Decorator.PIXEL_SHADER:
            write('#if defined(PYSL_PIXEL_SHADER)\n')

        dest_qualifier = stage[2:]
        for element in si.elements:
            is_builtin = True if element.semantic[:3] == 'SV_' else False
            if element.semantic.startswith(pysl.Language.Semantic.TARGET):  # Only valid as output
                slot = int(element.semantic[(len(pysl.Language.Semantic.TARGET)):])
                write('layout(location={0}) out {1} __{2}_{3};\n'.format(slot, TYPE(element.type.str), dest_qualifier, element.name))
            elif is_builtin:
                write('{0} {1} {2};\n'.format(dest_qualifier, TYPE(element.type.str), SEMANTIC(element.semantic, cur_stage)))
            else:
                write('layout (location={0}) {1} {2} __{3}_{4};\n'.format(LOCATION(element, si, prev_sis, stage), dest_qualifier, TYPE(element.type.str), dest_qualifier, element.name))
        write('#endif\n\n')


def entry_point_beg(func: pysl.Function, sin: pysl.StageInput, sout: pysl.StageInput):
    if func.stage == pysl.Language.Decorator.VERTEX_SHADER:
        write('#if defined(PYSL_VERTEX_SHADER)\n')
    elif func.stage == pysl.Language.Decorator.PIXEL_SHADER:
        write('#if defined(PYSL_PIXEL_SHADER)\n')
    write('void {0}()\n{{\n'.format(func.name))


def entry_point_end(func: pysl.Function):
    write('};\n')
    write('#endif\n\n')


def constant_buffer(cbuffer: pysl.ConstantBuffer):
    write('layout(std140) uniform {0}\n{{\n'.format(cbuffer.name))
    for constant in cbuffer.constants:
        write('\tlayout(offset = {0}) {1} {2};\n'.format(OFFSET_TO_CONSTANT(constant.offset), TYPE(constant.type.str), constant.name))
    if cbuffer.enforced_size is not None:
        write('\tlayout(offset = {0}) float __pysl_padding;\n'.format(OFFSET_TO_CONSTANT(cbuffer.enforced_size - 1)))
    write('};\n\n')


def sampler(sampler: pysl.Sampler):
    sampler_type = 'sampler' + sampler.type
    write('uniform {0} {1};\n'.format(sampler_type, sampler.name))


# Block-level
# ------------------------------------------------------------------------------


def arg_sep():
    write(', ')


def arg_end():
    write(')')


def _shadow_uv(sampler_type: str, uv_arg, cmp_arg):
    if sampler_type == '1DShadow':
        # Special case, second parameter is ignored as by specification
        write('vec3(')
        uv_arg()
        write(', 0.f, ')
        cmp_arg()
        arg_end()
        return
    elif sampler_type == '2DShadow':
        write('vec3(')
    elif sampler_type == '1DArrayShadow':
        write('vec3(')
    elif sampler_type == '2DArrayShadow':
        write('vec4(')
    uv_arg()
    arg_sep()
    cmp_arg()
    arg_end()


def _sampler_sample(sampler: pysl.Sampler, args):
    """
    texture(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/texture.xhtml
    textureOffset(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/textureOffset.xhtml
    """

    is_shadow = 'Shadow' in sampler.type
    is_bias = len(args) >= 2
    is_offset = len(args) >= 3

    uv = args[0]
    bias_cmp = args[1] if is_bias else None
    offset = args[2] if is_offset else None
    if is_offset:
        write('textureOffset({0}, '.format(sampler.name))
        if is_shadow:
            _shadow_uv(sampler.type, uv, bias_cmp)
            offset()  # no bias
        else:
            uv()
            arg_sep()
            offset()
            if is_bias:
                arg_sep()
                bias_cmp()
        arg_end()
    else:
        write('texture({0}, '.format(sampler.name))
        if is_shadow:
            _shadow_uv(sampler.type, uv, bias_cmp)
        else:
            uv()
            if is_bias:
                arg_sep()
                bias_cmp()
        arg_end()


def _sampler_load(sampler: pysl.Sampler, args):
    """
    texelFetch(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/texelFetch.xhtml
    texelFetchOffset(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/texelFetchOffset.xhtml
    """

    is_multisample = 'MS' in sampler.type
    is_offset = False
    if len(args) >= 3 and not is_multisample:
        is_offset = True

    texel_coord = args[0]
    miplevel = args[1]
    offset_sample = args[2] if len(args) >= 3 else None

    if is_offset:
        write('texelFetchOffset({0}, '.format(sampler.name))
        texel_coord()
        arg_sep()
        miplevel()
        arg_sep()
        offset_sample()
        arg_end()
    else:
        write('texelFetch({0}, '.format(sampler.name))
        texel_coord()
        arg_sep()
        miplevel()
        if is_multisample:
            arg_sep()
            offset_sample()
        arg_end()


def _sampler_sample_grad(sampler: pysl.Sampler, args):
    """
    textureGrad(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/textureGrad.xhtml
    """
    uv = args[0]
    ddx = args[1]
    ddy = args[2]

    write('textureGrad({0}, '.format(sampler.name))
    uv()
    arg_sep()
    ddx()
    arg_sep()
    ddy()
    arg_end()


def _sampler_sample_level(sampler: pysl.Sampler, args):
    """
    textureLod(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/textureLod.xhtml
    """
    uv = args[0]
    miplevel = args[1]

    write('textureLod({0}, '.format(sampler.name))
    uv()
    arg_sep()
    miplevel()
    arg_end()


def _sampler_gather(sampler: pysl.Sampler, args):
    """
    textureGather(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/textureGather.xhtml
    textureGatherOffset(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/textureGatherOffset.xhtml
    """

    is_shadow = 'Shadow' in sampler.type
    is_offset = len(args) >= 3

    uv = args[0]
    channel_cmp = args[1]
    offset = args[2] if is_offset else None

    if is_offset:
        write('textureGatherOffset({0}, '.format(sampler.name))
        if is_shadow:
            uv()
            arg_sep()
            channel_cmp()
            arg_sep()
            offset()
        else:
            uv()
            arg_sep()
            offset()
            arg_sep()
            channel_cmp()
        arg_end()
    else:
        write('textureGather({0}, '.format(sampler.name))
        uv()
        arg_sep()
        channel_cmp()
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


def _args(args):
    for i in range(len(args) - 1):
        args[i]()
        arg_sep()
    if args:
        args[-1]()


def constructor(type: str, args):
    write('{0}('.format(TYPE(type)))
    _args(args)
    write(')')


def intrinsic(type: str, args):
    if type == pysl.Language.Intrinsic.MUL.name:
        args[1]()
        write(' * ')
        args[0]()
    elif type.startswith(pysl.Language.Intrinsic.COL):
        args[0]()
        write('[')
        args[1]()
        write(']')
    elif type.startswith(pysl.Language.Intrinsic.ROW):
        mat = args[0]
        row = args[1]

        comps = int(type[-1])
        write('vec{0}('.format(comps))
        for comp in range(comps):
            mat()
            write('[{0}]['.format(comp))
            row()
            write(']{0}'.format(', ' if comp < comps - 1 else ''))
        write(')')
    else:
        name = type
        if type == pysl.Language.Intrinsic.LERP.name:
            name = 'mix'

        write('{0}('.format(name))
        _args(args)
        write(')')


def special_attribute(stage: str, si: pysl.StageInput, attribute: str, value: str):
    # Checking if the attribute is special
    semantic = si.get_element(value).semantic
    is_builtin = True if semantic[:3] == 'SV_' else False

    if not semantic.startswith(pysl.Language.Semantic.TARGET) and is_builtin:
        write(SEMANTIC(semantic, stage))
    elif attribute == pysl.Language.SpecialAttribute.INPUT:
        write('__in_{0}'.format(value))
    elif attribute == pysl.Language.SpecialAttribute.OUTPUT:
        write('__out_{0}'.format(value))