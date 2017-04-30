import pysl
import os

_OUT = None

# Core
#-------------------------------------------------------------------------------
def init(path : str) -> bool:
    try:
        global _OUT
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _OUT = open(path, 'w')
    except IOError as e:
        print("Failed to open file: {0} with error: {1}".format(path, e))
        return False

    return True

def write(string : str):
    _OUT.write(string)

def TYPE(type : str):
    type_map = [
        'bool', 'bvec2', 'bvec3', 'bvec4',
        'int', 'ivec3', 'ivec3', 'ivec4',
        'uint', 'uvec2', 'uvec3', 'uvec4',
        'float', 'vec2', 'vec3', 'vec4',
        'mat2x2', 'mat2x3', 'mat2x4',
        'mat3x2', 'mat3x3', 'mat3x4',
        'mat4x2', 'mat4x3', 'mat4x4'
    ]
    return type_map[pysl.TYPES.index(type)]

def OFFSET_TO_CONSTANT(offset : int):
    return offset * 4

def declaration(declaration : pysl.Declaration):
    for qualifier in declaration.qualifiers:
        write('{0} '.format(qualifier))
    write('{0} {1}'.format(declaration.type, declaration.name))

# Top-level
#-------------------------------------------------------------------------------
def options(options : [str]):
    pass

def struct(struct : pysl.Struct):
    write('struct {0}\n{{\n'.format(struct.name))
    for element in struct.elements:
        write('\t{0} {1};\n'.format(TYPE(element[0].str), element[1]))
    write('};\n\n')

def SEMANTIC(semantic : str) -> str:
    semantic_map = [
        'gl_Position'
        'gl_VertexID',
        'gl_InstanceID'
    ]
    return semantic_map[pysl.Keywords.Semantics.index(semantic)]

def LOCATION(element : pysl.InputElement, si : pysl.StageInput, prev_sis : [pysl.StageInput], stage : str) -> int:
    if stage[2:] == pysl.Keywords.Out:
        return si.elements.index(element)
    elif stage[:2] == pysl.Keywords.VertexShaderDecorator:
        return si.elements.index(element)
    else:
        # Linkings vs-ps
        prev_stage = pysl.Keywords.VertexShaderOutputDecorator
        for prev_si in prev_sis:
            if prev_stage in prev_si.stages:
                # Looking for the element with the matching semantic
                for prev_element in prev_si.elements:
                    if prev_element.semantic == element.semantic:
                        return prev_si.elements.index(prev_element)
                print("Unmatched semantic {0} in {1}".format(stage, si.name))
                return -1

def stage_input(si : pysl.StageInput, prev_sis : [pysl.StageInput]):
    for stage in si.stages:
        stage_idx = None
        if stage[:2] == pysl.Keywords.VertexShaderDecorator:
            write('#if defined(PYSL_VERTEX_SHADER)\n')
            stage_idx = 1
        elif stage[:2] == pysl.Keywords.PixelShaderDecorator:
            write('#if defined(PYSL_PIXEL_SHADER)\n')
            stage_idx = 2

        dest_qualifier = stage[2:]
        for element in si.elements:
            is_builtin = True if element.semantic[:3] == 'SV_' else False
            if element.semantic.startswith(pysl.Keywords.SVTargetSemantic):
                slot = int(element.semantic[(len(pysl.Keywords.SVTargetSemantic)):])
                write('layout(location={0}) out {1} __{2}_{3};\n'.format(slot, TYPE(element.type.str), dest_qualifier, element.name))
            elif is_builtin:
                write('{0} {1} __{2}_{3};\n'.format(dest_qualifier, TYPE(element.type.str), dest_qualifier, SEMANTIC(element.semantic)))
            else:
                write('layout (location={0}) {1} {2} __{3}_{4};\n'.format(LOCATION(element, si, prev_sis, stage), dest_qualifier, TYPE(element.type.str), dest_qualifier, element.name))
        write('#endif\n\n')

def entry_point_beg(func : pysl.Function, sin : pysl.StageInput, sout : pysl.StageInput):
    if func.stage == pysl.Keywords.VertexShaderDecorator:
        write('#if defined(PYSL_VERTEX_SHADER)\n')
    elif func.stage == pysl.Keywords.PixelShaderDecorator:
        write('#if defined(PYSL_PIXEL_SHADER)\n')
    write('void {0}()\n{{\n'.format(func.name))

def entry_point_end(func : pysl.Function):
    write('};\n')
    write('#endif\n\n')

def constant_buffer(cbuffer : pysl.ConstantBuffer):
    write('layout(std140) uniform {0}\n{{\n'.format(cbuffer.name))
    for constant in cbuffer.constants:
        write('\tlayout(offset = {0}) {1} {2};\n'.format(OFFSET_TO_CONSTANT(constant.offset), TYPE(constant.type.str), constant.name))
    if cbuffer.enforced_size is not None:
        write('\tlayout(offset = {0}) float __pysl_padding;\n'.format(OFFSET_TO_CONSTANT(cbuffer.enforced_size - 1)))
    write('};\n\n')

def sampler(sampler : pysl.Sampler):
    sampler_type = 'sampler' + sampler.type
    write('uniform {0} {1};\n'.format(sampler_type, sampler.name))

def arg_sep():
    write(', ')

def arg_end():
    write(')')

# Block-level
#-------------------------------------------------------------------------------
def _shadow_uv(sampler_type : str, uv_arg, cmp_arg) -> str:
    if sampler_type == '1DShadow':
        write('vec3(')
        uv_arg()
        write(', 0.f, ')
        cmp_arg()
        arg_end()
    elif sampler_type == '2DShadow':
        write('vec3(')
        uv_arg()
        arg_sep()
        cmp_arg()
        arg_end()
    elif sampler_type == '1DArrayShadow':
        write('vec3(')
        uv_arg()
        arg_sep()
        cmp_arg()
        arg_end()
    elif sampler_type == '2DArrayShadow':
        write('vec4(')
        uv_arg()
        arg_sep()
        cmp_arg()
        arg_end()

def _sampler_sample(sampler : pysl.Sampler, args):
    """ 
    texture(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/texture.xhtml
    textureOffset(): https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/textureOffset.xhtml
    """

    is_shadow = 'Shadow' in sampler.type
    is_offset = len(args) >= 3

    uv = args[0]
    bias_cmp = args[1]
    offset = args[2] if is_offset else None
    if is_offset:
        write('textureOffset({0}, '.format(sampler.name))
        if is_shadow:
            _shadow_uv(sampler.type, uv, bias_cmp)
            offset() # no bias
        else:
            uv()
            arg_sep()
            offset()
            arg_sep()
            bias_cmp()
        arg_end()
    else:
        write('texture({0}, '.format(sampler.name))    
        if is_shadow:
            _shadow_uv(sampler.type, uv, bias_cmp)
        else:
            uv()
            arg_sep()
            bias_cmp()
        arg_end()

def _sampler_load(sampler : pysl.Sampler, args):
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

def _sampler_sample_grad(sampler : pysl.Sampler, args):
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

def _sampler_sample_level(sampler : pysl.Sampler, args):
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

def _sampler_gather(sampler : pysl.Sampler, args):
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

def method_call(caller : pysl.Object, method : str, args):
    if isinstance(caller, pysl.Sampler):
        if method == pysl.Keywords.SamplerSampleMethod:
            _sampler_sample(caller, args)
        elif method == pysl.Keywords.SamplerLoadMethod:
            _sampler_load(caller, args)
        elif method == pysl.Keywords.SamplerSampleGradMethod:
            _sampler_sample_grad(caller, args)
        elif method == pysl.Keywords.SamplerSampleLevelMethod:
            _sampler_sample_level(caller, args)
        elif method == pysl.Keywords.SamplerGatherMethod:
            _sampler_gather(caller, args)

def _args(args):
    for i in range(len(args) - 1):
        args[i]()
        arg_sep()
    if args:
        args[-1]()

def constructor(type : str, args):
    write('{0}('.format(TYPE(type)))
    _args(args)
    write(')')

def intrinsic(type : str, args):
    write('{0}('.format(type))
    _args(args)
    write(')')

def special_attribute(attribute : str):
    if attribute == pysl.Keywords.InputValue:
        write('__in_')
    elif attribute == pysl.Keywords.OutputValue:
        write('__out_')