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

def TYPE(type : pysl.Type):
    type_map = [
        'bool', 'bvec2', 'bvec3', 'bvec4',
        'int', 'ivec3', 'ivec3', 'ivec4',
        'uint', 'uvec2', 'uvec3', 'uvec4',
        'float', 'vec2', 'vec3', 'vec4',
        'mat2x2', 'mat2x3', 'mat2x4',
        'mat3x2', 'mat3x3', 'mat3x4',
        'mat4x2', 'mat4x3', 'mat4x4'
    ]
    return type_map[pysl.TYPES.index(type.str)]

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
        write('\t{0} {1};\n'.format(TYPE(element[0]), element[1]))
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
                write('layout(location={0}) out {1} {2};\n'.format(slot, TYPE(element.type), element.name))
            elif is_builtin:
                write('{0} {1} {2};\n'.format(dest_qualifier, TYPE(element.type), SEMANTIC(element.semantic)))
            else:
                write('layout (location={0}) {1} {2} {3};\n'.format(LOCATION(element, si, prev_sis, stage), dest_qualifier, TYPE(element.type), element.name))
        write('#endif\n\n')

def entry_point_beg(func : pysl.Function, sin : pysl.StageInput, sout : pysl.StageInput):
    pass

def entry_point_end(func : pysl.Function):
    pass

def constant_buffer(cbuffer : pysl.ConstantBuffer):
    pass

def sampler(sampler : pysl.Sampler):
    pass

# Block-level
#-------------------------------------------------------------------------------
def method_call(caller : pysl.Object, method : str, args):
    pass

def constructor(type : str, args):
    pass

def intrinsic(type : str, args):
    pass

def special_attribute(attribute : str):
    pass