import pysl
import math
import os

_OUT = None

def init(path : str) -> bool:
    try:
        global _OUT
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _OUT = open(path, 'w')
    except IOError as e:
        print("Failed to open file: {0} with error: {1}".format(path, e))
        return False
    return True

def OUT(string : str):
    _OUT.write(string)

def TYPE(type : pysl.Type):
    return type.str

def OFFSET_TO_CONSTANT(offset : int):
    coff = 'c' + str(math.floor(offset / 4))
    if offset % 4 == 1:
        coff += '.y'
    elif offset % 4 == 2:
        coff += '.z'
    elif offset % 4 == 3:
        coff += '.w'
    return coff

def text(string : str):
    OUT(string)

def options(options : [str]):
    OUT('#if defined(__INTELLISENSE__)\n')
    for opt in options:
        OUT('#\tdefine {0}\n'.format(opt))
    OUT('#endif\n\n')

def struct(struct : pysl.Struct):
    OUT('struct {0}\n{{\n'.format(struct.name))
    for element in struct.elements:
        OUT('\t{0} {1};\n'.format(TYPE(element[0]), element[1]))
    OUT('};\n\n')

def stage_input(struct : pysl.StageInput):
    OUT('struct {0}\n{{\n'.format(struct.name))
    for element in struct.elements:
        for cond in element.conditions:
            OUT('{0}\n'.format(cond))
        OUT('\t{0} {1} : {2};\n'.format(TYPE(element.type), element.name, element.semantic))
    for cond in struct.post_conditions:
        OUT('{0}\n'.format(cond))
    OUT('};\n\n')

def declaration(type : str, name : str):
    OUT('{0} {1}'.format(type, name))

def entry_point_beg(func : pysl.Function, sin : pysl.StageInput, sout : pysl.StageInput):
    OUT('{0} {1}('.format(sout.name, func.name))
    OUT('{0} {1}'.format(sin.name, pysl.Keywords.InputValue))
    OUT(')\n{\n')
    OUT('\t{0} {1};\n'.format(sout.name, pysl.Keywords.OutputValue))

def entry_point_end(func : pysl.Function):
    OUT('\treturn {0};\n'.format(pysl.Keywords.OutputValue))
    OUT('};\n\n')

def constant_buffer(cbuffer : pysl.ConstantBuffer):
    OUT('cbuffer {0}\n{{\n'.format(cbuffer.name))
    for constant in cbuffer.constants:
        OUT('\t{0} {1}'.format(TYPE(constant.type), constant.name))
        if constant.offset is not None:
            OUT(' : packoffset({0})'.format(OFFSET_TO_CONSTANT(constant.offset)))
        OUT(';\n')

    if cbuffer.enforced_size is not None:
        OUT('\tfloat __pysl_padding : packoffset({0});\n'.format(OFFSET_TO_CONSTANT(cbuffer.enforced_size - 1)))

    OUT('};\n\n')

def sampler_state(sampler_state : pysl.SamplerState):
    OUT('SamplerState {0};\n'.format(sampler_state.name))

def texture(texture : pysl.Texture):
    OUT('Texture{0} {1};\n'.format(texture.type, texture.name))

def _args(args):
    for i in range(len(args) - 1):
        args[i]()
        OUT(', ')
    if args:
        args[-1]()

def method_call(caller : pysl.Object, method : str, args):
    if caller:
        OUT('{0}.{1}('.format(caller.name, method))
        _args(args)
        OUT(')')

def constructor(ctype : str, args):
    OUT('{0}('.format(ctype))
    _args(args)
    OUT(')')

def intrinsic(itype : str, args):
    OUT('{0}('.format(itype))
    _args(args)
    OUT(')')

def special_attribute(attribute : str):
    OUT(attribute)