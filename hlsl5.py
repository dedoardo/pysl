import pysl
import math
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

def declaration(declaration : pysl.Declaration):
    for qualifier in declaration.qualifiers:
        write('{0} '.format(qualifier))
    write('{0} {1}'.format(declaration.type, declaration.name))

# Top-level
#-------------------------------------------------------------------------------
def options(options : [str]):
    write('#if defined(__INTELLISENSE__)\n')
    for opt in options:
        write('#\tdefine {0}\n'.format(opt))
    write('#endif\n\n')

def struct(struct : pysl.Struct):
    write('struct {0}\n{{\n'.format(struct.name))
    for element in struct.elements:
        write('\t{0} {1};\n'.format(TYPE(element[0]), element[1]))
    write('};\n\n')

def stage_input(si : pysl.StageInput, prev_sis : [pysl.StageInput]):
    write('struct {0}\n{{\n'.format(si.name))
    for element in si.elements:
        for cond in element.conditions:
            write('{0}\n'.format(cond))
        write('\t{0} {1} : {2};\n'.format(TYPE(element.type), element.name, element.semantic))
    for cond in si.post_conditions:
        write('{0}\n'.format(cond))
    write('};\n\n')

def entry_point_beg(func : pysl.Function, sin : pysl.StageInput, sout : pysl.StageInput):
    write('{0} {1}('.format(sout.name, func.name))
    write('{0} {1}'.format(sin.name, pysl.Keywords.InputValue))
    write(')\n{\n')
    write('\t{0} {1};\n'.format(sout.name, pysl.Keywords.OutputValue))

def entry_point_end(func : pysl.Function):
    write('\treturn {0};\n'.format(pysl.Keywords.OutputValue))
    write('};\n\n')

def constant_buffer(cbuffer : pysl.ConstantBuffer):
    write('cbuffer {0}\n{{\n'.format(cbuffer.name))
    for constant in cbuffer.constants:
        write('\t{0} {1}'.format(TYPE(constant.type), constant.name))
        if constant.offset is not None:
            write(' : packoffset({0})'.format(OFFSET_TO_CONSTANT(constant.offset)))
        write(';\n')

    if cbuffer.enforced_size is not None:
        write('\tfloat __pysl_padding : packoffset({0});\n'.format(OFFSET_TO_CONSTANT(cbuffer.enforced_size - 1)))

    write('};\n\n')

def TEXTURE_NAME_FROM_SAMPLER(sampler_name : str) -> str:
    return sampler_name + '__tex'

def sampler(sampler : pysl.Sampler):
    texture_type = sampler.type
    write('Texture{0} {1} : register(t{2});\n'.format(texture_type, TEXTURE_NAME_FROM_SAMPLER(sampler.name), sampler.slot))
    write('SamplerState {0} : register(s{1});\n\n'.format(sampler.name, sampler.slot))

# Block-level
#-------------------------------------------------------------------------------
def _args(args):
    for i in range(len(args) - 1):
        args[i]()
        write(', ')
    if args:
        args[-1]()

def method_call(caller : pysl.Object, method : str, args):
    if caller:
        if isinstance(caller, pysl.Sampler):
            write('{0}.{1}({2}, '.format(TEXTURE_NAME_FROM_SAMPLER(caller.name), method, caller.name))
            _args(args)
            write(')')

def constructor(type : str, args):
    write('{0}('.format(type))
    _args(args)
    write(')')

def intrinsic(type : str, args):
    write('{0}('.format(type))
    _args(args)
    write(')')

def special_attribute(attribute : str):
    write(attribute)