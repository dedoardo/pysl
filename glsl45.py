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

    _OUT.write('#version 450')
    return True

def write(string : str):
    _OUT.write(string)

def TYPE(type : pysl.Type):
    return type.str

def declaration(declaration : pysl.Declaration):
    for qualifier in declaration.qualifiers:
        write('{0} '.format(qualifier))
    write('{0} {1}'.format(declaration.type, declaration.name))

# Top-level
#-------------------------------------------------------------------------------
def options(options : [str]):
    pass

def struct(struct : pysl.Struct):
    pass

def stage_input(si : pysl.StageInput):
    pass

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