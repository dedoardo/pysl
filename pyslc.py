import ast
import sys
import hlsl5
import glsl45
import pysl
import json
import copy
import argparse
import os

# Helpers
#-------------------------------------------------------------------------------
def pretty_node(node : ast.AST) -> str:
    return str(type(node))[13:-2] # <class '_ast.NAME'>

# Comment INFO() out for no verbos
def INFO(node : ast.AST, msg : str = ''):
    if not node:
        print('INFO {0}'.format(msg))
    else:
        print('INFO[{0}::{1}:{2}] {3}'.format(pretty_node(node), node.lineno, node.col_offset, msg))

def ERR(node : ast.AST, msg : str = ''):
    if not node:
        print('ERR {0}'.format(msg))
    else:
        print('ERR[{0}::{1}:{2}] {3}'.format(pretty_node(node), node.lineno, node.col_offset, msg))

class Assignment(pysl.Declaration):
    def __init__(self):
        pysl.Declaration.__init__(self)
        self.value : str = None

# Translation routines
# Note that args is not a string it's a closure. This is because the translation
# model involves a single pass and nodes are translated on a DFS fashion, but 
# at the same time we need to give the backend control on the translation. Instead
# of going back and forth or more complex approaches, we simply delay the evaluation
# of the node until really needed.
#-------------------------------------------------------------------------------
class Metadata:
    _JSON = None
    _ROOT = None
    _CPP  = None

    @staticmethod
    def init(metadata : str, cpp  : str):
        if metadata:
            os.makedirs(os.path.dirname(metadata), exist_ok=True)
            try:
                Metadata._JSON = open(metadata, 'w')
            except IOError as e:
                print("Failed to open file: {0} with error: {1}".format(metadata + '.json', e))
                return False
            Metadata._ROOT = { }
            Metadata._ROOT[pysl.Keywords.SamplerStatesKey] = []
            Metadata._ROOT[pysl.Keywords.TexturesKey] = []
            Metadata._ROOT[pysl.Keywords.ConstantBuffersKey] = []
            Metadata._ROOT[pysl.Keywords.OptionsKey] = []

        if cpp:
            os.makedirs(os.path.dirname(cpp), exist_ok=True)
            Metadata._CPP = open(cpp, 'w')
            Metadata._CPP.write('#pragma once\n')
            Metadata._CPP.write('#pragma pack(push, 1)\n')

        return True

    @staticmethod
    def finalize():
        if Metadata._ROOT:
            Metadata._JSON.write(json.dumps(Metadata._ROOT, indent = 4, separators=(',', ': ')))
       
        if Metadata._CPP:
            Metadata._CPP.write('#pragma pack(pop)\n')
            Metadata._CPP.close()

    @staticmethod
    def entry_point(stage : str, name : str):
        if not Metadata._ROOT:
            return

        if stage == pysl.Keywords.VertexShaderDecorator:
            if pysl.Keywords.VertexShaderKey in Metadata._ROOT:
                ERR(None, "Trying to register multiple vertex shaders")
            else:
                Metadata._ROOT[pysl.Keywords.VertexShaderKey] = name
        elif stage == pysl.Keywords.PixelShaderDecorator:
            if pysl.Keywords.PixelShaderKey in Metadata._ROOT:
                ERR(None, "Trying to register multiple pixel shaders")
            else:
                Metadata._ROOT[pysl.Keywords.PixelShaderKey] = name
        else:
            ERR(None, "Invalid function stage: {0}".format(stage))

    @staticmethod
    def constant_buffer_attrs(cbuffer : pysl.ConstantBuffer):
        if Metadata._ROOT:
            cb = { }
            cb[pysl.Keywords.NameKey] = cbuffer.name

            if cbuffer.enforced_size:
                cb[pysl.Keywords.SizeKey] = cbuffer.enforced_size

            Metadata._ROOT[pysl.Keywords.ConstantBuffersKey].append(cb)

        if not Metadata._CPP:
            return

        Metadata._CPP.write('struct {0}\n{{\n'.format(cbuffer.name))
        cur_offset = 0
        paddings = 0
        for constant in cbuffer.constants:
            if constant.offset:
                diff = constant.offset - cur_offset
                if diff < 0: # Error in offset calculation
                    ERR(None, "Invalid offset for constant: {0} in ConstantBuffer: {1}".format(constant.name, cbuffer.name))
                elif diff > 0:
                    Metadata._CPP.write('\tfloat __pysl_padding{0}[{1}];\n'.format(paddings, diff))
                    paddings += 1

            Metadata._CPP.write('\t{0} {1}'.format(pysl.scalar_to_cpp(constant.type.type), constant.name))
            if constant.type.dim0 > 1:
                Metadata._CPP.write('[{0}]'.format(constant.type.dim0))
            if constant.type.dim1 > 1:
                Metadata._CPP.write('[{0}]'.format(constant.type.dim1))
            Metadata._CPP.write(';\n')
            cur_offset += constant.type.dim0 * constant.type.dim1

        if cbuffer.enforced_size:
            diff = cbuffer.enforced_size - cur_offset
            if diff < 0:
                ERR(None, "Invalid enforced size in ConstantBuffer: {0}".format(cbuffer.name))
            elif diff > 0:
                Metadata._CPP.write('\tfloat __pysl_padding{0}[{1}];\n'.format(paddings, diff))

        Metadata._CPP.write('};\n')
        Metadata._CPP.write('static_assert(sizeof({0}) == {1}, "Invalid size");\n\n'.format(cbuffer.name, cbuffer.enforced_size * 4))
       
    @staticmethod
    def sampler_attrs(sampler : pysl.Sampler):
        if Metadata._ROOT:
            state = { }
            state[pysl.Keywords.NameKey] = sampler.name
            for key, val in sampler.attributes:
                state[key] = val
            Metadata._ROOT[pysl.Keywords.SamplerStatesKey].append(state)

    @staticmethod
    def options(options : [str]):
        if Metadata._ROOT:
            Metadata._ROOT[pysl.Keywords.OptionsKey] += options

class Translate:
    _HLSL : bool = False
    _GLSL : bool = False
    _SYMBOLS : dict = None

    @staticmethod
    def init(hlsl_path : str, glsl_path : str) -> bool:
        """Initializes the backends and the symbol table -> bool"""
        if hlsl_path:
            Translate._HLSL = hlsl5.init(hlsl_path)
            if not Translate._HLSL:
                return False
        if glsl_path:
            Translate._GLSL = glsl45.init(glsl_path)
            if not Translate._GLSL:
                return False
        Translate._SYMBOLS = { } 
        return True

    @staticmethod
    def text(string : str):
        """Writes text directly to the output such as preprocessor strings"""
        if Translate._HLSL:
            hlsl5.write(string)
        if Translate._GLSL:
            glsl45.write(string)

    # TOP-LEVEL
    #---------------------------------------------------------------------------
    @staticmethod
    def options(strings : [str]):
        """Adds a bunch of compilation options"""
        Metadata.options(strings)

        # HlslTools helper: https://github.com/tgjones/HlslTools
        if Translate._HLSL:
            hlsl5.options(strings)
        if Translate._GLSL:
            glsl45.options(strings)

    @staticmethod
    def decl_struct(struct : pysl.Struct):
        if struct.name in Translate._SYMBOLS:
            ERR(None, "Already defined symbol: {0} as {1}".format(Translate._SYMBOLS, Translate._SYMBOLS[struct.name]))
            return

        if Translate._HLSL:
            hlsl5.struct(struct)
        if Translate._GLSL:
            glsl45.struct(struct)

        Translate._SYMBOLS[struct.name] = struct

    @staticmethod
    def decl_stage_input(si : pysl.StageInput):
        """Stage input declaration"""
        if si.name in Translate._SYMBOLS:
            ERR(None, "Already defined symbol: {0} as {1}".format(Translate._SYMBOLS, Translate._SYMBOLS[si.name]))
            return

        prev_sis = [v for k, v in Translate._SYMBOLS.items() if isinstance(v, pysl.StageInput)]
        if Translate._HLSL:
            hlsl5.stage_input(si, prev_sis)
        if Translate._GLSL:
            glsl45.stage_input(si, prev_sis)

        Translate._SYMBOLS[si.name] = si

    @staticmethod
    def constant_buffer(cbuffer : pysl.ConstantBuffer):
        """Constant buffer declaration"""
        if cbuffer.name in Translate._SYMBOLS:
            ERR(None, "Already defined symbol: {0} as {1}".format(Translate._SYMBOLS, Translate._SYMBOLS[cbuffer.name]))
            return

        Metadata.constant_buffer_attrs(cbuffer)
        if Translate._HLSL:
            hlsl5.constant_buffer(cbuffer)
        if Translate._GLSL:
            glsl45.constant_buffer(cbuffer)

        Translate._SYMBOLS[cbuffer.name] = cbuffer

    @staticmethod
    def sampler(sampler : pysl.Sampler):
        """Sampler state declaration"""
        if sampler.name in Translate._SYMBOLS:
            ERR(None, "Already defined symbol: {0} as {1}".format(Translate._SYMBOLS, Translate._SYMBOLS[sampler.name]))
            return

        Metadata.sampler_attrs(sampler)
        if Translate._HLSL:
            hlsl5.sampler(sampler)
        if Translate._GLSL:
            glsl45.sampler(sampler)

        Translate._SYMBOLS[sampler.name] = sampler

    @staticmethod
    def _parameter(arg : pysl.Declaration):
        """Writes a function parameter """
        if (arg.type not in pysl.TYPES and(
            arg.type not in Translate._SYMBOLS or(
            not isinstance(Translate._SYMBOLS[arg.type], pysl.StageInput) and
            not isinstance(Translate._SYMBOLS[arg.type], pysl.Struct)
            ))):
            ERR(None, "Type not found: {0}".format(arg.type))
            return

        if Translate._HLSL:
            hlsl5.declaration(arg)
        if Translate._GLSL:
            glsl45.declaration(arg)

    @staticmethod
    def function_beg(func : pysl.Function):
        if func.stage:
            func_in = None
            func_out = None
            # Special case, looking up input
            for name, obj in Translate._SYMBOLS.items():
                if isinstance(obj, pysl.StageInput):
                    for stage in obj.stages:
                        if func.stage + pysl.Keywords.In == stage:
                            if func_in:
                                ERR(None, "Multiple possible input values found for entry point: {0}".format(func.name))
                            func_in = obj
                        if func.stage + pysl.Keywords.Out == stage:
                            if func_out:
                                ERR(None, "Multiple possible output values found for entry point: {0}".format(func.name))
                            func_out = obj

            if func_in == None or func_out == None:
                ERR(None, "Undeclared input or output for function stage: {0}:{1}".format(func.name, func.stage))
                return
            if Translate._HLSL:
                hlsl5.entry_point_beg(func, func_in, func_out)
            if Translate._GLSL:
                glsl45.entry_point_beg(func, func_in, func_out)
        else:
            # Standard C-like function declaration
            Translate.text('{0} {1}('.format(func.return_value, func.name))
            for arg in func.args:
                Translate._parameter(arg)
            Translate.text(')\n{\n')
            Translate._SYMBOLS[func.name] = func

    @staticmethod
    def function_end(func : pysl.Function):
        if func.stage:
            if Translate._HLSL:
                hlsl5.entry_point_end(func)
            if Translate._GLSL:
                glsl45.entry_point_end(func)
        else:
            Translate.text('};\n\n')

    # BLOCK-LEVEL
    #---------------------------------------------------------------------------
    @staticmethod
    def declaration(assignment : Assignment):
        """Plain old declaration, usually a ast.AnnAssign"""

        # Type is either a scalar basic type
        # or a StageInput. No other type is allowed at the block level
        if (assignment.type not in pysl.TYPES and(
            assignment.type not in Translate._SYMBOLS or(
            not isinstance(Translate._SYMBOLS[assignment.type], pysl.StageInput) and
            not isinstance(Translate._SYMBOLS[assignment.type], pysl.Struct)
            ))):
            ERR(None, "Type not found: {0}".format(assignment.type))
            return

        if Translate._HLSL:
            hlsl5.declaration(assignment)
        if Translate._GLSL:
            glsl45.declaration(assignment)

    @staticmethod
    def method_call(caller : str, name : str, args):
        """Method calls are used simply as a stylistic way to expose intrinsics""" 

        obj = None
        
        # Right now method calls are supported exclusively by textures, thus
        # the caller has to be registered in the symbol table
        if caller not in Translate._SYMBOLS or not isinstance(Translate._SYMBOLS[caller], pysl.Sampler):
            ERR(None, "Expected sampler object in method call: {0}".format(caller))
            return
        obj = Translate._SYMBOLS[caller]

        if Translate._HLSL:
            hlsl5.method_call(obj, name, args)
        if Translate._GLSL:
            glsl45.method_call(obj, name, args)

    @staticmethod
    def constructor(typename : str, args):
        """Type constructor, assuming that typename is in pysl.TYPES""" 
        if Translate._HLSL:
            hlsl5.constructor(typename, args)
        if Translate._GLSL:
            glsl45.constructor(typename, args)

    @staticmethod
    def intrinsic(type : str, args):
        """Intrinsic function, assuming that itype is in pysl.INTRINSICS"""
        if Translate._HLSL:
            hlsl5.intrinsic(type, args)
        if Translate._GLSL:
            glsl45.intrinsic(type, args)

    @staticmethod
    def function_call(function : str, args):
        """Function call encountered, same for both backends"""
        if function not in Translate._SYMBOLS or not isinstance(Translate._SYMBOLS[function], pysl.Function):
            ERR(None, "Unrecognized function name: {0}".format(function))
            return

        Translate.text('{0}('.format(function))
        for i in range(len(args) - 1):
            args[i]()
            Translate.text(', ')
        if args:
            args[-1]()
        Translate.text(')')

    @staticmethod
    def special_attribute(attribute : str):
        if Translate._HLSL:
            hlsl5.special_attribute(attribute)
        if Translate._GLSL:
            glsl45.special_attribute(attribute)

# AST Utilities
#-------------------------------------------------------------------------------
def str_to_pysl_type(string : str) -> pysl.Type:
    ret = pysl.Type()
    ret.str = string 

    dims = ''
    if string[:4] == 'bool':
        ret.type = pysl.Scalar.BOOL
        dims = string[4:]
    elif string[:3] == 'int':
        ret.type = pysl.Scalar.INT
        dims = string[3:]
    elif string[:4] == 'uint':
        ret.type = pysl.Scalar.UINT 
        dims = string[4:]
    elif string[:5] == 'float':
        ret.type = pysl.Scalar.FLOAT
        dims = string[5:]
    else:
        ERR(None, "Invalid type: {0}".format(string))
    if dims:
        ret.dim0 = int(dims[0:1])
    if dims[1:2] == 'x':
        ret.dim1 = int(dims[2:3])
    return ret

class Decorator:
    def __init__(self):
        self.name : str = None
        self.args : [str] = []

def eval_numeric_constexpr(node : ast.AST):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return +eval_numeric_constexpr(node.operand)
        elif isinstance(node.op, ast.USub):
            return -eval_numeric_constexpr(node.operand)
        else:
            return None
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return eval_numeric_constexpr(node.left) + eval_numeric_constexpr(node.right)
        if isinstance(node.op, ast.Sub):
            return eval_numeric_constexpr(node.left) - eval_numeric_constexpr(node.right)
        if isinstance(node.op, ast.Mult):
            return eval_numeric_constexpr(node.left) * eval_numeric_constexpr(node.right)
        if isinstance(node.op, ast.Div):
            return eval_numeric_constexpr(node.left) / eval_numeric_constexpr(node.right)
        return None

def parse_decorator(node : ast.AST):
    if isinstance(node, ast.Name):
        ret = Decorator()
        ret.name = node.id
        return ret
    elif isinstance(node, ast.Call):
        ret = Decorator()
        ret.name = node.func.id
        for arg in node.args:
            if isinstance(arg, ast.Num):
                ret.args.append(str(arg.n))
            elif isinstance(arg, ast.Str):
                ret.args.append(str(arg.n))
            elif isinstance(arg, ast.Name):
                ret.args.append(arg.id)
            else:
                v = eval_numeric_constexpr(arg)
                if v:
                    ret.args.append(str(v))
                else:
                    ERR(arg, "Unsupported decorator type")
        return ret
    else:
        ERR(node, "Supported decorators are Name and Call")
        return None

def parse_preprocessor(call : ast.Call) -> str:
    if not isinstance(call, ast.Call):
        ERR(call, "Preprocessor takes the form of _('directive')")
        return ''
    if len(call.args) != 1:
        ERR(call, "Preprocessor takes 1 argument in the of of _('directive')")
        return ''
    if not isinstance(call.args[0], ast.Str):
        ERR(call, "Preprocessor requires arguments to be strings in the form of _('directive')")
    return call.args[0].s

def parse_options(call : ast.Call) -> [str]:
    if not isinstance(call, ast.Call):
        ERR(call, "Option takes the form of option('option')")
        return ''

    ret = []
    for arg in call.args:
        if not isinstance(arg, ast.Str):
            ERR(arg, "Expected string literal")
            continue
        ret.append(arg.s)
    return ret

def parse_attribute_no_eval(node : ast.Attribute) -> str:
    if isinstance(node.value, ast.Name):
        return node.value.id
    return parse_attribute_no_eval(node.value) + '.|{0}|'.format(node.attr)

def parse_assignment(node : ast.AST) -> Assignment:
    """Block level assignment, can either be a declaration or not"""
    if not isinstance(node, ast.Assign) and not isinstance(node, ast.AnnAssign):
        ERR(node, "Expected Assignment")
        return

    ret = Assignment()
    ret.qualifiers = []
    if isinstance(node, ast.AnnAssign):
        # Declaration
        ret.name = node.target.id
        if isinstance(node.annotation, ast.Name):
            ret.type = node.annotation.id
        elif isinstance(node.annotation, ast.Attribute) and isinstance(node.annotation.value, ast.Name):
            if node.annotation.value.id == pysl.Keywords.ConstQualifier:
                ret.qualifiers.append(pysl.Keywords.ConstQualifier)
            else:
                ERR(node.annotation, "Unsupported qualifier, only const is supported in a block assignment/declaration")
            ret.type = node.annotation.attr

        else:
            ERR(node.annotation, "Unsupported annotation, supported are <type> or const.<type>")
    else:
        # Assignment
        if isinstance(node.targets[0], ast.Name):
            ret.name = node.targets[0].id
        elif isinstance(node.targets[0], ast.Attribute):
            ret.name = parse_attribute_no_eval(node.targets[0])
        else:
            ERR(node, "Unsupported assignment target")
    if node.value and isinstance(node.value, ast.Name):
        ret.value = node.value.id
    elif node.value and isinstance(node.value, ast.Num):
        ret.value = str(node.value.n)
    else:
        ret.value = None
    return ret

def TO_INT(node : ast) -> (bool, int):
    if isinstance(node, ast.Num):
        return (True, node.n)
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Num):
        if isinstance(node.op, ast.UAdd):
            return (True, +node.operand.n)
        if isinstance(node.op, ast.USub):
            return (True, -node.operand.n)
        ERR(node, "Expected +/- Num")
        return (False, 0)
    ERR(node, "Expected signed integer")
    return (False, 0)

def parse_for_range(node : ast.Call) -> (str, str, str):
    if len(node.args) != 3:
        ERR(node, "Expected 3 integer arguments in range(start, end, stop) found: {0}".format(len(node.args)))
        return ('ERR', 'ERR', 'ERR')
    
    if isinstance(node.args[0], ast.Name):
        ok1, v1 = True, node.args[0].id
    else:
        ok1, v1 = TO_INT(node.args[0])

    if isinstance(node.args[1], ast.Name):
        ok2, v2 = True, node.args[1].id
    else:
        ok2, v2 = TO_INT(node.args[1])   

    if isinstance(node.args[2], ast.Name):
        ok3, v3 = True, node.args[2].id
    else:
        ok3, v3 = TO_INT(node.args[2])

    if ok1 and ok2 and ok3:
        return (str(v1), str(v2), str(v3))
    return ('ERR', 'ERR', 'ERR')

def unop_str(op : ast.AST) -> str:
    if isinstance(op, ast.UAdd):
        return '+'
    if isinstance(op, ast.USub):
        return '-'
    if isinstance(op, ast.Not):
        return '!'
    if isinstance(op, ast.Invert):
        return '~'
    ERR(op, "Invalid unary operator encountered: {0}:{1}. Check supported intrinsics.".format(op.lineno, op.col_offset))
    return 'INVALID_UNOP'

def binop_str(op : ast.AST) -> str:
    if isinstance(op, ast.Add):
        return '+'
    if isinstance(op, ast.Sub):
        return '-'
    if isinstance(op, ast.Mult):
        return '*'
    if isinstance(op, ast.Div):
        return '/ '
    if isinstance(op, ast.Mod):
        return '%'
    if isinstance(op, ast.LShift):
        return '<<'
    if isinstance(op, ast.RShift):
        return '>>'
    if isinstance(op, ast.BitOr):
        return '|'
    if isinstance(op, ast.BitXor):
        return '^'
    if isinstance(op, ast.BitAnd):
        return '&'
    if isinstance(op, ast.MatMult):
        return '@'
    ERR(op, "Invalid binary operator encountered: {0}:{1}. Check supported intrinsics.".format(op.lineno, op.col_offset))
    return 'INVALID_BINOP'

def cmpop_str(op : ast.AST) -> str:
    if isinstance(op, ast.Eq):
        return '=='
    if isinstance(op, ast.NotEq):
        return '!='
    if isinstance(op, ast.Lt):
        return '<'
    if isinstance(op, ast.LtE):
        return '<='
    if isinstance(op, ast.Gt):
        return '>'
    if isinstance(op, ast.GtE):
        return '>='
    ERR(op, "Invalid compare operator encountered: {0}:{1}. Check supported intrisics.".format(op.lineno, op.col_offset))
    return 'INVALID_CMPOP'

# Parsing declarations
#-------------------------------------------------------------------------------
def parse_struct(node : ast.ClassDef) -> pysl.Struct:
    struct = pysl.Struct()
    struct.name = node.name
    struct.elements = []

    for decl_node in node.body:
        if isinstance(decl_node, ast.AnnAssign):
            assignment = parse_assignment(decl_node)
            struct.elements.append((str_to_pysl_type(assignment.type), assignment.name))
        else:
            ERR(decl_node, "Unrecognized node inside structure: {0}".format(struct.name))

    return struct


def parse_stage_input(node : ast.ClassDef, stages : str) -> pysl.StageInput:
    struct = pysl.StageInput()
    struct.name = node.name
    struct.elements = []
    struct.stages = stages

    conditions = []
    for decl_node in node.body:
        if isinstance(decl_node, ast.AnnAssign):
            assignment = parse_assignment(decl_node)
            element = pysl.InputElement()
            element.name = assignment.name
            element.type = str_to_pysl_type(assignment.type)
            element.semantic = assignment.value
            element.conditions = list(conditions)
            conditions[:] = [] #Copy
            struct.elements.append(element)

        elif isinstance(decl_node, ast.Expr) and isinstance(decl_node.value, ast.Call):
            if decl_node.value.func.id is '_':
                conditions.append(parse_preprocessor(decl_node.value))
            else:
                ERR(decl_node, "Unsupported function call: {0} inside StageInput: {1}".format(decl_node.value.func.id, struct.name))
        else:
            ERR(decl_node, "Unrecognized node inside StageInput: {0}".format(struct.name))

    struct.post_conditions = list(conditions)
    return struct

def parse_constant_buffer(node : ast.ClassDef) -> pysl.ConstantBuffer:
    cbuffer = pysl.ConstantBuffer()
    cbuffer.name = node.name
    cbuffer.constants = []

    for decl_node in node.body:
        if isinstance(decl_node, ast.AnnAssign):
            assignment = parse_assignment(decl_node)
            constant = pysl.Constant()
            constant.name = assignment.name
            if assignment.value:
                try:
                    constant.offset = int(assignment.value)
                except ValueError:
                    ERR(decl_node, "Expected numberic offset as argument")
            assignment.value
            constant.type = str_to_pysl_type(assignment.type)
            cbuffer.constants.append(constant)
        else:
            ERR(decl_node, "Unrecognized node inside ConstantBuffer: {0}".format(struct.name))

    for d in node.decorator_list:
        decorator = parse_decorator(d)
        if decorator.args:
            try:
                cbuffer.enforced_size = int(decorator.args[0])
            except ValueError:
                ERR(d, "Expected integer argument to constructor indicating enforced size(in constants), but evaluated: {0}".format(decorator.args[0]))
    return cbuffer

def parse_sampler(name : str, type : str, slot : int, value : ast.AST) -> pysl.Sampler:
    sampler = pysl.Sampler()
    sampler.name = name
    sampler.type = type
    sampler.slot = slot

    if value:
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or not value.func.id == pysl.Keywords.Export:
            ERR(value, "Expected export() expression")
        else:
            for kw in value.keywords:
                val = None
                if isinstance(kw.value, ast.Name):
                    val = kw.value.id
                elif isinstance(kw.value, ast.Num):
                    val = kw.value.n
                elif isinstance(kw.value, ast.Str):
                    val = kw.value.s
                else:
                    ERR(kw, "Unsupported export value, only literals or names are currently supported")
                    continue
                sampler.attributes.append((kw.arg, val))

    return sampler

# Evalution code
#-------------------------------------------------------------------------------
# Creates a new evalution closure for delayed evaluation
# <3 http://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture-in-python
def PYSL_eval_closure(node):
    return lambda : PYSL_eval(node) 

def PYSL_eval(node : ast.AST):
    """Core routine, doesn't return anything but directly writes to output"""
    if not node:
        return
    elif isinstance(node, ast.Call):
        # Calls can be of different types
        # - Method call, this is true IFF the function itself is an attribute.
        #   as custom objects are not part of the language, it is probably going to be  a
        if isinstance(node.func, ast.Attribute):
            Translate.method_call(node.func.value.id, node.func.attr, [PYSL_eval_closure(n) for n in node.args])
        # - Intrinsics, that is any kind of built in function 
        elif node.func.id in pysl.INTRINSICS:
            Translate.intrinsic(node.func.id, [PYSL_eval_closure(n) for n in node.args])
        # - Constructor, just a new type being declared
        elif node.func.id in pysl.TYPES:
            Translate.constructor(node.func.id, [PYSL_eval_closure(n) for n in node.args])
        # - '_' is the special code for the preprocessor, writes the string contained in the braces as it is
        elif node.func.id is '_':
            Translate.text(parse_preprocessor(node) + '\n\n')
        # - As we didnt' recognize the name, it is probably a user's routine
        else: 
            Translate.function_call(node.func.id, [PYSL_eval_closure(n) for n in node.args])

    elif isinstance(node, ast.Attribute):
        # node.attr is the attribute name (string), recursively evaluating the value
        if isinstance(node.value, ast.Name) and (node.value.id == pysl.Keywords.InputValue or node.value.id == pysl.Keywords.OutputValue):
            Translate.special_attribute(node.value.id)
        else:
            PYSL_eval(node.value)
            Translate.text('.')

        Translate.text('{0}'.format(node.attr))

    elif isinstance(node, ast.IfExp):
        # Ternary if
        PYSL_eval(node.test)
        Translate.text(' ? ')
        PYSL_eval(node.body)
        Translate.text(' : ')
        PYSL_eval(node.orelse)

    elif isinstance(node, ast.UnaryOp):
        Translate.text(unop_str(node.op))
        Translate.text('(')
        PYSL_eval(node.operand)
        Translate.text(')')

    elif isinstance(node, ast.BinOp):
        # Checking if it's a cast operation
        op_str = binop_str(node.op)
        if op_str == '@':
            if not isinstance(node.left, ast.Name):
                ERR(node.left, "Expected type to the left of the cast operator");
            elif node.left.id not in pysl.TYPES:
                ERR(node.left, "Invalid destination type: {0}".format(node.left.id))
            else:
                Translate.text('(')
                PYSL_eval(node.left)
                Translate.text(')')
                PYSL_eval(node.right)
        else:
            PYSL_eval(node.left)
            Translate.text(' {0} '.format(op_str))
            PYSL_eval(node.right)

    elif isinstance(node, ast.BoolOp):
        if len(node.values) > 2:
            ERR(node, "Unsupported multiple consecutive boolean expressions")
            return
        Translate.text('(')
        PYSL_eval(node.values[0])
        Translate.text(')')
        if isinstance(node.op, ast.And):
            Translate.text(' && ')
        if isinstance(node.op, ast.Or):
            Translate.text(' || ')
        Translate.text('(')
        PYSL_eval(node.values[1])
        Translate.text(')')

    elif isinstance(node, ast.Set):
        Translate.text('{ ')
        for i in range(len(node.elts)):
            PYSL_eval(node.elts[i])
            if i != len(node.elts) -1:
                Translate.text(', ')
        Translate.text(' }')

    elif isinstance(node, ast.Subscript):
        PYSL_eval(node.value)
        Translate.text('[')
        PYSL_eval(node.slice)
        Translate.text(']')

    elif isinstance(node, ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            ERR(node, "Unsupported multiple comparison operators")
            return
        PYSL_eval(node.left)
        Translate.text(cmpop_str(node.ops[0]))
        PYSL_eval(node.comparators[0])
    
    elif isinstance(node, ast.Num):
        Translate.text(str(node.n))

    elif isinstance(node, ast.Index):
        PYSL_eval(node.value)

    elif isinstance(node, ast.Str):
        Translate.text(node.s)

    elif isinstance(node, ast.Name):
        Translate.text(node.id)

    elif isinstance(node, ast.Expr):
        PYSL_eval(node.value)

    else:
        print(node)
        ERR(node, "PYSL : Unsupported expression")

def INDENT(indent : int):
    Translate.text('\t' * indent)

def PYSL_block(nodes : [ast.AST], indent : int):
    """Evaluates line by line all the instructions"""
    for node in nodes:
        if isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Assign):
            INDENT(indent)
            PYSL_eval(node.targets[0])
            Translate.text(' = ')
            PYSL_eval(node.value)
            Translate.text(';\n')

        elif isinstance(node, ast.AnnAssign):
            INDENT(indent)
            Translate.declaration(parse_assignment(node))
            if node.value:
                Translate.text(' = ')
                PYSL_eval(node.value)
            Translate.text(';\n')

        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if node.value.func.id is '_':
                    Translate.text(parse_preprocessor(node.value) + '\n')
            else:
                ERR(node, "Unsupported block expression")

        elif isinstance(node, ast.AugAssign):
            INDENT(indent)
            PYSL_eval(node.target)
            Translate.text(' {0}= '.format(binop_str(node.op)))
            PYSL_eval(node.value)
            Translate.text(';\n')

        elif isinstance(node, ast.Return):
            INDENT(indent)
            Translate.text('return ')
            PYSL_eval(node.value)
            Translate.text(';\n')
        elif isinstance(node, ast.If):
            INDENT(indent)
            Translate.text('if (')
            PYSL_eval(node.test)
            Translate.text(')\n')
            INDENT(indent)
            Translate.text('{\n')
            PYSL_block(node.body, indent + 1)
            INDENT(indent)
            Translate.text('}\n')
            if node.orelse:
                INDENT(indent)
                Translate.text('else\n')
                INDENT(indent)
                Translate.text('{\n')
                PYSL_block(node.orelse, indent + 1)
                INDENT(indent)
                Translate.text('}\n')

        elif isinstance(node, ast.For):
            iters = []
            # Currently only support range-based loops
            # for i in range(start, end, step) <- signed
            # for (i : uint = start; i [<>] end; step)
            # if step is < 0 -> compare is > and viceversa
            if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call):
                val = node.iter
                if val.func.id != 'range' and val.func.id != 'rrange':
                    ERR(val, "Currently only range-based loops are supported. range(start, end, step).")
                limits = parse_for_range(val)
                iters.append((node.target.id, limits[0], limits[1], limits[2], False if val.func.id == 'range' else True))
                
            elif isinstance(node.target, ast.Tuple):
                if isinstance(node.iter, ast.Tuple) and len(node.target.elts) == len(node.iter.elts):
                    for i in range(len(node.target.elts)):
                        target = node.target.elts[i]
                        vals = node.iter.elts[i]
                        if not isinstance(target, ast.Name) or not isinstance(vals, ast.Call):
                            ERR(node, "Expected Name = Call in {0}-th assignment".format(i))
                            continue
                        if vals.func.id != 'range' and vals.func.id != 'rrange':
                            ERR(vals, "Currently only range-based loops are supported. range(start, end, step)")
                        limits = parse_for_range(vals)
                        iters.append((target.id, limits[0], limits[1], limits[2], False if vals.func.id == 'range' else True))

                else:
                    ERR(node, "Expected tuple of same length in iter")
            else:
                ERR(node, "Expected same number of Name, Tuple associations")

            INDENT(indent)
            Translate.text('for(')
            # Assignment
            for i in iters[:-1]:
                Translate.text('int {0} = {1}, '.format(i[0], i[1]))
            if iters:
                Translate.text('int {0} = {1};'.format(iters[-1][0], iters[-1][1]))

            # Bounds
            for i in iters[:-1]:
                op = '<' if i[4] == False else '>'
                Translate.text('{0} {1} {2}, '.format(i[0], op, i[2]))
            if iters:
                op = '<' if iters[-1][4] == False else '>'
                Translate.text('{0} {1} {2};'.format(iters[-1][0], op, iters[-1][2]))

            # Increments
            for i in iters[:-1]:
                op = '+' if i[4] == False else '-'
                Translate.text('{0} {1}= {2}, '.format(i[0], op, i[3]))
            if iters:
                op = '+' if iters[-1][4] == False else '-'
                Translate.text('{0} {1}= {2})\n'.format(iters[-1][0], op, iters[-1][3]))

            INDENT(indent)
            Translate.text('{\n')
            PYSL_block(node.body, indent + 1)
            INDENT(indent)
            Translate.text('}\n')

        elif isinstance(node, ast.While):
            INDENT(indent)
            Translate.text('while(')
            PYSL_eval(node.test)
            Translate.text(')\n')
            INDENT(indent)
            Translate.text('{\n')
            PYSL_block(node.body, indent + 1)
            INDENT(indent)
            Translate.text('}\n')

        elif isinstance(node, ast.Continue):
            INDENT(indent)
            Translate.text('continue;\n')
        
        elif isinstance(node, ast.Break):
            INDENT(indent)
            Translate.text('break;\n')

        else:
            ERR(node, "Unsupported block expression")

def PYSL_arg(func : ast.FunctionDef, arg : ast.arg) -> pysl.Declaration:
    if isinstance(arg.annotation, ast.Name):
        return pysl.Declaration(arg.annotation.id, arg.arg, [])
    elif isinstance(arg.annotation, ast.Attribute) and isinstance(arg.annotation.value, ast.Name):
        if arg.annotation.value.id == pysl.Keywords.OutQualifier:
            return pysl.Declaration(arg.annotation.attr, arg.arg, [pysl.Keywords.OutQualifier])
    else:
        ERR(arg, "Expected type for parameter: {0} of function: {1}".format(arg.arg, func.name))
        return pysl.Declaration(None, None, None)

def PYSL_function_signature(func : ast.FunctionDef):
    ret = pysl.Function()
    ret.name = func.name
    ret.args = []

    if len(func.decorator_list) > 2:
        ERR(func, "Multiple decorators are not supported, only the first one will be processed")

    # Entry point or user-defined function ?
    if func.decorator_list:
        if isinstance(func.decorator_list[0], ast.Name):
            decorator = func.decorator_list[0].id
            if decorator in pysl.Keywords.FunctionDecorators:
                ret.stage = decorator
            else:
                ERR(func.decorator_list[0], "Unknown decorator: {0}".format(func.decorator_list[0].id))             
        else:
            ERR(func.decorator_list, "Expected name as decorator")

    if isinstance(func.returns, ast.Name) and ret.stage:
        ERR(func, "superfluous return type for entry point (already known): {0}".format(func.name))
    elif isinstance(func.returns, ast.Name):
        ret.return_value = func.returns.id

    # Parsing arguments
    for arg in func.args.args[:-1]:
        ret.args.append(PYSL_arg(func, arg))
    if func.args.args:
        ret.args.append(PYSL_arg(func, func.args.args[-1]))

    Translate.function_beg(ret)
    PYSL_block(func.body, 1)
    Translate.function_end(ret)

def PYSL_tl_decl(node : ast.AnnAssign):
    """Parses a specific top-level declaration"""
    if not isinstance(node.annotation, ast.Call) or (
        node.annotation.func.id != 'register' or 
        len(node.annotation.args) != 2 or
        not isinstance(node.annotation.args[0], ast.Name) or
        not isinstance(node.annotation.args[1], ast.Num)):
        ERR(node, "Invalid top level resource declaration, see docs. <name> : register(<type>, <slot>) = (...)")

    res_name = node.target.id
    res_type = node.annotation.args[0].id
    res_slot = node.annotation.args[1].n

    if res_type in pysl.Keywords.SamplerTypes:
        Translate.sampler(parse_sampler(res_name, res_type[7:], res_slot, node.value))
    else:
        ERR(node, "Unrecognized top-level resource declaration {0} : {1}".format(res_name, res_type))    

def PYSL(path : str):
    """Finds all the top-level nodes"""
    try:
        with open(path, 'r') as file:
            text = file.read()
    except IOError as e:
        print("Failed to open '{0}' error: {1}".format(path, e))
        return

    try:
        root = ast.parse(text)
    except SyntaxError as e:
        print("Failed to parse '{0}' error: {1}".format(path, e))
        return

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.ClassDef):
            # If no decorators, it is simply a user-defined structure
            if not node.decorator_list:
                Translate.decl_struct(parse_struct(node))
            else:
                decorator = parse_decorator(node.decorator_list[0])
                if decorator.name == pysl.Keywords.StageInputDecorator:
                    stages = []
                    for arg in decorator.args:
                        if arg in pysl.Keywords.StageInputDecorators:
                            stages.append(arg)
                        else:
                            ERR(decorator, "Unrecognized stageinput decorator: {0}".format(arg))
                    Translate.decl_stage_input(parse_stage_input(node, stages))
                elif decorator.name == pysl.Keywords.ConstantBufferDecorator:
                    Translate.constant_buffer(parse_constant_buffer(node))
                else:
                    ERR(node, "Unsupported decorator: {0}".format(decorator.name))
        elif isinstance(node, ast.AnnAssign):
            PYSL_tl_decl(node)

        elif isinstance(node, ast.FunctionDef):
            PYSL_function_signature(node)

        elif isinstance(node, ast.Assign):
            ERR(node, "Encountered untyped assignment")
        
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if node.value.func.id == '_':
                    Translate.text(parse_preprocessor(node.value) + '\n\n')
                if node.value.func.id == 'options':
                    Translate.options(parse_options(node.value))
                else:
                    ERR(node, "Unsupported top-level function call")
            else:
                ERR(node, "Unsupported top-level expression")

        else:
            ERR(node, "Unsupported top-level expression")

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Please specify file to transpile")
        sys.exit()

    parser = argparse.ArgumentParser(description='PYthon Shading Language compiler')
    parser.add_argument('output')
    parser.add_argument('-ohlsl5', type=str, action='store', default=None, help="HLSL destination path")
    parser.add_argument('-oglsl45', type=str, action='store', default=None, help="GLSL destination path")
    parser.add_argument('-ojson', type=str, action='store', default=None, help="JSON metadata destination path")
    parser.add_argument('-ohpp', type=str, action='store', default=None, help="C++ header destination path")
    args = parser.parse_args()

    if not Translate.init(args.ohlsl5, args.oglsl45):
        print('Failed to open destination HLSL/GLSL file, check permissions or paths')
        sys.exit()

    if not Metadata.init(args.ojson, args.ohpp):
        print('Failed to open destination metadata file, check permissions or paths')
        sys.exit()
 
    PYSL(sys.argv[1])
    Metadata.finalize()