import ast
import sys
import hlsl
import pysl
import json
import copy

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

class Assignment:
    def __init__(self):
        self.name : str = None
        self.type : str = None
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
    
    objects = { }

    @staticmethod
    def init(metadata_path : str):
        try:
            Metadata._JSON = open(metadata_path + '.json', 'w')
        except IOError as e:
            print("Failed to open file: {0} with error: {1}".format(metadata_path + '.json', e))
            return False
        Metadata._ROOT = { }
        Metadata._ROOT[pysl.Keywords.SamplerStatesKey] = []
        Metadata._ROOT[pysl.Keywords.TexturesKey] = []
        Metadata._ROOT[pysl.Keywords.ConstantBuffersKey] = []
        Metadata._ROOT[pysl.Keywords.OptionsKey] = []

        Metadata._CPP = open(metadata_path + '.hpp', 'w')
        Metadata._CPP.write('#pragma once\n')
        Metadata._CPP.write('#pragma pack(push, 1)\n')

        return True

    @staticmethod
    def finalize():
        Metadata._JSON.write(json.dumps(Metadata._ROOT, indent = 4, separators=(',', ': ')))
        Metadata._CPP.write('#pragma pack(pop)\n')
        Metadata._CPP.close()

    @staticmethod
    def entry_point(stage : str, name : str):
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
        cb = { }
        cb[pysl.Keywords.NameKey] = cbuffer.name

        if cbuffer.enforced_size:
            cb[pysl.Keywords.SizeKey] = cbuffer.enforced_size

        Metadata._ROOT[pysl.Keywords.ConstantBuffersKey].append(cb)
        Metadata.objects[cbuffer.name] = copy.deepcopy(cbuffer)

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
    def sampler_state_attrs(sampler_state : pysl.SamplerState):
        state = { }
        state[pysl.Keywords.NameKey] = sampler_state.name

        for key, val in sampler_state.attributes:
            state[key] = val

        Metadata._ROOT[pysl.Keywords.SamplerStatesKey].append(state)
        Metadata.objects[sampler_state.name] = copy.deepcopy(sampler_state)

    @staticmethod
    def texture_attrs(texture : pysl.Texture):
        t = { } 
        t[pysl.Keywords.NameKey] = texture.name

        for key, val in texture.attributes:
            t[key] = val

        Metadata._ROOT[pysl.Keywords.TexturesKey].append(t)
        Metadata.objects[texture.name] = copy.deepcopy(texture)

    @staticmethod
    def options(options : [str]):
        Metadata._ROOT[pysl.Keywords.OptionsKey] += options

    @staticmethod
    def find(name : str) -> pysl.Object:
        try:
            return Metadata.objects[name]
        except KeyError:
            ERR(None, "Failed to find object: {0}. Are you sure you declared it?".format(name))

class Translate:
    @staticmethod
    def init(hlsl_path : str, glsl_path : str):
        return hlsl.init(hlsl_path)

    @staticmethod
    def text(string : str):
        hlsl.OUT(string)

    @staticmethod
    def options(strings : [str]):
        Metadata.options(strings)
        hlsl.options(strings)

    @staticmethod
    def stage_input(struct : pysl.StageInput):
        hlsl.stage_input(struct)

    @staticmethod
    def declaration(assignment : Assignment):
        hlsl.declaration(assignment.type, assignment.name)

    # Declaration
    @staticmethod
    def constant_buffer(cbuffer : pysl.ConstantBuffer):
        Metadata.constant_buffer_attrs(cbuffer)
        hlsl.constant_buffer(cbuffer)

    # Declaration
    @staticmethod
    def sampler_state(sampler_state : pysl.SamplerState):
        Metadata.sampler_state_attrs(sampler_state)
        hlsl.sampler_state(sampler_state)

    # Declaration
    @staticmethod
    def texture(texture : pysl.Texture):
        Metadata.texture_attrs(texture)
        hlsl.texture(texture)

    @staticmethod
    def method_call(caller : str, name : str, args):
        obj : pysl.Object = Metadata.find(caller)
        if obj:
            hlsl.method_call(obj, name, args)
        else:
            ERR(None, "Failed to translate {0}.{1} as the object type couldn't be deduced.".format(caller, name))

    @staticmethod
    def constructor(ctype : str, args):
        hlsl.constructor(ctype, args)

    @staticmethod
    def intrinsic(itype : str, args):
        hlsl.intrinsic(itype, args)

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
            else:
                v = eval_numeric_constexpr(arg)
                if v:
                    ret.args.append(str(v))
                else:
                    ERR(arg, "Decorators support only literals")
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

# Either assignment 
def parse_assignment(node : ast.AST) -> Assignment:
    if not isinstance(node, ast.Assign) and not isinstance(node, ast.AnnAssign):
        ERR(node, "Expected Assignment")
        return

    ret = Assignment()
    if isinstance(node, ast.AnnAssign):
        ret.name = node.target.id
        ret.type = node.annotation.id
    else:
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
def parse_stage_input(node : ast.ClassDef) -> pysl.StageInput:
    struct = pysl.StageInput()
    struct.name = node.name
    struct.elements = []

    conditions = []
    for decl_node in node.body:
        if isinstance(decl_node, ast.AnnAssign):
            assignment = parse_assignment(decl_node)
            element = pysl.InputElement()
            element.name = assignment.name
            element.type = str_to_pysl_type(assignment.type)
            element.semantic = assignment.value
            element.conditions = list(conditions)
            conditions[:] = []
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

def parse_sampler_state(name : str, slot : int, value : ast.AST) -> pysl.SamplerState:
    state = pysl.SamplerState()
    state.name = name

    if value:
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or not value.func.id == pysl.Keywords.SamplerStateConstructor:
            ERR(value, "Expected SamplerState constructor")
        else:
            for kw in value.keywords:
                if not isinstance(kw.value, ast.Name):
                    ERR(kw, "Expected name as value")
                    continue
                state.attributes.append((kw.arg, kw.value.id))

    return state

def parse_texture(name : str, ttype : str, slot : int, value : ast.AST) -> pysl.Texture:
    texture = pysl.Texture()
    texture.name = name
    texture.type = ttype[7:]

    if value:
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or not value.func.id[:7] == pysl.Keywords.TextureConstructor:
            ERR(value, "Expected Texture constructor")
        else:
            for kw in value.keywords:
                if not isinstance(kw.value, ast.Name):
                    ERR(kw, "Expected name as value")
                    continue
                texture.attributes.append((kw.arg, kw.value.id))

    return texture;

# Evalution code
#-------------------------------------------------------------------------------
# Creates a new evalution closure for delayed evaluation
# <3 http://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture-in-python
def PYSL_eval_closure(node):
    return lambda : PYSL_eval(node) 

# This is the core routines, it doesn't return anything, as stated earlier the
# translation is strictly single pass
def PYSL_eval(node : ast.AST):
    if not node:
        return
    elif isinstance(node, ast.Call):
        # Method call
        if isinstance(node.func, ast.Attribute):
            Translate.method_call(node.func.value.id, node.func.attr, [PYSL_eval_closure(n) for n in node.args])
        elif node.func.id in pysl.INTRINSICS:
            Translate.intrinsic(node.func.id, [PYSL_eval_closure(n) for n in node.args])
        elif node.func.id in pysl.CONSTRUCTORS:
            Translate.constructor(node.func.id, [PYSL_eval_closure(n) for n in node.args])
        elif node.func.id is '_':
            Translate.text(parse_preprocessor(node) + '\n\n')
        else: # User Routine
            Translate.text('{0}('.format(node.func.id))
            for i in range(len(node.args) - 1):
                PYSL_eval(node.args[i])
                Translate.text(', ')
            if node.args:
                PYSL_eval(node.args[-1])
            Translate.text(')')


    elif isinstance(node, ast.Attribute):
        PYSL_eval(node.value)
        Translate.text('.{0}'.format(node.attr))

    elif isinstance(node, ast.IfExp):
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
        PYSL_eval(node.left)
        Translate.text(' {0} '.format(binop_str(node.op)))
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
        ERR(node, "PYSL_eval : Unsupported expression")

def INDENT(indent : int):
    Translate.text('\t' * indent)

def PYSL_block(nodes : [ast.AST], indent : int):
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
            ERR(node, "Unsupported block node type")

# Top level resource declaration
def PYSL_tl_decl(node : ast.AnnAssign):
    res_name = node.target.id
    res_type = None
    res_slot = None

    if isinstance(node.annotation, ast.Call) and node.annotation.func.id == 'register' and len(node.annotation.args) == 2:
        if isinstance(node.annotation.args[0], ast.Name) and isinstance(node.annotation.args[1], ast.Num):
            res_type = node.annotation.args[0].id
            res_slot = node.annotation.args[1].n
        else:
            ERR(node, "Invalid register call, it should be register(Name, Num)") 
    elif isinstance(node.annotation, ast.Name):
        res_type = node.annotation.id
    else:
        ERR(node, "Unrecognized top-level declaration: {0}".format(res_name))
    
    if res_type == pysl.Keywords.SamplerStateConstructor:
        Translate.sampler_state(parse_sampler_state(res_name, res_slot, node.value))
    elif res_type[:7] == pysl.Keywords.TextureConstructor:
        Translate.texture(parse_texture(res_name, res_type, res_slot, node.value))
    else:
        ERR(node, "Unrecognized type: {0} for top-level declaration: {1}".format(res_type, res_name))

def PYSL_function_signature(func : ast.FunctionDef):
    if not isinstance(func.returns, ast.Name):
        ERR(func, "Expected return type for function: {0}".format(func.name))
        return

    if len(func.decorator_list) > 2:
        ERR(func, "Multiple decorators are not supported, only the first one will be processed")

    if func.decorator_list:
        if isinstance(func.decorator_list[0], ast.Name):
            Metadata.entry_point(func.decorator_list[0].id, func.name)
        else:
            ERR(func.decorator_list, "Expected name as decorator")

    Translate.text('{0} {1}('.format(func.returns.id, func.name))
    for arg in func.args.args[:-1]:
        if not isinstance(arg.annotation, ast.Name):
            ERR(arg, "Expected type for parameter: {0} of function: {1}".format(arg.arg, func.name))
        else:
            print(arg.value)
            Translate.text('{0} {1}, '.format(arg.annotation.id, arg.arg))

    if func.args.args:
        arg = func.args.args[-1];
        if not isinstance(arg.annotation, ast.Name):
            ERR(arg, "Expected type for parameter: {0} of function: {1}".format(arg.arg, func.name))
        else:
            Translate.text('{0} {1}'.format(arg.annotation.id, arg.arg))
    Translate.text(')\n{\n')        

def PYSL(path : str):
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
            # Stage input
            if not node.decorator_list:
                Translate.stage_input(parse_stage_input(node))
            else:
                decorator = parse_decorator(node.decorator_list[0])
                if decorator.name is pysl.Keywords.StageInputDecorator:
                    Translate.stage_input(parse_stage_input(node))
                elif decorator.name is pysl.Keywords.ConstantBufferDecorator:
                    Translate.constant_buffer(parse_constant_buffer(node))
                else:
                    ERR(node, "Unsupported decorator: {0}".format(decorator.name))
        elif isinstance(node, ast.AnnAssign):
            PYSL_tl_decl(node)

        elif isinstance(node, ast.FunctionDef):
            PYSL_function_signature(node)
            PYSL_block(node.body, 1)
            Translate.text('};\n\n')

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
            print(node)
            ERR(node, "PYSL : Unsupported expression")

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Please specify file to transpile")
        sys.exit()

    out_path = 'out'
    if (len(sys.argv) > 2):
        out_path = sys.argv[2]

    if not Translate.init(out_path + '.hlsl', out_path + '.glsl'):
        print('Failed to open destination HLSL/GLSL file, check permissions or paths')
        sys.exit()

    if not Metadata.init(out_path):
        print('Failed to open destination metadata file, check permissions or paths')
        sys.exit()
 
    PYSL(sys.argv[1])
    Metadata.finalize()