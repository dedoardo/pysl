import ast
import sys
import argparse
from . import pysl
from . import emitter
from . import exporter
from . import validator
from .error import error, info, init, get_status


def loc(node: ast.AST) -> pysl.Locationable:
    return pysl.Locationable(node)


# AST Utilities
# ------------------------------------------------------------------------------


def str_to_pysl_type(loc: pysl.Locationable, string: str) -> pysl.Type:
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
        error(loc, "Invalid type: {0}".format(string))
    if dims:
        ret.dim0 = int(dims[0:1])
    if dims[1:2] == 'x':
        ret.dim1 = int(dims[2:3])
    return ret


class Decorator:
    def __init__(self):
        self.name: str = None
        self.args: [str] = []


def eval_numeric_constexpr(node: ast.AST) -> int:
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


def parse_decorator(node: ast.AST):
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
                    error(loc(node), "Unsupported decorator type")
        return ret
    else:
        error(loc(node), "Supported decorators are Name and Call")
        return None


def parse_preprocessor(call: ast.Call) -> str:
    if not isinstance(call, ast.Call):
        error(loc(call), "Preprocessor takes the form of _('directive')")
        return ''
    if len(call.args) != 1:
        error(loc(call), "Preprocessor takes 1 argument in the of of _('directive')")
        return ''
    if not isinstance(call.args[0], ast.Str):
        error(loc(call), "Preprocessor requires arguments to be strings in the form of _('directive')")
    return call.args[0].s


def parse_options(call: ast.Call) -> [str]:
    if not isinstance(call, ast.Call):
        error(loc(call), "Option takes the form of option('option')")
        return ''

    ret = []
    for arg in call.args:
        if not isinstance(arg, ast.Str):
            error(loc(arg), "Expected string literal")
            continue
        ret.append(arg.s)
    return ret


def parse_attribute_no_eval(node: ast.Attribute) -> str:
    if isinstance(node.value, ast.Name):
        return node.value.id
    return parse_attribute_no_eval(node.value) + '.|{0}|'.format(node.attr)


def parse_assignment(node: ast.AST) -> pysl.Assignment:
    """Block level assignment, can either be a declaration or not"""
    if not isinstance(node, ast.Assign) and not isinstance(node, ast.AnnAssign):
        error(loc(node), "Expected Assignment")
        return

    ret = pysl.Assignment()
    ret.set_location(node)
    ret.qualifiers = []
    if isinstance(node, ast.AnnAssign):
        # Declaration
        ret.name = node.target.id
        if isinstance(node.annotation, ast.Name):
            ret.type = node.annotation.id
        elif (isinstance(node.annotation, ast.Attribute) and
              isinstance(node.annotation.value, ast.Name)):
            if node.annotation.value.id == pysl.Language.Qualifier.CONST:
                ret.qualifiers.append(pysl.Language.Qualifier.CONST)
            else:
                error(loc(node.annotation), "Unsupported qualifier, only const is supported in a block assignment/declaration")
            ret.type = node.annotation.attr

        else:
            error(loc(node.annotation), "Unsupported annotation, supported are <type> or const.<type>")
    else:
        # Assignment
        if isinstance(node.targets[0], ast.Name):
            ret.name = node.targets[0].id
        elif isinstance(node.targets[0], ast.Attribute):
            ret.name = parse_attribute_no_eval(node.targets[0])
        else:
            error(loc(node), "Unsupported assignment target")
    if node.value and isinstance(node.value, ast.Name):
        ret.value = node.value.id
    elif node.value and isinstance(node.value, ast.Num):
        ret.value = str(node.value.n)
    else:
        ret.value = None
    return ret


def TO_INT(node: ast) -> (bool, int):
    if isinstance(node, ast.Num):
        return (True, node.n)
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Num):
        if isinstance(node.op, ast.UAdd):
            return (True, +node.operand.n)
        if isinstance(node.op, ast.USub):
            return (True, -node.operand.n)
        error(loc(node), "Expected +/- Num")
        return (False, 0)
    error(loc(node), "Expected signed integer")
    return (False, 0)


def parse_for_range(node: ast.Call) -> (str, str, str):
    if len(node.args) != 3:
        error(loc(node), "Expected 3 integer arguments in range(start, end, stop) found: {0}".format(len(node.args)))
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


def unop_str(op: ast.AST) -> str:
    if isinstance(op, ast.UAdd):
        return '+'
    if isinstance(op, ast.USub):
        return '-'
    if isinstance(op, ast.Not):
        return '!'
    if isinstance(op, ast.Invert):
        return '~'
    error(loc(op), "Invalid unary operator encountered: {0}:{1}. Check supported intrinsics.".format(op.lineno, op.col_offset))
    return 'INVALID_UNOP'


def binop_str(op: ast.AST) -> str:
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
    error(loc(op), "Invalid binary operator encountered: {0}:{1}. Check supported intrinsics.".format(op.lineno, op.col_offset))
    return 'INVALID_BINOP'


def cmpop_str(op: ast.AST) -> str:
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
    error(loc(op), "Invalid compare operator encountered: {0}:{1}. Check supported intrisics.".format(op.lineno, op.col_offset))
    return 'INVALID_CMPOP'


# Parsers
# ------------------------------------------------------------------------------


def parse_struct(node: ast.ClassDef) -> pysl.Struct:
    struct = pysl.Struct()
    struct.set_location(node)
    struct.name = node.name
    struct.elements = []
    struct.set_location(node)

    for decl_node in node.body:
        if isinstance(decl_node, ast.AnnAssign):
            assignment = parse_assignment(decl_node)
            struct.elements.append((str_to_pysl_type(loc(node), assignment.type), assignment.name))
        else:
            error(loc(decl_node), "Unrecognized node inside structure: {0}".format(struct.name))

    return struct


def parse_stage_input(node: ast.ClassDef, stages: str) -> pysl.StageInput:
    si = pysl.StageInput()
    si.set_location(node)
    si.name = node.name
    si.elements = []
    si.stages = stages

    conditions = []
    for decl_node in node.body:
        if isinstance(decl_node, ast.AnnAssign):
            assignment = parse_assignment(decl_node)
            element = pysl.InputElement()
            element.set_location(decl_node)
            element.name = assignment.name
            element.type = str_to_pysl_type(loc(decl_node), assignment.type)
            element.semantic = assignment.value
            element.conditions = list(conditions)
            conditions[:] = []  # Copy
            si.elements.append(element)

        elif (isinstance(decl_node, ast.Expr) and
              isinstance(decl_node.value, ast.Call)):
            if decl_node.value.func.id is '_':
                conditions.append(parse_preprocessor(decl_node.value))
            else:
                error(loc(decl_node), "Unsupported function call: {0} inside StageInput: {1}".format(decl_node.value.func.id, si.name))
        else:
            error(loc(decl_node), "Unrecognized node inside StageInput: {0}".format(si.name))

    si.post_conditions = list(conditions)
    return si


def parse_constant_buffer(node: ast.ClassDef) -> pysl.ConstantBuffer:
    cbuffer = pysl.ConstantBuffer()
    cbuffer.set_location(node)
    cbuffer.name = node.name
    cbuffer.constants = []

    for decl_node in node.body:
        if isinstance(decl_node, ast.AnnAssign):
            assignment = parse_assignment(decl_node)
            constant = pysl.Constant()
            constant.set_location(decl_node)
            constant.name = assignment.name
            if assignment.value:
                try:
                    constant.offset = int(assignment.value)
                except ValueError:
                    error(loc(decl_node), "Expected numberic offset as argument")
            assignment.value
            constant.type = str_to_pysl_type(loc(decl_node), assignment.type)
            cbuffer.constants.append(constant)
        else:
            error(loc(decl_node), "Unrecognized node inside ConstantBuffer: {0}".format(cbuffer.name))

    for d in node.decorator_list:
        decorator = parse_decorator(d)
        if decorator.args:
            try:
                cbuffer.enforced_size = int(decorator.args[0])
            except ValueError:
                error(loc(d), "Expected integer argument to constructor indicating enforced size(in constants), but evaluated: {0}".format(decorator.args[0]))
    return cbuffer


def parse_sampler(loc: pysl.Locationable, name: str, type: str, slot: int, value: ast.AST) -> pysl.Sampler:
    sampler = pysl.Sampler()
    sampler.set_location(value)
    sampler.name = name
    sampler.type = type
    sampler.slot = slot

    if value:
        if (not isinstance(value, ast.Call) or
           not isinstance(value.func, ast.Name) or
           not value.func.id == pysl.Language.Export.KEYWORD):
            error(loc(value), "Expected export() expression")
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
                    error(loc(kw), "Unsupported export value, only literals or names are currently supported")
                    continue
                sampler.attributes.append((kw.arg, val))

    return sampler


# Evalution code
# ------------------------------------------------------------------------------


class EvalClosure:
    """
    # Creates a new evalution closure for delayed evaluation
    http://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture-in-python
    """
    def __init__(self, func, node):
        self.func = func
        self.node = node

    def __call__(self):
        PYSL_eval(self.func, self.node)


def PYSL_eval(func: pysl.Function, node: ast.AST):
    """Core routine, doesn't return anything but directly writes to output"""
    if not node:
        return
    elif isinstance(node, ast.Call):
        # Calls can be of different types
        # - Method call
        if isinstance(node.func, ast.Attribute):
            emitter.method_call(loc(node.func.value), node.func.value.id, node.func.attr,
                                [EvalClosure(func, n) for n in node.args])
        # - Intrinsics, that is any kind of built in function
        elif pysl.Language.Intrinsic.is_in(node.func.id):
            emitter.intrinsic(loc(node.func), node.func.id,
                              [EvalClosure(func, n) for n in node.args])
        # - Constructor, just a new type being declared
        elif pysl.Language.NativeType.is_in(node.func.id):
            emitter.constructor(node.func.id,
                                [EvalClosure(func, n) for n in node.args])
        # - '_' is the special code for the preprocessor, writes the string
        # contained in the braces as it is
        elif node.func.id is '_':
            emitter.text(parse_preprocessor(node) + '\n\n')
        # - As we didnt' recognize the name, it is probably a user's routine
        else:
            emitter.function_call(loc(node.func), node.func.id,
                                  [EvalClosure(func, n) for n in node.args])

    elif isinstance(node, ast.Attribute):
        # node.attr is the attribute name (string)
        if isinstance(node.value, ast.Name):
            if (node.value.id == pysl.Language.SpecialAttribute.INPUT or
               node.value.id == pysl.Language.SpecialAttribute.OUTPUT):
                emitter.special_attribute(loc(node), func, node.value.id, node.attr)
                return

        PYSL_eval(func, node.value)
        emitter.text('.{0}'.format(node.attr))

    elif isinstance(node, ast.IfExp):
        # Ternary if
        PYSL_eval(func, node.test)
        emitter.text(' ? ')
        PYSL_eval(func, node.body)
        emitter.text(' : ')
        PYSL_eval(func, node.orelse)

    elif isinstance(node, ast.UnaryOp):
        emitter.text(unop_str(node.op))
        emitter.text('(')
        PYSL_eval(func, node.operand)
        emitter.text(')')

    elif isinstance(node, ast.BinOp):
        # Checking if it's a cast operation
        op_str = binop_str(node.op)
        if op_str == '@':
            if not isinstance(node.left, ast.Name):
                error(loc(node.left), "Expected type to the left of the cast operator")
            elif not pysl.Language.NativeType.is_in(node.left.id):
                error(loc(node.left), "Invalid destination type: {0}".format(node.left.id))
            else:
                emitter.text('(')
                PYSL_eval(func, node.left)
                emitter.text(')')
                PYSL_eval(func, node.right)
        else:
            PYSL_eval(func, node.left)
            emitter.text(' {0} '.format(op_str))
            PYSL_eval(func, node.right)

    elif isinstance(node, ast.BoolOp):
        if len(node.values) > 2:
            error(loc(node), "Unsupported multiple consecutive boolean expressions")
            return
        emitter.text('(')
        PYSL_eval(func, node.values[0])
        emitter.text(')')
        if isinstance(node.op, ast.And):
            emitter.text(' && ')
        if isinstance(node.op, ast.Or):
            emitter.text(' || ')
        emitter.text('(')
        PYSL_eval(func, node.values[1])
        emitter.text(')')

    elif isinstance(node, ast.Set):
        emitter.text('{ ')
        for i in range(len(node.elts)):
            PYSL_eval(func, node.elts[i])
            if i != len(node.elts) - 1:
                emitter.text(', ')
        emitter.text(' }')

    elif isinstance(node, ast.Subscript):
        PYSL_eval(func, node.value)
        emitter.text('[')
        PYSL_eval(func, node.slice)
        emitter.text(']')

    elif isinstance(node, ast.Compare):
        if len(node.ops) > 1 or len(node.comparators) > 1:
            error(loc(node), "Unsupported multiple comparison operators")
            return
        PYSL_eval(func, node.left)
        emitter.text(cmpop_str(node.ops[0]))
        PYSL_eval(func, node.comparators[0])

    elif isinstance(node, ast.Num):
        emitter.text(str(node.n))

    elif isinstance(node, ast.Index):
        PYSL_eval(func, node.value)

    elif isinstance(node, ast.Str):
        emitter.text(node.s)

    elif isinstance(node, ast.Name):
        emitter.text(node.id)

    elif isinstance(node, ast.Expr):
        PYSL_eval(func, node.value)

    else:
        print(node)
        error(loc(node), "PYSL : Unsupported expression")


def INDENT(indent: int):
    emitter.text('\t' * indent)


def PYSL_block(func: pysl.Function, nodes: [ast.AST], indent: int):
    """Evaluates line by line all the instructions"""
    for node in nodes:
        if isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Assign):
            INDENT(indent)
            PYSL_eval(func, node.targets[0])
            emitter.text(' = ')
            PYSL_eval(func, node.value)
            emitter.text(';\n')

        elif isinstance(node, ast.AnnAssign):
            INDENT(indent)
            emitter.declaration(parse_assignment(node))
            if node.value:
                emitter.text(' = ')
                PYSL_eval(func, node.value)
            emitter.text(';\n')

        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if node.value.func.id is '_':
                    emitter.text(parse_preprocessor(node.value) + '\n')
            else:
                error(loc(node), "Unsupported block expression")

        elif isinstance(node, ast.AugAssign):
            INDENT(indent)
            PYSL_eval(func, node.target)
            emitter.text(' {0}= '.format(binop_str(node.op)))
            PYSL_eval(func, node.value)
            emitter.text(';\n')

        elif isinstance(node, ast.Return):
            INDENT(indent)
            emitter.text('return ')
            PYSL_eval(func, node.value)
            emitter.text(';\n')
        elif isinstance(node, ast.If):
            INDENT(indent)
            emitter.text('if (')
            PYSL_eval(func, node.test)
            emitter.text(')\n')
            INDENT(indent)
            emitter.text('{\n')
            PYSL_block(node.body, indent + 1)
            INDENT(indent)
            emitter.text('}\n')
            if node.orelse:
                INDENT(indent)
                emitter.text('else\n')
                INDENT(indent)
                emitter.text('{\n')
                PYSL_block(node.orelse, indent + 1)
                INDENT(indent)
                emitter.text('}\n')

        elif isinstance(node, ast.For):
            iters = []
            # Currently only support range-based loops
            # for i in range(start, end, step) <- signed
            # for (i : uint = start; i [<>] end; step)
            # if step is < 0 -> compare is > and viceversa
            if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call):
                val = node.iter
                if val.func.id != 'range' and val.func.id != 'rrange':
                    error(loc(val), "Currently only range-based loops are supported. range(start, end, step).")
                limits = parse_for_range(val)
                iters.append((node.target.id, limits[0], limits[1], limits[2], False if val.func.id == 'range' else True))

            elif isinstance(node.target, ast.Tuple):
                if isinstance(node.iter, ast.Tuple) and len(node.target.elts) == len(node.iter.elts):
                    for i in range(len(node.target.elts)):
                        target = node.target.elts[i]
                        vals = node.iter.elts[i]
                        if not isinstance(target, ast.Name) or not isinstance(vals, ast.Call):
                            error(loc(node), "Expected Name = Call in {0}-th assignment".format(i))
                            continue
                        if vals.func.id != 'range' and vals.func.id != 'rrange':
                            error((vals), "Currently only range-based loops are supported. range(start, end, step)")
                        limits = parse_for_range(vals)
                        iters.append((target.id, limits[0], limits[1], limits[2], False if vals.func.id == 'range' else True))

                else:
                    error(loc(node), "Expected tuple of same length in iter")
            else:
                error(loc(node), "Expected same number of Name, Tuple associations")

            INDENT(indent)
            emitter.text('for(')
            # Assignment
            for i in iters[:-1]:
                emitter.text('int {0} = {1}, '.format(i[0], i[1]))
            if iters:
                emitter.text('int {0} = {1};'.format(iters[-1][0], iters[-1][1]))

            # Bounds
            for i in iters[:-1]:
                op = '<' if i[4] is False else '>'
                emitter.text('{0} {1} {2}, '.format(i[0], op, i[2]))
            if iters:
                op = '<' if iters[-1][4] is False else '>'
                emitter.text('{0} {1} {2};'.format(iters[-1][0], op, iters[-1][2]))

            # Increments
            for i in iters[:-1]:
                op = '+' if i[4] is False else '-'
                emitter.text('{0} {1}= {2}, '.format(i[0], op, i[3]))
            if iters:
                op = '+' if iters[-1][4] is False else '-'
                emitter.text('{0} {1}= {2})\n'.format(iters[-1][0], op, iters[-1][3]))

            INDENT(indent)
            emitter.text('{\n')
            PYSL_block(node.body, indent + 1)
            INDENT(indent)
            emitter.text('}\n')

        elif isinstance(node, ast.While):
            INDENT(indent)
            emitter.text('while(')
            PYSL_eval(func, node.test)
            emitter.text(')\n')
            INDENT(indent)
            emitter.text('{\n')
            PYSL_block(node.body, indent + 1)
            INDENT(indent)
            emitter.text('}\n')

        elif isinstance(node, ast.Continue):
            INDENT(indent)
            emitter.text('continue;\n')

        elif isinstance(node, ast.Break):
            INDENT(indent)
            emitter.text('break;\n')

        else:
            error((node), "Unsupported block expression")


def PYSL_arg(func: ast.FunctionDef, arg: ast.arg) -> pysl.Declaration:
    if isinstance(arg.annotation, ast.Name):
        decl = pysl.Declaration(arg.annotation.id, arg.arg, [])
        decl.set_location(arg)
        return decl
    elif isinstance(arg.annotation, ast.Attribute) and isinstance(arg.annotation.value, ast.Name):
        if arg.annotation.value.id == pysl.Language.Qualifier.OUT:
            decl = pysl.Declaration(arg.annotation.attr, arg.arg, [pysl.Language.Qualifier.OUT])
            decl.set_location(arg)
    else:
        error(loc(arg), "Expected type for parameter: {0} of function: {1}".format(
              arg.arg, func.name))
        return pysl.Declaration(None, None, None)


def PYSL_function_signature(func: ast.FunctionDef):
    ret = pysl.Function()
    ret.set_location(func)
    ret.name = func.name
    ret.args = []

    if len(func.decorator_list) > 2:
        error((func), "Multiple decorators are not supported, only the first one will be processed")

    # Entry point or user-defined function ?
    if func.decorator_list:
        if isinstance(func.decorator_list[0], ast.Name):
            decorator = func.decorator_list[0].id
            if decorator in pysl.Language.Decorator.STAGES:
                ret.stage = decorator
            else:
                error(loc(func.decorator_list[0]), "Unknown decorator: {0}".format(func.decorator_list[0].id))
        else:
            error(loc(func.decorator_list), "Expected name as decorator")

    if isinstance(func.returns, ast.Name) and ret.stage:
        error(loc(func), "superfluous return type for entry point (already known): {0}".format(func.name))
    elif isinstance(func.returns, ast.Name):
        ret.return_value = func.returns.id

    # Parsing arguments
    for arg in func.args.args[:-1]:
        ret.args.append(PYSL_arg(func, arg))
    if func.args.args:
        ret.args.append(PYSL_arg(func, func.args.args[-1]))

    emitter.function_beg(ret)
    PYSL_block(ret, func.body, 1)
    emitter.function_end(ret)


def PYSL_tl_decl(node: ast.AnnAssign):
    """Parses a specific top-level declaration"""
    if not isinstance(node.annotation, ast.Call) or (node.annotation.func.id != 'register' or
       len(node.annotation.args) != 2 or
       not isinstance(node.annotation.args[0], ast.Name) or
       not isinstance(node.annotation.args[1], ast.Num)):
        error(loc(node), "Invalid top level resource declaration, see docs. <name> : register(<type>, <slot>) = (...)")

    res_name = node.target.id
    res_type = node.annotation.args[0].id
    res_slot = node.annotation.args[1].n

    if res_type in pysl.Language.Sampler.TYPES:
        emitter.sampler(parse_sampler(node, res_name, res_type[7:], res_slot, node.value))
    else:
        error((node), "Unrecognized top-level resource declaration {0} : {1}".format(res_name, res_type))


def PYSL(path: str):
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
                emitter.decl_struct(parse_struct(node))
            else:
                decorator = parse_decorator(node.decorator_list[0])
                if decorator.name == pysl.Language.Decorator.STAGE_INPUT:
                    stages = []
                    for arg in decorator.args:
                        if arg in pysl.Language.Decorator.STAGE_INPUTS:
                            stages.append(arg)
                        else:
                            error(loc(decorator), "Unrecognized stageinput decorator: {0}".format(arg))
                    emitter.decl_stage_input(parse_stage_input(node, stages))
                elif decorator.name == pysl.Language.Decorator.CONSTANT_BUFFER:
                    emitter.constant_buffer(parse_constant_buffer(node))
                else:
                    error(loc(node), "Unsupported decorator: {0}".format(decorator.name))
        elif isinstance(node, ast.AnnAssign):
            PYSL_tl_decl(node)

        elif isinstance(node, ast.FunctionDef):
            PYSL_function_signature(node)

        elif isinstance(node, ast.Assign):
            error(loc(node), "Encountered untyped assignment")

        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if node.value.func.id == '_':
                    emitter.text(parse_preprocessor(node.value) + '\n\n')
                if node.value.func.id == 'options':
                    emitter.options(parse_options(node.value))
                else:
                    error(loc(node), "Unsupported top-level function call")
            else:
                error((node), "Unsupported top-level expression")

        else:
            error(loc(node), "Unsupported top-level expression")


def main() -> int:
    if (len(sys.argv) < 2):
        print("Please specify file to transpile")
        sys.exit()

    parser = argparse.ArgumentParser(description='PYthon Shading Language compiler')
    parser.add_argument('output')
    parser.add_argument('-ohlsl5', type=str, action='store', default=None, help="HLSL destination path")
    parser.add_argument('-oglsl45', type=str, action='store', default=None, help="GLSL destination path")
    parser.add_argument('-ojson', type=str, action='store', default=None, help="JSON metadata destination path")
    parser.add_argument('-ohpp', type=str, action='store', default=None, help="C++ header destination path")
    parser.add_argument('-vhlsl5', type=str, action='store', default=None, help="HLSL compiler path for validation(fxc)")
    parser.add_argument('-vglsl45', type=str, action='store', default=None, help="GLSL compiler path for validation(glslLangValidator)")
    args = parser.parse_args()

    if not emitter.init(args.ohlsl5, args.oglsl45):
        print('Failed to open destination HLSL/GLSL file, check permissions or paths')
        sys.exit()

    if not exporter.init(args.ojson, args.ohpp):
        print('Failed to open destination metadata file, check permissions or paths')
        sys.exit()

    init(args.output)
    PYSL(args.output)
    exporter.finalize()
    emitter.finalize()

    if get_status() is False:
        error(None, 'Failed to compile: {0}, see above for more errors.'.format(args.output))
        return 1

    if args.vhlsl5 and not args.ohlsl5:
        error(None, "Requested HLSL validation, but no output was specified")
        return 1

    if args.vglsl45 and not args.oglsl45:
        error(None, "Requested GLSL validation, but no output was specified")
        return 1

    if args.vhlsl5:
        if not validator.validate_hlsl5(args.ohlsl5, args.vhlsl5):
            error(None, "Failed to validate HLSL shader, see above for errors")
            return 1

    if args.vglsl45:
        if not validator.validate_glsl45(args.oglsl45, args.vglsl45):
            error(None, "Failed to validate GLSL shader, see above for errors")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())